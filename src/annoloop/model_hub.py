"""Class-aware remote model registry and resumable, non-executing model installer.

Download status, taxonomy evidence, and native runtime readiness are independent.
No torch/pickle import and no project/model selection changes on installation.
"""
import argparse
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from .io import digest, inside, lock, now, read, write
from .reused.training_monitor import process_identity

CATALOG=Path(__file__).parent/'catalog/models.json'
WEIGHT_HOSTS={'download.openmmlab.com','drive.usercontent.google.com'}
CONFIG_HOSTS={'raw.githubusercontent.com'}


def load_catalog(path=None):
    doc=read(path or CATALOG)
    ids=set()
    for m in doc['models']:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,100}',m['id']) or m['id'] in ids:
            raise ValueError('Registry có ID model không hợp lệ/trùng')
        ids.add(m['id'])
        validate_url(m['weights_url'],WEIGHT_HOSTS)
        if not m.get('classes') or not all(isinstance(c,str) and c.strip() for c in m['classes']):
            raise ValueError('Registry thiếu class đã xác minh')
        if m['download_verification']['status']!='binary_reachable':
            raise ValueError('Registry chứa checkpoint chưa kiểm tra binary')
        for relative, spec in m.get('config_files',{}).items():
            if PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts or '\\' in relative:
                raise ValueError('Config path không hợp lệ')
            validate_url(spec['url'],CONFIG_HOSTS)
    return doc


def validate_url(url, hosts):
    u=urllib.parse.urlsplit(url)
    if u.scheme!='https' or u.hostname not in hosts or u.username or u.password or u.port not in (None,443):
        raise ValueError('URL không thuộc kho HTTPS được hỗ trợ')


def canonical(name):
    # Case/spacing normalization only; never broaden Vehicle -> car/truck/bus.
    return re.sub(r'[\s_-]+','_',name.strip().casefold())


def recommendations(classes, *, dataset=None, architecture=None, modality='point_cloud', catalog=None):
    wanted=list(dict.fromkeys(classes))
    rows=[]
    for m in (catalog or load_catalog())['models']:
        if modality and m.get('modality','point_cloud')!=modality:continue
        if architecture and m['architecture']!=architecture:continue
        supported={canonical(c) for c in m['classes']}
        covered=[c for c in wanted if canonical(c) in supported]
        missing=[c for c in wanted if c not in covered]
        notes=[]
        if any(canonical(c) in {'car','truck','bus'} for c in missing) and 'vehicle' in supported:
            notes.append('Vehicle rộng hơn car/truck/bus; không thể tự tách class bằng đổi tên.')
        if 'person' in supported and any(canonical(c)=='pedestrian' for c in missing):
            notes.append('person → pedestrian cần xác nhận định nghĩa; chưa tính là hỗ trợ tự động.')
        rows.append(dict(m,covered_classes=covered,missing_classes=missing,mapping_notes=notes,
                         all_classes_supported=bool(wanted) and not missing,
                         quality_status='Chưa chấm trên dữ liệu của bạn'))
    # Published AP across different datasets is not a comparable ranking signal.
    return sorted(rows,key=lambda m:(-int(m['all_classes_supported']),-len(m['covered_classes']),
        -int(m['runtime_verified']),-int(bool(dataset) and str(m['dataset']).casefold()==dataset.casefold()),m['name']))


def model_root(data,identifier):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,100}',identifier):raise ValueError('ID model không hợp lệ')
    return inside(Path(data)/'model_cache',identifier)


def status(data,identifier):
    root=model_root(data,identifier)
    state=read(root/'download.json') if (root/'download.json').is_file() else dict(status='not_installed',bytes_downloaded=0)
    launch=read(root/'launch.json') if (root/'launch.json').is_file() else {}
    if launch.get('process_start') and process_identity(launch.get('pid'))==launch['process_start']:
        if state['status'] not in ('downloading',):
            return dict(state,status='queued')
    elif state['status']=='downloading' and state.get('process_start') and process_identity(state.get('pid'))!=state['process_start']:
        return dict(state,status='interrupted',error='Worker đã dừng; có thể tiếp tục phần tải dở.')
    elif launch and state['status']=='not_installed':
        return dict(state,status='failed',error='Worker đã dừng trước khi bắt đầu tải; xem worker.log trong folder model.')
    if state['status']=='completed' and not (root/'weights.pth').is_file():return dict(state,status='cache_invalid',error='Weights không còn trên đĩa')
    return state


def installed(data,model):
    root=model_root(data,model['id']);receipt=root/'installed.json'
    if not receipt.is_file():return False
    r=read(receipt)
    if r.get('weights_url')!=model['weights_url'] or not (root/'weights.pth').is_file():return False
    if digest(root/'weights.pth')!=r.get('sha256'):return False
    return all((root/'upstream'/relative).is_file() and digest(root/'upstream'/relative)==spec['sha256'] for relative,spec in model['config_files'].items())


def binary_header(chunk):
    return chunk.startswith((b'PK\x03\x04',b'\x80\x02',b'\x80\x03',b'\x80\x04',b'\x80\x05'))


def download(data,model, *, opener=urllib.request.urlopen, resume=False):
    """Atomic final install; partial bytes are retained on cancel/network error."""
    validate_url(model['weights_url'],WEIGHT_HOSTS)
    root=model_root(data,model['id']);root.mkdir(parents=True,exist_ok=True)
    with lock(root/'download.lock',blocking=False):
        if resume:(root/'cancel').unlink(missing_ok=True)
        if installed(data,model):
            write(root/'download.json',dict(status='completed',model_id=model['id'],cache_reused=True,finished_at=now()))
            return read(root/'installed.json')
        part=root/'weights.pth.part';partial=root/'partial.json';final=root/'weights.pth'
        progress=dict(model_id=model['id'],status='downloading',started_at=now(),bytes_downloaded=0,total_bytes=None,
                      pid=os.getpid(),process_start=process_identity(os.getpid()))
        write(root/'download.json',progress)
        try:
            old=read(partial) if partial.exists() else {}
            offset=part.stat().st_size if part.exists() and old.get('url')==model['weights_url'] else 0
            headers={'User-Agent':'AnnoLoop-model-hub/1.0'}
            validator=old.get('etag') or old.get('last_modified')
            # Resume only with a server validator; otherwise restart safely.
            if offset and validator:headers.update(Range=f'bytes={offset}-',**{'If-Range':validator})
            else:offset=0
            request=urllib.request.Request(model['weights_url'],headers=headers)
            try:
                response=opener(request,timeout=45)
            except urllib.error.HTTPError as error:
                if error.code!=416:raise
                # Complete .part + config-stage interruption: restart body if server
                # rejects the end-of-file range; never silently accept a 416 body.
                offset=0
                response=opener(urllib.request.Request(model['weights_url'],headers={'User-Agent':'AnnoLoop-model-hub/1.0'}),timeout=45)
            with response:
                validate_url(response.geturl(),WEIGHT_HOSTS)
                code=response.status
                if code==206:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=offset:raise ValueError('Server trả Content-Range không khớp phần tải dở')
                    total=int(match[3]);new_validator=response.headers.get('ETag') or response.headers.get('Last-Modified')
                    if offset and new_validator!=validator:raise ValueError('Phiên bản weights trên server thay đổi; không nối hai file khác nhau')
                elif code==200:
                    offset=0;total=int(response.headers.get('Content-Length','0')) or None
                else:raise ValueError(f'HTTP không hợp lệ: {code}')
                write(partial,dict(url=model['weights_url'],etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified')))
                progress.update(total_bytes=total,bytes_downloaded=offset)
                with part.open('ab' if offset else 'wb') as stream:
                    while True:
                        if (root/'cancel').exists():raise InterruptedError('Người dùng dừng tải; giữ phần đã tải để tiếp tục')
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        if progress['bytes_downloaded']==0 and not binary_header(chunk):raise ValueError('Link trả HTML/login hoặc dữ liệu không phải checkpoint')
                        stream.write(chunk);progress['bytes_downloaded']+=len(chunk)
                        if progress['bytes_downloaded']>20*1024**3:raise ValueError('Checkpoint vượt giới hạn 20 GiB')
                        write(root/'download.json',progress)
                if total and progress['bytes_downloaded']!=total:raise ValueError('Weights tải chưa đủ dung lượng')
            if part.stat().st_size<4096:raise ValueError('Weights quá ngắn')
            with part.open('rb') as stream:prefix=stream.read(4096)
            if not binary_header(prefix):raise ValueError('Header checkpoint không hợp lệ')
            verified=model['download_verification']
            if verified.get('prefix_sha256') and __import__('hashlib').sha256(prefix[:verified['prefix_bytes']]).hexdigest()!=verified['prefix_sha256']:
                raise ValueError('Nội dung weights khác bản đã kiểm tra trong registry; cần cập nhật danh mục')
            sha=digest(part)
            if model.get('publisher_sha256') and sha!=model['publisher_sha256']:raise ValueError('SHA256 weights không khớp nhà phát hành')
            for relative,spec in model['config_files'].items():
                validate_url(spec['url'],CONFIG_HOSTS)
                path=inside(root/'upstream',relative);path.parent.mkdir(parents=True,exist_ok=True)
                with opener(urllib.request.Request(spec['url'],headers={'User-Agent':'AnnoLoop-model-hub/1.0'}),timeout=30) as response:
                    validate_url(response.geturl(),CONFIG_HOSTS);body=response.read(2*1024*1024)
                if __import__('hashlib').sha256(body).hexdigest()!=spec['sha256']:raise ValueError('SHA256 config không khớp phiên bản đã pin')
                path.write_bytes(body)
            part.replace(final)
            receipt=dict(model_id=model['id'],weights=str(final),weights_url=model['weights_url'],sha256=sha,
                publisher_sha256_available=bool(model.get('publisher_sha256')),local_integrity_verified=True,
                bytes=final.stat().st_size,classes=model['classes'],architecture=model['architecture'],
                framework=model['framework'],revision=model['revision'],source_url=model['source_url'],
                runtime_verified=False,runtime_status=model['runtime_status'],config_path=str(root/'upstream'/model['config_path']),
                installed_at=now(),note='Local SHA256 verifies cached bytes; it is not publisher attestation or GPU compatibility verification.')
            write(root/'installed.json',receipt)
            progress.update(status='completed',finished_at=now(),sha256=sha)
            write(root/'download.json',progress)
            return receipt
        except BaseException as error:
            progress.update(status='paused' if isinstance(error,InterruptedError) else 'failed',error=str(error),finished_at=now())
            write(root/'download.json',progress)
            raise


def start_download(settings,identifier, *, prepare=False):
    model=next(m for m in load_catalog()['models'] if m['id']==identifier)
    if prepare:
        from .adapter_factory import recipe_for
        recipe_for(model)
    root=model_root(settings.data,identifier);root.mkdir(parents=True,exist_ok=True)
    with lock(root/'launch.lock',blocking=False):
        # Probe the worker lock, independent of stale JSON/PIDs after an app restart.
        try:
            with lock(root/'download.lock',blocking=False):pass
        except BlockingIOError:raise ValueError('Model này đang tải')
        launch=read(root/'launch.json') if (root/'launch.json').exists() else {}
        if launch.get('process_start') and process_identity(launch.get('pid'))==launch['process_start']:
            raise ValueError('Model đang chờ worker tải')
        (root/'cancel').unlink(missing_ok=True)
        env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).parents[1])+os.pathsep+env.get('PYTHONPATH','')
        with (root/'worker.log').open('a') as log:
            process=subprocess.Popen([str(settings.python),'-m','annoloop.model_hub','setup' if prepare else 'download',identifier,'--data',str(settings.data)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        # Worker updates its own receipt. Parent stores launch separately to avoid overwriting it.
        write(root/'launch.json',dict(pid=process.pid,process_start=process_identity(process.pid),started_at=now(),model_id=model['id']))
        return root


def cancel_download(data,identifier):
    root=model_root(data,identifier);root.mkdir(parents=True,exist_ok=True)
    (root/'cancel').touch()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['list','download','setup'])
    p.add_argument('model',nargs='?');p.add_argument('--data',type=Path);p.add_argument('--classes',nargs='*',default=[])
    a=p.parse_args()
    if a.action=='list':
        import json
        print(json.dumps([dict(id=m['id'],name=m['name'],classes=m['classes'],missing=m['missing_classes'],runtime_status=m['runtime_status']) for m in recommendations(a.classes)],ensure_ascii=False,indent=2))
    else:
        if not a.model:p.error('Cần ID model')
        from .settings import Settings
        data=a.data or Settings.load().data
        model=next((m for m in load_catalog()['models'] if m['id']==a.model),None)
        if model is None:p.error('Model không có trong registry')
        if a.action=='setup':
            from .adapter_factory import select_and_prepare
            profile=select_and_prepare(data,model);print(profile['status'],profile['checkpoint'])
        else:
            receipt=download(data,model,resume=True);print(receipt['weights'],receipt['sha256'])
            # Post-download connection uses installed files only. Unsupported
            # frameworks keep a report without turning a valid download into
            # an invented executable adapter.
            from .connect_model import connect
            report=connect(data,model)
            print('Connection:',report['status'],report['receipt'])


if __name__=='__main__':main()
