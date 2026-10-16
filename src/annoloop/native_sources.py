"""Pinned ST3D-related source weights; explicit download/import, no model launch."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
import zipfile
from .io import read,write,digest,inside,lock,now
from .settings import Settings
from .reused.range_zip import RemoteZip,google_url
from .reused.training_monitor import process_identity


def catalog():
    doc=read(Path(__file__).parent/'catalog/native_sources.json')
    seen=set()
    for m in doc['models']:
        if m['id'] in seen or not re.fullmatch(r'lsf-[a-z0-9-]+',m['id']):raise ValueError('Native source ID không hợp lệ')
        seen.add(m['id'])
        if m['architecture'] not in ('SECONDNetIoU','PVRCNN') or m['classes']!=['Vehicle']:
            raise ValueError('Native source chưa có bằng chứng class/profile được hỗ trợ')
        if not re.fullmatch(r'[a-f0-9]{64}',m['sha256']) or not 0<m['bytes']<300*1024**2:raise ValueError('Thiếu identity payload')
        profile=inside(Path(__file__).parent/'catalog',m['profile'])
        if digest(profile)!=m['profile_sha256']:raise ValueError('Native model profile đã thay đổi')
    return doc


def entry(identifier):
    model=next((m for m in catalog()['models'] if m['id']==identifier),None)
    if model is None:raise ValueError('Native source không có trong danh mục')
    return model


def root_for(data,identifier):
    entry(identifier)
    return inside(Path(data)/'native_model_cache',identifier)


def ready(data,model):
    root=root_for(data,model['id']);path=root/'weights.pth'
    return path.is_file() and path.stat().st_size==model['bytes'] and digest(path)==model['sha256']


def install(data,identifier,checkpoint=None,remote_factory=None):
    """Only known payload hashes can become class-verified source initializers."""
    model=entry(identifier);root=root_for(data,identifier)
    with lock(root/'download.lock',blocking=False):
        if not ready(data,model):
            temporary=root/('weights.'+uuid.uuid4().hex+'.download')
            write(root/'download.json',dict(status='downloading',started_at=now(),bytes_downloaded=0,total_bytes=model['bytes']))
            try:
                if checkpoint:
                    path=Path(checkpoint).expanduser()
                    if path.stat().st_size!=model['bytes'] or digest(path)!=model['sha256']:raise ValueError('Checkpoint không khớp payload đã pin')
                    shutil.copyfile(path,temporary)
                else:
                    remote=(remote_factory or (lambda m:RemoteZip(google_url(m['drive_id']))))(model)
                    with zipfile.ZipFile(remote) as archive:
                        info=archive.getinfo(model['member'])
                        if info.file_size!=model['bytes'] or info.flag_bits & 1:raise ValueError('ZIP member sai kích thước hoặc bị mã hóa')
                        count=0
                        with archive.open(info) as source,temporary.open('xb') as output:
                            while block:=source.read(1024*1024):
                                if (root/'cancel').exists():raise InterruptedError('Đã dừng tải; giữ artifacts')
                                count+=len(block)
                                if count>model['bytes']:raise ValueError('ZIP member vượt giới hạn')
                                output.write(block)
                                write(root/'download.json',dict(status='downloading',bytes_downloaded=count,total_bytes=model['bytes']))
                if temporary.stat().st_size!=model['bytes'] or digest(temporary)!=model['sha256']:raise ValueError('SHA256 checkpoint không khớp')
                temporary.replace(root/'weights.pth')
            except BaseException as error:
                write(root/'download.json',dict(status='failed',error=str(error),finished_at=now()))
                raise
        # Known payload identity + recorded source config/head evidence. No
        # inference claim, no Vehicle -> car/pedestrian mapping.
        metadata=dict(checkpoint=str(root/'weights.pth'),checkpoint_sha256=model['sha256'],
            classes=model['classes'],classes_verified=True,architecture=model['architecture'],
            namespace='lsf:waymo:Vehicle:source_only',source_dataset='Waymo',epoch=30,
            native_source_id=identifier,class_verification=dict(method='pinned_previously_inspected_payload',
                evidence=model['class_evidence'],member=model['member'],source_url=model['source_url']),
            runtime_verified=False,forward_backward_verified=False)
        write(root/'checkpoint.metadata.json',metadata)
        write(root/'download.json',dict(status='completed',bytes_downloaded=model['bytes'],total_bytes=model['bytes'],
            sha256=model['sha256'],finished_at=now(),runtime_verified=False))
        return root/'checkpoint.metadata.json'


def start_download(settings,identifier):
    root=root_for(settings.data,identifier)
    with lock(root/'launch.lock',blocking=False):
        with lock(root/'download.lock',blocking=False):pass
        old=read(root/'launch.json') if (root/'launch.json').exists() else {}
        if old.get('process_start') and process_identity(old.get('pid'))==old['process_start']:raise ValueError('Model đang tải')
        (root/'cancel').unlink(missing_ok=True)
        env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).parents[1])+os.pathsep+env.get('PYTHONPATH','')
        with (root/'worker.log').open('a') as log:
            process=subprocess.Popen([str(settings.python),'-m','annoloop.native_sources','download',identifier,'--data',str(settings.data)],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        write(root/'launch.json',dict(pid=process.pid,process_start=process_identity(process.pid),started_at=now()))
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['list','download','import'])
    parser.add_argument('model',nargs='?');parser.add_argument('--checkpoint',type=Path);parser.add_argument('--data',type=Path)
    args=parser.parse_args();data=args.data or Settings.load().data
    if args.action=='list':
        import json
        print(json.dumps(catalog(),ensure_ascii=False,indent=2));return
    if not args.model:parser.error('Cần native source ID')
    if args.action=='import' and not args.checkpoint:parser.error('Import cần --checkpoint')
    print(install(data,args.model,args.checkpoint if args.action=='import' else None))


if __name__=='__main__':main()
