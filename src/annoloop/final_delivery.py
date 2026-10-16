"""Immutable final-prediction jobs and explicit CVAT delivery, never training."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import os
import subprocess
import uuid
from dataclasses import asdict
from .io import read, write, digest, now, lock, file_signature
from .settings import ROOT
from .ensemble_fusion import EnsemblePolicy
from . import local_ensemble
from .reused.training_monitor import process_identity

SCOPES = {'non_holdout': {'adaptation', 'delivery'}, 'delivery': {'delivery'},
          'all': {'adaptation', 'delivery', 'holdout'}}


def jobs(store, pid):
    return sorted((store.path(pid) / 'delivery_jobs').glob('*/state.json'),
                  key=lambda p:p.stat().st_mtime, reverse=True)


def active(store, pid):
    return any(local_ensemble.status(p.parent)['process_alive'] for p in jobs(store,pid))


def plan(store, pid, scope='non_holdout', ensemble_job=None):
    """Read-only plan; selected checkpoints, not the source initializer or labels."""
    if scope not in SCOPES:
        raise ValueError('Phạm vi bàn giao không hợp lệ')
    root=store.path(pid);project=store.get(pid)
    if project['schema']['data_type']!='3d':
        raise ValueError('Chưa có adapter final inference 2D')
    inventory=read(root/'dataset.json')
    frames=[f for f in inventory['frames'] if f['split'] in SCOPES[scope]]
    if not frames:
        raise ValueError('Phạm vi đã chọn không có frame')
    ensemble=local_ensemble.configuration(store,pid)
    source_snapshot=None
    if ensemble.get('enabled'):
        candidates=[p.parent for p in local_ensemble.jobs(store,pid)
                    if local_ensemble.status(p.parent)['status']=='completed' and (p.parent/'result.json').is_file()]
        chosen=Path(ensemble_job) if ensemble_job else (candidates[0] if candidates else None)
        if chosen not in candidates:
            raise ValueError('Cần một lượt ghép local hoàn tất trước khi bàn giao')
        saved=read(chosen/'spec.json')
        config=deepcopy(saved['config']);version='ensemble/'+chosen.name
        source_snapshot=dict(path=str(chosen/'spec.json'),sha256=digest(chosen/'spec.json'))
    else:
        version=project['selected_version']
        if not version or version not in project['completed_rounds']:
            raise ValueError('Chọn một vòng hoàn tất ở So sánh phiên bản trước khi final predict')
        metadata=root/'rounds'/version/'round.json';snapshot=read(metadata)
        adaptation=read(root/'adaptation.json')
        models=[]
        for key in ('A','B'):
            model=deepcopy(adaptation['models'][key])
            if model.get('backend','native_st3d')!='native_st3d':
                raise ValueError('Final adapter chưa hỗ trợ framework này')
            reference=snapshot['checkpoints'][key]
            model.update(path=reference['path'],checkpoint_sha256=reference['sha256'],vote_group=model['architecture'])
            models.append(model)
        config=dict(models=models,classes=snapshot['metrics'].get('adaptation_classes',
                    adaptation.get('adaptation_classes',project['schema']['classes'])),
                    bin_columns=adaptation['bin_columns'],coordinates_confirmed=adaptation['coordinates_confirmed'],
                    policy=asdict(EnsemblePolicy(match_iou=.3)),fusion_mode='ab')
        source_snapshot=dict(path=str(metadata),sha256=digest(metadata))
    if not config.get('coordinates_confirmed'):
        raise ValueError('Xác nhận đơn vị mét và hệ trục XYZ ở bước chuẩn bị model')
    if not config.get('classes') or not set(config['classes']).issubset(project['schema']['classes']):
        raise ValueError('Class bàn giao không khớp schema dự án')
    if config.get('bin_columns') not in (3,4,5):
        raise ValueError('Thiếu số cột BIN đã xác nhận')
    return dict(project_id=pid,purpose='final_delivery',version=version,scope=scope,
                frame_splits=sorted(SCOPES[scope]),frame_ids=[f['frame_id'] for f in frames],
                source_snapshot=source_snapshot,config=config,
                source_evidence=read(root/'setup.json').get('source_evidence',{}),
                dataset_sha256=project['dataset_sha256'])


def start(store,pid,settings,scope='non_holdout',ensemble_job=None):
    from .adaptation import active_job
    from .runtime import require_runtime,worker_environment
    from .device import gpu_idle
    with lock(store.path(pid)/'.adaptation_dispatch.lock'):
        if active(store,pid) or active_job(store,pid) or local_ensemble.active(store,pid):
            raise ValueError('Chờ job dự án hiện tại dừng trước khi final predict')
        spec=plan(store,pid,scope,ensemble_job)
        store.verify_dataset(pid)
        if not spec['source_evidence']:
            raise ValueError('Thiếu source validation để calibration checkpoint cuối')
        for item in spec['source_evidence'].values():
            if digest(item['path'])!=item['sha256']:
                raise ValueError('Source infos thay đổi')
        for model in spec['config']['models']:
            if digest(model['path'])!=model['checkpoint_sha256'] or digest(model['config'])!=model['config_sha256']:
                raise ValueError('Checkpoint vòng đã chọn hoặc config đã thay đổi')
            require_runtime(model,settings)
        with lock(settings.data/'.gpu.lock',blocking=False):
            gpu_idle()
            root=store.path(pid)/'delivery_jobs'/uuid.uuid4().hex
            write(root/'spec.json',spec)
            write(root/'state.json',dict(status='starting',stage='queued',created_at=now()))
            env=dict(worker_environment(settings),ANNOLOOP_DATA_DIR=str(settings.data),
                     VINAI_PYTHON=str(settings.python),ANNOLOOP_WORKER_PYTHON=str(settings.trainer_python))
            with (root/'worker.log').open('w') as log:
                process=subprocess.Popen([str(settings.python),'-u','-m','annoloop.ensemble_worker','--run',str(root)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            write(root/'state.json',dict(status='starting',stage='queued',created_at=now(),
                  pid=process.pid,process_start=process_identity(process.pid)))
    return root


def verified_export(root,kind='shape'):
    if kind!='shape':
        raise ValueError('Bản final delivery hiện xuất shape; không tạo track giả khi thiếu pose/timestamp')
    root=Path(root)
    state=local_ensemble.status(root)
    if state['status']!='completed':
        raise ValueError('Final predict chưa hoàn tất')
    if digest(root/'result.json')!=state.get('result_sha256'):
        raise ValueError('Kết quả final đã thay đổi')
    result=read(root/'result.json')
    if result.get('purpose')!='final_delivery':
        raise ValueError('Snapshot adaptation không phải final delivery')
    receipt=read(root/'exports'/kind/'export.json')
    directory=root/'exports'/kind
    if digest(directory/'export.json')!=result['export_manifest_sha256']:
        raise ValueError('Manifest bàn giao đã thay đổi')
    for name,identity in receipt['files'].items():
        if digest(directory/name)!=identity:
            raise ValueError('Gói bàn giao đã thay đổi: '+name)
    document=read(directory/'annotations.json')
    if document['frame_ids']!=result['frame_ids']:
        raise ValueError('Thứ tự frame nhãn final bị thay đổi')
    return directory,document


def _export_signature(root,kind):
    root=Path(root);directory=root/'exports'/kind
    paths=[root/'state.json',root/'result.json',directory/'export.json',
           directory/'media.zip',directory/'media.json',directory/'annotations.json']
    return tuple(file_signature(path) for path in paths)


@lru_cache(maxsize=8)
def _preview_export(root,kind,signature):
    directory,document=verified_export(root,kind)
    if _export_signature(root,kind)!=signature:
        raise ValueError('Gói bàn giao thay đổi trong lúc kiểm tra; tải lại trang')
    return directory,document


def preview_export(root,kind='shape'):
    """Read-only UI cache; upload always calls full verified_export separately."""
    if kind!='shape':return verified_export(root,kind)
    directory,document=_preview_export(str(Path(root).resolve()),kind,_export_signature(root,kind))
    return directory,deepcopy(document)


def upload(root,host,username,password,name):
    from .cvat import upload as send
    directory,document=verified_export(root)
    if not host or not username or not password or not name.strip():
        raise ValueError('Điền địa chỉ CVAT, tài khoản, mật khẩu và tên task')
    return send(directory,document,host,username,password,name.strip())


def prepare_export(root,inventory,documents,schema,bin_columns):
    """CPU media packing called only after complete real final predictions."""
    from .cvat import package,export_media
    from .inspection import load_xyz
    import zipfile
    root=Path(root);directory=root/'exports/shape'
    frame_ids=[d['frame_id'] for d in documents]
    export_media(inventory,frame_ids,directory,lambda f:load_xyz(f['media'],bin_columns))
    write(directory/'annotations.json',package(documents,schema,'shape'))
    # Bundle is transport for the AnnoLoop uploader, not an invented CVAT importer.
    names=['media.zip','media.json','annotations.json']
    write(directory/'export.json',dict(frame_ids=frame_ids,kind='shape',
          files={name:digest(directory/name) for name in names},target_gold_read=False))
    with zipfile.ZipFile(directory/'annoloop_delivery.zip','w',zipfile.ZIP_STORED) as archive:
        for name in names+['export.json']:
            archive.write(directory/name,name)
    return directory
