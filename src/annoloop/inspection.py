"""Checkpoint inspection/evaluation jobs; never starts or resumes training."""
import os
from pathlib import Path
import subprocess
import uuid
import math
import numpy as np
import yaml
from .device import gpu_idle
from .io import read,write,digest,now,lock
from .runtime import worker_environment,require_runtime
from .reused.training_monitor import process_identity
from .settings import ROOT


def checkpoint_choices(store,pid,settings):
    choices=[]
    for path in sorted((settings.data/'jobs').glob('*/job.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        job=read(path);config=Path(job['spec']['config'])
        if not config.is_relative_to(store.path(pid)):continue
        cfg=yaml.safe_load(config.read_text())
        for weight in sorted((path.parent/'checkpoints').glob('checkpoint_epoch_*.pth'),reverse=True):
            choices.append(dict(path=str(weight),config=str(config),runtime=job['spec']['runtime'],
                name=f"{cfg['MODEL']['NAME']} · {weight.stem.replace('checkpoint_epoch_', 'epoch ')}",
                classes=cfg['CLASS_NAMES'],origin_run=str(path.parent),intermediate=not (path.parent/'checkpoint.metadata.json').is_file()))
    return choices


def matched_counts(iou,scores,threshold):
    iou=np.asarray(iou);scores=np.asarray(scores)
    if iou.ndim!=2 or iou.shape[0]!=len(scores):raise ValueError('IoU shape không khớp prediction')
    used=set();tp=0
    for index in np.argsort(-scores,kind='stable'):
        candidates=np.argsort(-iou[index],kind='stable')
        for target in candidates:
            if iou[index,target]<threshold:break
            if int(target) not in used:used.add(int(target));tp+=1;break
    return dict(tp=tp,fp=len(scores)-tp,fn=iou.shape[1]-tp)


def load_xyz(path,bin_columns=None):
    path=Path(path)
    if path.suffix.lower()=='.npy':points=np.load(path,allow_pickle=False)
    elif path.suffix.lower()=='.pcd':
        from pypcd4 import PointCloud
        points=PointCloud.from_path(str(path)).numpy(('x','y','z'))
    elif path.suffix.lower()=='.bin':
        from .point_format import binary_contract
        contract=binary_contract(dict(media=str(path)))
        names=contract['feature_names'] if contract else None
        if names:
            if bin_columns is not None and bin_columns!=len(names):
                raise ValueError('Số cột BIN đã chọn không khớp khai báo dataset: '+str(len(names)))
            bin_columns=len(names)
        if bin_columns not in (3,4,5):raise ValueError('BIN cần xác nhận 3/4/5 cột và ý nghĩa feature')
        raw=np.fromfile(path,dtype='<f4')
        if len(raw)%bin_columns:raise ValueError('BIN không khớp số cột đã chọn')
        points=raw.reshape(-1,bin_columns)
        if names:points=points[:,[names.index(axis) for axis in ('x','y','z')]]
    else:raise ValueError('Media chưa hỗ trợ preview 3D')
    if points.ndim!=2 or points.shape[1]<3 or not len(points):raise ValueError('Point cloud cần N×3 hoặc hơn')
    xyz=np.asarray(points[:,:3],dtype=np.float32)
    if not np.isfinite(xyz).all():raise ValueError('XYZ chứa NaN/Inf')
    return xyz


def start_inspection(store,pid,choice,mode,settings,score=.3,iou=.5,coordinates_confirmed=False,bin_columns=4):
    if mode not in ('validation','target_sample'):raise ValueError('Mode inspection không hợp lệ')
    if not math.isfinite(score) or not 0<=score<=1 or not math.isfinite(iou) or not 0<iou<=1:raise ValueError('Ngưỡng không hợp lệ')
    available={c['path']:c for c in checkpoint_choices(store,pid,settings)}
    if choice not in available:raise ValueError('Checkpoint không thuộc dự án')
    candidate=available[choice]
    if mode=='target_sample' and not coordinates_confirmed:raise ValueError('Xác nhận đơn vị và trục point cloud trước khi predict target')
    inventory=store.verify_dataset(pid)
    require_runtime(candidate,settings)
    gpu_idle()
    parent=store.path(pid)/'inspections'
    with lock(settings.data/'.gpu.lock',blocking=False):
        for path in settings.data.glob('projects/*/inspections/*/inspection.json'):
            state=read(path)
            if state.get('process_start') and process_identity(state.get('pid'))==state['process_start']:
                raise ValueError('Một job đánh giá/predict đang chạy; chờ hoàn tất')
        root=parent/uuid.uuid4().hex;root.mkdir(parents=True)
        sample_ids=set(store.get(pid)['sample_frame_ids'])
        frames=[f for f in inventory['frames'] if f['frame_id'] in sample_ids]
        spec=dict(**candidate,mode=mode,score_threshold=score,iou_threshold=iou,frames=frames,bin_columns=bin_columns,
                  coordinates_confirmed=coordinates_confirmed,checkpoint_sha256=digest(choice),config_sha256=digest(candidate['config']))
        write(root/'spec.json',spec)
        write(root/'inspection.json',dict(status='starting',mode=mode,created_at=now(),checkpoint=choice))
        with (root/'inspection.log').open('w') as stream:
            process=subprocess.Popen([str(settings.trainer_python),'-u','-m','annoloop.inspection_worker','--run',str(root)],
                cwd=ROOT,env={**worker_environment(settings),'ANNOLOOP_DATA_DIR':str(settings.data)},stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        state=read(root/'inspection.json');state.update(pid=process.pid,process_start=process_identity(process.pid));write(root/'inspection.json',state)
        return str(root)


def inspection_status(root):
    state=read(Path(root)/'inspection.json')
    alive=bool(state.get('process_start') and process_identity(state.get('pid'))==state['process_start'])
    if state['status'] in ('starting','running') and not alive:state={**state,'status':'interrupted'}
    return dict(state,process_alive=alive)
