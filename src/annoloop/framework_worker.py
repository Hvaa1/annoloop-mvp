"""Deferred framework inference worker. Imported only after explicit predict.

No optimizer, target annotations, invented features, or fabricated IoU-head score.
Framework runtime must already be installed in the selected model environment.
"""
from pathlib import Path
import os
import sys
import numpy as np
from .io import read,write,digest,now


def points_for(frame,fields,bin_columns=None):
    path=Path(frame['media'])
    if digest(path)!=frame['sha256']:raise ValueError('Point cloud đã thay đổi')
    if path.suffix.lower()=='.pcd':
        from pypcd4 import PointCloud
        try:array=PointCloud.from_path(str(path)).numpy(tuple(fields))
        except (ValueError,KeyError) as e:raise ValueError('PCD thiếu feature thật: '+str(fields)) from e
    else:
        names=frame.get('feature_names')
        if path.suffix.lower()=='.bin':
            from .point_format import binary_contract
            contract=binary_contract(frame)
            if contract:
                names=contract['feature_names']
                if bin_columns is not None and bin_columns!=len(names):
                    raise ValueError('Số cột BIN không khớp khai báo dataset')
                bin_columns=len(names)
        array=np.load(path,allow_pickle=False) if path.suffix.lower()=='.npy' else np.fromfile(path,dtype='<f4').reshape(-1,bin_columns) if path.suffix.lower()=='.bin' and bin_columns else None
        if array is None:raise ValueError('Cần NPY/PCD hoặc BIN đã khai báo số cột')
        if names is None and array.ndim==2 and array.shape[1]==3:names=['x','y','z']
        if not names or array.ndim!=2 or len(names)!=array.shape[1] or len(set(names))!=len(names):
            raise ValueError('NPY/BIN cần feature_names đúng thứ tự cột; không đoán intensity/time')
        if not set(fields)<=set(names):raise ValueError('Input thiếu feature model cần: '+str(set(fields)-set(names)))
        array=array[:,[names.index(f) for f in fields]]
    if array.ndim!=2 or not len(array) or array.shape[1]!=len(fields) or not np.isfinite(array).all():
        raise ValueError('Point cloud sai shape hoặc NaN/Inf')
    return np.asarray(array,dtype=np.float32)


def boxes_document(centers,dims,yaws,scores,labels,classes, *, label_base=0,frame_id,coordinate_frame):
    arrays=[np.asarray(v) for v in (centers,dims,yaws,scores,labels)]
    centers,dims,yaws,scores,labels=arrays
    if centers.shape!=(len(scores),3) or dims.shape!=centers.shape or len(yaws)!=len(scores) or len(labels)!=len(scores):
        raise ValueError('Framework output shape không khớp')
    if any(not np.isfinite(a).all() for a in arrays) or (dims<=0).any() or (scores<0).any() or (scores>1).any():
        raise ValueError('Framework output không hợp lệ')
    boxes=[]
    for i,(center,size,yaw,score,label) in enumerate(zip(centers,dims,yaws,scores,labels)):
        index=int(label)-label_base
        if label!=int(label) or not 0<=index<len(classes):raise ValueError('Label index ngoài taxonomy')
        boxes.append(dict(id=f'det_{i:05d}',class_name=classes[index],box=[*center.tolist(),*size.tolist(),float(yaw)],
                          score=float(score),origin='framework_checkpoint_inference',score_semantics='native_detection_score'))
    return dict(frame_id=frame_id,boxes=boxes,coordinate_frame=coordinate_frame,
                box_origin='geometric_center',target_annotations_read=False,adaptation_ready=False)


def openpcdet(profile,request):
    variable=profile['recipe']['runtime_variable'];runtime=os.getenv(variable)
    if not runtime or not Path(runtime,'pcdet').is_dir():raise ValueError(f'Cần runtime OpenPCDet riêng: {variable}. Không dùng ngầm ST3D fork.')
    sys.path.insert(0,str(Path(runtime).resolve()))
    from pcdet.config import cfg,cfg_from_yaml_file
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network,load_data_to_gpu
    import torch
    # Official YAML _BASE_CONFIG_ paths are relative to the upstream tools folder.
    os.chdir(Path(profile['upstream_root'])/'tools')
    cfg_from_yaml_file(profile['config'],cfg)
    if list(cfg.CLASS_NAMES)!=profile['classes']:raise ValueError('Config class order khác registry')
    if cfg.DATA_CONFIG.get('MAX_SWEEPS',1)>1:raise ValueError('Model cần multi-sweep thật; adapter chưa nhận sweep/pose manifest')
    fields=list(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list)
    dataset=DatasetTemplate(cfg.DATA_CONFIG,cfg.CLASS_NAMES,False,Path(request['frames'][0]['media']).parent)
    model=build_network(cfg.MODEL,len(cfg.CLASS_NAMES),dataset)
    state=torch.load(profile['checkpoint'],map_location='cpu')['model_state']
    if any(not torch.isfinite(v).all() for v in state.values() if torch.is_floating_point(v)):raise ValueError('Weights NaN/Inf')
    _,loaded=model._load_state_dict(state,strict=False)
    missing=[k for k in model.state_dict() if k not in loaded and not k.endswith('num_batches_tracked')]
    if missing:raise ValueError('Strict weights coverage không đủ: '+str(missing[:8]))
    if not request['device'].startswith('cuda'):raise ValueError('OpenPCDet adapter này cần CUDA')
    torch.cuda.set_device(request['device']);model.to(request['device']).eval()
    shift=np.asarray(cfg.DATA_CONFIG.get('SHIFT_COOR',[0,0,0]),dtype=np.float32)
    for frame in request['frames']:
        points=points_for(frame,fields,request.get('bin_columns'));points[:,:3]+=shift
        item=dataset.prepare_data(dict(points=points,frame_id=frame['frame_id']))
        batch=dataset.collate_batch([item]);load_data_to_gpu(batch)
        with torch.no_grad():pred=model(batch)[0][0]
        boxes=pred['pred_boxes'].detach().cpu().numpy();boxes[:,:3]-=shift
        yield boxes_document(boxes[:,:3],boxes[:,3:6],boxes[:,6],pred['pred_scores'].cpu().numpy(),pred['pred_labels'].cpu().numpy(),
                             profile['classes'],label_base=1,frame_id=frame['frame_id'],coordinate_frame='LIDAR')


def mmdetection3d(profile,request):
    from mmengine.config import Config
    from mmdet3d.apis import init_model,inference_detector
    import torch
    config=Config.fromfile(profile['config']);dataset=config.test_dataloader.dataset
    while dataset.get('dataset'):dataset=dataset.dataset
    pipeline=dataset.pipeline
    def stages(rows):
        for row in rows:
            yield row
            yield from stages(row.get('transforms',[]))
    loaders=[r for r in stages(pipeline) if r['type'].endswith('LoadPointsFromFile')]
    if len(loaders)!=1:raise ValueError('Chưa nhận dạng được point loader của model')
    if any('MultiSweeps' in r['type'] for r in stages(pipeline)):
        raise ValueError('Model cần multi-sweep/time/ego pose; chưa thể dùng một frame thay thế')
    loader=loaders[0];load_dim=loader.get('load_dim',6)
    fields=request['frames'][0].get('feature_names')
    if fields is None and Path(request['frames'][0]['media']).suffix.lower()=='.bin':
        from .point_format import binary_contract
        contract=binary_contract(request['frames'][0])
        if contract:fields=contract['feature_names']
    if fields is None and load_dim==3:fields=['x','y','z']
    if not fields or len(fields)!=load_dim:raise ValueError('Cần khai báo feature_names đủ '+str(load_dim)+' cột của model')
    model=init_model(config,profile['checkpoint'],device=request['device'])
    if list(model.dataset_meta['classes'])!=profile['classes']:raise ValueError('Checkpoint class metadata khác registry')
    # Upstream init_model may load non-strictly. Check full parameter coverage ourselves.
    state=torch.load(profile['checkpoint'],map_location='cpu')['state_dict']
    state={k.removeprefix('module.'):v for k,v in state.items()}
    if any(not torch.isfinite(v).all() for v in state.values() if torch.is_floating_point(v)):raise ValueError('Weights NaN/Inf')
    missing=[k for k,v in model.state_dict().items() if (k not in state or state[k].shape!=v.shape) and not k.endswith('num_batches_tracked')]
    if missing:raise ValueError('Strict weights coverage không đủ: '+str(missing[:8]))
    for frame in request['frames']:
        points=points_for(frame,fields,request.get('bin_columns'))
        result=inference_detector(model,points)
        sample=result[0] if isinstance(result,tuple) else result
        pred=sample.pred_instances_3d;boxes=pred.bboxes_3d
        yield boxes_document(boxes.gravity_center.detach().cpu().numpy(),boxes.dims.detach().cpu().numpy(),boxes.yaw.detach().cpu().numpy(),
            pred.scores_3d.detach().cpu().numpy(),pred.labels_3d.detach().cpu().numpy(),profile['classes'],
            frame_id=frame['frame_id'],coordinate_frame=loader.get('coord_type','UNKNOWN'))


def run(root):
    root=Path(root).resolve();request=read(root/'request.json');profile=request['adapter']
    if not request.get('coordinates_confirmed'):raise ValueError('Chưa xác nhận trục/đơn vị input')
    if digest(profile['checkpoint'])!=profile['checkpoint_sha256'] or digest(profile['config'])!=profile['config_sha256']:
        raise ValueError('Adapter config/weights đã thay đổi')
    write(root/'state.json',dict(status='running',started_at=now(),optimizer_steps=0))
    try:
        handler={'openpcdet':openpcdet,'mmdetection3d':mmdetection3d}[profile['recipe']['backend']]
        rows=[]
        for i,row in enumerate(handler(profile,request)):
            name=f'frames/{i:06d}.json';write(root/name,row);rows.append(dict(frame_id=row['frame_id'],file=name))
            write(root/'state.json',dict(status='running',completed_frames=len(rows),optimizer_steps=0))
        write(root/'result.json',dict(model_id=profile['model_id'],frames=rows,optimizer_steps=0,
            target_annotations_read=False,adaptation_ready=False,checkpoint_sha256=profile['checkpoint_sha256']))
        write(root/'state.json',dict(status='completed',completed_frames=len(rows),optimizer_steps=0,finished_at=now()))
    except Exception as e:
        write(root/'state.json',dict(status='failed',error=str(e),optimizer_steps=0,finished_at=now()));raise


if __name__=='__main__':run(sys.argv[1])
