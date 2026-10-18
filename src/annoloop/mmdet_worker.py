"""MMDetection3D 1.x leaf worker: explicit predict, verify or monitored training.

No target annotations; source-assisted geometry training with batch size one.
Velocity dimensions are storage placeholders with zero regression loss weight.
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
from .io import read,write,digest,now
from .model_hub import canonical
from .mmdet_recipe import current
from .mmdet_quality import ANGLE,rotate_points,consistency

VERIFY_TARGET_BOXES=32


def sources(spec,split):
    setup=spec['source_setup'];evidence=setup['source_evidence'][split]
    if digest(evidence['path'])!=evidence['sha256']:raise ValueError('Source infos đã đổi')
    with Path(evidence['path']).open('rb') as stream:infos=pickle.load(stream)
    if isinstance(infos,dict):infos=infos.get('infos')
    if not isinstance(infos,list):raise ValueError('Recipe source hiện nhận nuScenes OpenPCDet infos, không nhận data_list v2')
    roots=[Path(setup[k]) for k in ('loader_root','source_root')]
    result=[]
    for index,info in enumerate(infos):
        if 'token' not in info or not isinstance(info.get('sweeps'),list):
            raise ValueError('Source chưa phải infos nuScenes OpenPCDet được recipe hỗ trợ; không đoán layout BIN')
        path=Path(info['lidar_path']);options=[path] if path.is_absolute() else [r/path for r in roots]
        media=next((p.resolve() for p in options if p.is_file()),None)
        if media is None:raise ValueError('Thiếu source point cloud: '+str(path))
        names=[canonical(str(n)) for n in info['gt_names']];boxes=np.asarray(info['gt_boxes'],dtype=np.float32)
        if boxes.ndim!=2 or len(boxes)!=len(names) or boxes.shape[1]<7 or not np.isfinite(boxes[:,:7]).all() or (boxes[:,3:6]<=0).any():raise ValueError('Source geometry không hợp lệ')
        result.append(dict(frame_id=str(info.get('token',index)),media=str(media),sha256=digest(media),
            bin_columns=5,feature_names=['x','y','z','intensity','ring_index'],
            source_boxes=boxes[:,:7].tolist(),source_names=names))
    if not result:raise ValueError('Source split rỗng')
    return result


def prepare(spec,device='cuda:0'):
    if digest(spec['path'])!=spec['checkpoint_sha256'] or digest(spec['config'])!=spec['config_sha256']:
        raise ValueError('Checkpoint/config thay đổi')
    recipe=spec['mmdet_recipe']
    if not current(recipe) or not recipe['implementation_supported'] or not recipe['input_supported']:
        raise ValueError('Recipe/input chưa hỗ trợ: '+'; '.join(recipe['blockers']))
    import torch
    from mmengine.config import Config
    from .mmdet_model import build_model
    cfg=Config.fromfile(spec['config'])
    # Never fetch backbone initializer URLs: every parameter comes from the
    # selected full checkpoint and must subsequently pass strict coverage.
    def clear_init(value):
        if isinstance(value,dict):
            if 'init_cfg' in value:value['init_cfg']=None
            if 'pretrained' in value:value['pretrained']=None
            for item in value.values():clear_init(item)
        elif isinstance(value,(list,tuple)):
            for item in value:clear_init(item)
    clear_init(cfg.model)
    model=build_model(cfg,device=device)
    payload=torch.load(spec['path'],map_location='cpu')
    from .mmdet_weights import load_weights
    load_weights(model,payload['state_dict'])
    names=cfg.get('class_names',cfg.get('metainfo',{}).get('classes',[]))
    if list(names)!=spec['classes']:raise ValueError('Config class order khác checkpoint binding')
    saved_classes=payload.get('meta',{}).get('dataset_meta',{}).get('classes',payload.get('meta',{}).get('CLASSES'))
    if saved_classes is not None and list(saved_classes)!=list(names):raise ValueError('Checkpoint taxonomy khác config')
    head=getattr(model,'pts_bbox_head',getattr(model,'bbox_head',None))
    if head is None or type(head).__name__!=recipe['head']:raise ValueError('Head runtime không khớp recipe')
    from .mmdet_ignore import install
    masks=install(head,recipe['head'])
    model.dataset_meta=dict(classes=list(names))
    return torch,model,head,masks,payload


def points(spec,frame):
    from .framework_worker import points_for
    recipe=spec['mmdet_recipe'];fields=recipe['features'];columns=frame.get('bin_columns',spec['bin_columns'])
    if recipe.get('input_mode')=='current_scan_only' and 'time' in fields:
        from .model_filter import point_features
        available=point_features(frame)
        if 'time' in available:
            array=points_for(frame,fields,columns)
            if (array[:,fields.index('time')]!=0).any():raise ValueError('Current-scan mode không nhận điểm có elapsed time khác0')
        else:
            measured=[f for f in fields if f!='time'];raw=points_for(frame,measured,columns)
            array=np.zeros((len(raw),len(fields)),dtype=np.float32)
            for i,name in enumerate(measured):array[:,fields.index(name)]=raw[:,i]
    else:array=points_for(frame,fields,columns)
    bounds=spec['mmdet_recipe'].get('point_cloud_range')
    if bounds is not None:
        bounds=np.asarray(bounds,dtype=np.float32)
        if bounds.shape!=(6,) or not np.isfinite(bounds).all() or not (bounds[3:]>bounds[:3]).all():raise ValueError('Point cloud range config không hợp lệ')
        array=array[((array[:,:3]>=bounds[:3])&(array[:,:3]<bounds[3:])).all(axis=1)]
    if not len(array):raise ValueError('Không có điểm trong range của checkpoint; kiểm tra đơn vị/trục')
    return array


def batch(torch,model,head,array,truth=None):
    from mmengine.structures import InstanceData
    from mmdet3d.structures import Det3DDataSample,LiDARInstance3DBoxes,Box3DMode
    sample=Det3DDataSample();sample.set_metainfo(dict(box_type_3d=LiDARInstance3DBoxes,box_mode_3d=Box3DMode.LIDAR))
    if truth is not None:
        geometries,labels=truth
        dimension=9 if getattr(head,'box_code_size',7)>7 or hasattr(head,'task_heads') else 7
        rows=np.zeros((len(geometries),dimension),dtype=np.float32);rows[:,:7]=np.asarray(geometries,dtype=np.float32).reshape(-1,7)
        sample.gt_instances_3d=InstanceData(bboxes_3d=LiDARInstance3DBoxes(torch.as_tensor(rows),box_dim=dimension,origin=(.5,.5,.5)),
                                         labels_3d=torch.as_tensor(labels,dtype=torch.long))
    data=dict(inputs=dict(points=[torch.as_tensor(array)]),data_samples=[sample])
    return model.data_preprocessor(data,training=truth is not None)


def predict(torch,model,head,array,classes):
    from .framework_worker import boxes_document
    model.eval()
    with torch.no_grad():
        result=model(**batch(torch,model,head,array),mode='predict')[0].pred_instances_3d
    boxes=result.bboxes_3d
    doc=boxes_document(boxes.gravity_center.detach().cpu().numpy(),boxes.dims.detach().cpu().numpy(),
        boxes.yaw.detach().cpu().numpy(),result.scores_3d.detach().cpu().numpy(),result.labels_3d.detach().cpu().numpy(),
        classes,frame_id='prediction',coordinate_frame='LIDAR')
    output=[]
    for b in doc['boxes']:
        if b['score']>=.05:b['class_name']=canonical(b['class_name']);output.append(b)
    return output


def predictions(torch,model,head,array,classes):
    return consistency(predict(torch,model,head,array,classes),predict(torch,model,head,rotate_points(array,ANGLE),classes))


def truth_for(row,classes):
    mapped=[canonical(c) for c in classes];boxes=[];labels=[]
    for b,name in zip(row['source_boxes'],row['source_names']):
        if name in mapped:boxes.append(b);labels.append(mapped.index(name))
    return boxes,labels


def loss(torch,model,head,masks,array,truth,ignored):
    model.train()
    # Batch size1 is supported without mutating BN running statistics on the
    # small adaptation set. Affine parameters retain gradients.
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    masks['boxes']=torch.as_tensor(ignored,dtype=torch.float32,device='cuda').reshape(-1,7);masks['masked']=0
    values=model(**batch(torch,model,head,array,truth),mode='loss')
    terms=[]
    for key,value in values.items():
        if 'loss' in key:terms.extend(value if isinstance(value,list) else [value])
    if not terms:raise ValueError('Head không trả loss')
    total=sum(v.mean() for v in terms)
    if not torch.isfinite(total):raise ValueError('Loss NaN/Inf')
    return total


def infer(root,spec):
    from .reused.geometry import cuboid_iou
    torch,model,head,masks,_=prepare(spec)
    rows=sources(spec,'validation') if spec['mode']=='validation' else spec['frames']
    if spec.get('limit') is not None:rows=rows[:spec['limit']]
    frames=[];records=[]
    for i,row in enumerate(rows):
        boxes=predictions(torch,model,head,points(spec,row),spec['classes'])
        if spec['mode']=='validation':
            for b in boxes:
                overlaps=[cuboid_iou(b['box'],gt) for gt,n in zip(row['source_boxes'],row['source_names']) if n==b['class_name']]
                records.append(dict(b,source_gt_iou=max(overlaps,default=0.)))
        name=f'frames/{i:06d}.json';write(root/name,dict(frame_id=row['frame_id'],boxes=boxes,target_gold_read=False))
        frames.append(dict(frame_id=row['frame_id'],file=name))
        write(root/'inspection.json',dict(status='running',total_frames=len(rows),completed_frames=i+1))
        print(f'{i+1}/{len(rows)} {row["frame_id"]}: {len(boxes)} boxes',flush=True)
    write(root/'result.json',dict(frames=frames,domain='source_validation' if spec['mode']=='validation' else 'target_unlabelled',
        checkpoint_sha256=spec['checkpoint_sha256'],classes=spec['classes'],target_gold_read=False,
        quality_semantics='rotation_consistency_iou',input_mode=spec['mmdet_recipe'].get('input_mode','strict'),
        calibration_records=dict(domain='source_validation',boxes=records) if spec['mode']=='validation' else None))
    write(root/'inspection.json',dict(status='completed',total_frames=len(rows),completed_frames=len(rows)))


def verify(root,spec):
    from .native_training import checked_gradients
    torch,model,head,masks,_=prepare(spec)
    source=next((r for r in sources(spec,'train') if truth_for(r,spec['classes'])[0]),None)
    if source is None:raise ValueError('Source không có positive phù hợp')
    array=points(spec,source);truth=truth_for(source,spec['classes'])
    from .reused.core import points_in_box
    if not any(points_in_box(array[:,:3],b).sum()>=5 for b in truth[0]):
        raise ValueError('GT source không phủ đủ điểm trong hệ LiDAR; kiểm tra convention box/đơn vị/trục')
    # Geometry of an actual source cloud point, far from labelled positives.
    box=list(truth[0][0]);box[:2]=[0.,0.];box[3:6]=[8.,8.,3.]
    value=loss(torch,model,head,masks,array,truth,[box]);value.backward()
    source_grad=checked_gradients(torch,model);count=masks['masked'];source_loss=float(value.detach().cpu())
    if not count:raise ValueError('Ignore diagnostic không mask được negative loss; chưa cấp chứng nhận')
    del value;model.zero_grad(set_to_none=True)
    target=spec['frames'][0];array=points(spec,target)
    boxes=predictions(torch,model,head,array,spec['classes'])
    if not boxes:raise ValueError('Target sample không có positive để kiểm tra backward')
    # A backward check needs a few positives, not every raw box: hundreds of
    # boxes x ~400k anchors exhaust a 4GB GPU in the IoU assigner.
    boxes=sorted(boxes,key=lambda b:-b['score'])[:VERIFY_TARGET_BOXES]
    names=[canonical(c) for c in spec['classes']]
    value=loss(torch,model,head,masks,array,([b['box'] for b in boxes],[names.index(b['class_name']) for b in boxes]),[])
    value.backward();target_grad=checked_gradients(torch,model)
    write(root/'verified.json',dict(checkpoint_sha256=spec['checkpoint_sha256'],config_sha256=spec['config_sha256'],
        optimizer_steps=0,target_gold_read=False,separate_scores_checked=True,forward_backward_checked=True,
        target_adapter_backward_checked=True,signed_ignore_effect_checked=True,ignore_counts=dict(dense=count),
        source_gradient_tensors=source_grad,target_gradient_tensors=target_grad,source_loss=source_loss,
        target_loss=float(value.detach().cpu()),target_verify_boxes=len(boxes),quality_semantics='rotation_consistency_iou',recipe_version=1,
        input_mode=spec['mmdet_recipe'].get('input_mode','strict')))


def train(root,spec):
    from .predictions import assert_no_gold
    from .native_data import signed_labels
    from .native_training import checked_gradients
    from .reused.core import points_in_box
    policy=spec['policy'];epochs=policy['epochs']
    if policy['ckpt_interval']!=10 or policy['save_final'] is not True:raise ValueError('Cần checkpoint10 epochs + final')
    labels=read(spec['labels_path']);assert_no_gold(labels)
    if digest(spec['labels_path'])!=spec['shared_labels_sha256']:raise ValueError('Pseudo-label hash đã đổi')
    torch,model,head,masks,payload=prepare(spec);names=[canonical(n) for n in spec['classes']]
    labelled={r['frame_id']:r for r in labels['frames']}
    if set(labelled)!={r['frame_id'] for r in spec['frames']}:raise ValueError('Pseudo-label frame coverage không khớp')
    usable=[]
    for frame in spec['frames']:
        row=labelled[frame['frame_id']];signed,_,_=signed_labels(row,names);array=points(spec,frame)
        if any(b[-1]>0 and points_in_box(array[:,:3],b[:7]).sum()>=5 for b in signed):usable.append(frame)
    if not usable:raise ValueError('Không có positive đủ điểm để train')
    source=[r for r in sources(spec,'train') if truth_for(r,spec['classes'])[0]]
    if not source:raise ValueError('Source train không có positive')
    optimizer=torch.optim.AdamW(model.parameters(),lr=policy.get('learning_rate',.0001))
    steps=0;start=0
    if policy.get('resume_checkpoint'):
        if payload.get('annoloop_labels_sha256')!=spec['shared_labels_sha256'] or payload.get('annoloop_config_sha256')!=spec['config_sha256']:
            raise ValueError('Resume checkpoint khác nhãn/config')
        if payload.get('annoloop_input_mode','strict')!=spec.get('mmdet_recipe',{}).get('input_mode','strict'):raise ValueError('Resume checkpoint khác input mode')
        optimizer.load_state_dict(payload['optimizer']);start=payload['epoch'];steps=payload['it']
        if not 0<=start<=epochs:raise ValueError('Resume epoch ngoài phạm vi')
    ckpts=root/'checkpoints';ckpts.mkdir(parents=True,exist_ok=True)
    def save(epoch):
        path=ckpts/f'checkpoint_epoch_{epoch}.pth';temporary=path.with_suffix('.tmp')
        torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,it=steps,
            annoloop_labels_sha256=spec['shared_labels_sha256'],annoloop_config_sha256=spec['config_sha256'],
            annoloop_input_mode=spec.get('mmdet_recipe',{}).get('input_mode','strict'),
            meta=dict(dataset_meta=dict(classes=spec['classes']))),temporary);temporary.replace(path);return path
    for epoch in range(start+1,epochs+1):
        for index in np.random.permutation(len(usable)):
            optimizer.zero_grad(set_to_none=True)
            row=source[steps%len(source)];value=loss(torch,model,head,masks,points(spec,row),truth_for(row,spec['classes']),[])
            source_value=float(value.detach().cpu());(policy.get('source_weight',.5)*value).backward();del value
            frame=usable[int(index)];row=labelled[frame['frame_id']]
            signed,_,_=signed_labels(row,names);positive=signed[signed[:,-1]>0];ignored=signed[signed[:,-1]<0,:7]
            truth=(positive[:,:7],(positive[:,-1]-1).astype(int))
            array=points(spec,frame)
            active=set(labels.get('adaptation_classes',names))
            if active!=set(names):
                other=[b['box'] for b in predict(torch,model,head,array,spec['classes']) if b['class_name'] not in active]
                if other:ignored=np.concatenate((ignored,np.asarray(other,dtype=np.float32)),axis=0)
            value=loss(torch,model,head,masks,array,truth,ignored);value.backward()
            checked_gradients(torch,model);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            if not torch.isfinite(norm):raise ValueError('Gradient norm NaN/Inf')
            optimizer.step();steps+=1
            target_value=float(value.detach().cpu());del value
            torch.cuda.empty_cache()  # release fragments before the next, differently sized frame
            write(root/'progress.json',dict(status='running',epoch=epoch,epoch_completed=epoch-1,epochs=epochs,optimizer_steps=steps,
                loss=source_value*policy.get('source_weight',.5)+target_value,source_loss=source_value,target_loss=target_value))
            print(f'Epoch [{epoch}/{epochs}] total_it={steps} source={source_value:.5f} target={target_value:.5f}',flush=True)
        if epoch%10==0 or epoch==epochs:save(epoch)
    final=ckpts/f'checkpoint_epoch_{epochs}.pth'
    if not final.exists() and start==epochs:save(epochs)
    if digest(spec['labels_path'])!=spec['shared_labels_sha256']:raise ValueError('Nhãn đã đổi trong khi train')
    write(root/'trained.json',dict(checkpoint=str(final),checkpoint_sha256=digest(final),shared_labels_sha256=spec['shared_labels_sha256'],
        optimizer_steps=steps,epochs=epochs,target_gold_used=False,source_assisted=True,classes=spec['classes'],
        architecture=spec['architecture'],quality_semantics='rotation_consistency_iou',
        input_mode=spec.get('mmdet_recipe',{}).get('input_mode','strict'),finished_at=now()))
    write(root/'progress.json',dict(status='completed',epoch_completed=epochs,optimizer_steps=steps))


def run(root,mode):
    root=Path(root);spec=read(root/'spec.json')
    import os
    # Mixed source/target frame sizes fragment the 4GB GPU cache; must be set before CUDA init.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
    try:
        import importlib.util
        if importlib.util.find_spec('cumm'):
            # spconv2 compiles training kernels with NVRTC; same toolkit fix as native workers.
            from .runtime import configure_sparse_headers
            configure_sparse_headers()
        {'infer':infer,'verify':verify,'train':train}[mode](root,spec)
    except BaseException as error:
        name='progress.json' if mode=='train' else 'inspection.json'
        path=root/name;state=read(path) if path.exists() else {}
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=type(error).__name__+': '+str(error),finished_at=now())
        write(path,state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--verify',action='store_true');parser.add_argument('--infer',action='store_true')
    args=parser.parse_args();run(args.run,'verify' if args.verify else 'infer' if args.infer else 'train')
