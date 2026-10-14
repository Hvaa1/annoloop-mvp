"""Native multi-class checkpoint inference; source GT only for validation metrics."""
import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np
from .io import read,write,digest,now,lock
from .inspection import matched_counts,load_xyz
from .runtime import configure_sparse_headers


def run(root):
    root=Path(root);spec=read(root/'spec.json')
    if digest(spec['path'])!=spec['checkpoint_sha256'] or digest(spec['config'])!=spec['config_sha256']:
        raise ValueError('Checkpoint/config đã thay đổi')
    configure_sparse_headers()
    runtime=Path(spec['runtime']);sys.path[:0]=[str(runtime),str(runtime/'tools')]
    import torch
    from pcdet.config import cfg,cfg_from_yaml_file
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network,load_data_to_gpu
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
    cfg_from_yaml_file(spec['config'],cfg)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH=spec['score_threshold']
    logger=logging.getLogger('inspection')
    classes=list(cfg.CLASS_NAMES)
    if classes!=spec['classes']:raise ValueError('Class order checkpoint/config không khớp')
    if spec['mode']=='validation':
        dataset=NuScenesDataset(cfg.DATA_CONFIG,classes,training=False,logger=logger)
    else:
        class TargetDataset(DatasetTemplate):
            def __init__(self):
                config=deepcopy(cfg.DATA_CONFIG)
                config.pop('USE_PSEUDO_LABEL',None);config.pop('TARGET',None)
                config.POINT_FEATURE_ENCODING.src_feature_list=['x','y','z']
                config.POINT_FEATURE_ENCODING.used_feature_list=['x','y','z']
                super().__init__(config,classes,False,Path(spec['frames'][0]['media']).parent,logger)
                self.frames=spec['frames']
            def __len__(self):return len(self.frames)
            def __getitem__(self,index):
                frame=self.frames[index]
                if digest(frame['media'])!=frame['sha256']:raise ValueError('Target media thay đổi')
                points=load_xyz(frame['media'],spec['bin_columns'])
                points=points+np.asarray(cfg.DATA_CONFIG.get('SHIFT_COOR',[0,0,0]),dtype=np.float32)
                return self.prepare_data(dict(points=points,frame_id=frame['frame_id']))
        dataset=TargetDataset()
    model=build_network(cfg.MODEL,len(classes),dataset)
    checkpoint=torch.load(spec['path'],map_location='cpu')
    if int(checkpoint.get('epoch',0))<=0 or int(checkpoint.get('it',0))<=0:raise ValueError('Checkpoint chưa có epoch/optimizer steps')
    weights=checkpoint['model_state']
    if any(not torch.isfinite(v).all() for v in weights.values() if torch.is_floating_point(v)):
        raise ValueError('Checkpoint có weights không hữu hạn')
    _,loaded=model._load_state_dict(weights,strict=False)
    missing=[name for name in model.state_dict() if name not in loaded and not name.endswith('num_batches_tracked')]
    if missing:raise ValueError('Checkpoint thiếu weights: '+str(missing[:10]))
    model.cuda().eval()
    totals={c:dict(tp=0,fp=0,fn=0) for c in classes}
    outputs=[];calibration_boxes=[];start=time.monotonic()
    state=read(root/'inspection.json');state.update(status='running',total_frames=len(dataset),completed_frames=0);write(root/'inspection.json',state)
    for index in range(min(len(dataset), spec.get('limit', len(dataset)))):
        if spec['mode']=='target_sample':
            from .inference_cache import cached_frame
            reused=cached_frame(root,spec,index)
            if reused is not None:
                fid=reused['frame_id'];name=f'{index:06d}.json'
                outputs.append(dict(frame_id=fid,file='frames/'+name))
                state=read(root/'inspection.json');state.update(completed_frames=index+1,total_frames=len(dataset))
                write(root/'inspection.json',state)
                print(f'{index+1}/{len(dataset)} {fid}: {len(reused["boxes"])} boxes (reused)',flush=True)
                continue
        batch=dataset.collate_batch([dataset[index]]);load_data_to_gpu(batch)
        with torch.inference_mode():predictions,_=model(batch)
        prediction=predictions[0]
        geometries=prediction['pred_boxes'][:,:7];scores=prediction['pred_scores'];labels=prediction['pred_labels']
        if not torch.isfinite(geometries).all() or not torch.isfinite(scores).all():raise ValueError('Prediction không hữu hạn')
        if len(labels) and (int(labels.min())<1 or int(labels.max())>len(classes)):raise ValueError('Class ID ngoài config')
        components = {}
        for key in ('pred_cls_scores', 'pred_iou_scores'):
            if key in prediction:
                values = prediction[key].reshape(-1)
                if len(values) != len(scores) or not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError('Prediction component không hợp lệ: ' + key)
                components[key] = values.cpu().tolist()
            elif spec.get('require_components'):
                raise ValueError('Runtime thiếu confidence/IoU riêng: ' + key)
        source_iou = np.zeros(len(scores), dtype=np.float32)
        if spec['mode']=='validation':
            gt=batch['gt_boxes'][0]
            for class_id,name in enumerate(classes,1):
                mask=labels==class_id;truth=gt[gt[:,-1]==class_id,:7]
                overlaps=boxes_iou3d_gpu(geometries[mask],truth).cpu().numpy() if mask.any() and len(truth) else np.zeros((int(mask.sum()),len(truth)))
                if len(truth) and int(mask.sum()):
                    source_iou[mask.cpu().numpy()] = overlaps.max(axis=1)
                counts=matched_counts(overlaps,scores[mask].cpu().numpy(),spec['iou_threshold'])
                for key,value in counts.items():totals[name][key]+=value
        shift=np.array(cfg.DATA_CONFIG.get('SHIFT_COOR',[0,0,0]))
        boxes=[]
        for number,(geometry,score,label) in enumerate(zip(geometries.cpu().numpy(),scores.cpu().tolist(),labels.cpu().tolist())):
            geometry=geometry.copy();geometry[:3]-=shift
            box = dict(id=f'det_{number:04d}',class_name=classes[label-1],box=geometry.tolist(),score=score,origin='checkpoint_inference')
            if components:
                box.update(cls_score=components['pred_cls_scores'][number], iou_score=components['pred_iou_scores'][number])
                if spec['mode']=='validation':
                    calibration_boxes.append(dict(box, source_gt_iou=float(source_iou[number])))
            boxes.append(box)
        fid=str(batch['frame_id'][0]);document=dict(frame_id=fid,boxes=boxes,target_gold_read=False)
        name=f'{index:06d}.json';write(root/'frames'/name,document)
        record=dict(frame_id=fid,file='frames/'+name)
        if spec['mode']=='target_sample' and spec.get('save_point_preview', True):
            points=load_xyz(spec['frames'][index]['media'],spec['bin_columns'])
            sampled=points[::max(1,len(points)//12000)]
            point_path=root/'points'/f'{index:06d}.npy';point_path.parent.mkdir(exist_ok=True)
            np.save(point_path,sampled,allow_pickle=False)
            record.update(media=spec['frames'][index]['media'],point_preview='points/'+point_path.name,point_count=len(points))
        outputs.append(record)
        state=read(root/'inspection.json');state.update(completed_frames=index+1,total_frames=len(dataset));write(root/'inspection.json',state)
        print(f'{index+1}/{len(dataset)} {fid}: {len(boxes)} boxes',flush=True)
    metrics=[]
    for name,counts in totals.items():
        tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
        metrics.append(dict(class_name=name,**counts,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None))
    write(root/'result.json',dict(mode=spec['mode'],domain='source_validation' if spec['mode']=='validation' else 'target_unlabelled',
        checkpoint=spec['path'],checkpoint_sha256=spec['checkpoint_sha256'],epoch=int(checkpoint['epoch']),optimizer_steps=int(checkpoint['it']),classes=classes,
        calibration_records=dict(domain='source_validation', boxes=calibration_boxes) if spec['mode']=='validation' else None,
        frames=outputs,metrics=metrics if spec['mode']=='validation' else [],iou_threshold=spec['iou_threshold'],score_threshold=spec['score_threshold'],
        metric_definition='Score-ordered one-to-one matching per frame/class at 3D IoU threshold; not official nuScenes mAP',
        target_gold_read=False,reused_frames=len(spec.get('reused_frames',{})),
        new_forward_frames=len(outputs)-len(spec.get('reused_frames',{})),
        elapsed_seconds=time.monotonic()-start,loaded_keys=len(loaded),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()))
    print('Inference/evaluation completed; no training performed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    root=Path(args.run)
    try:
        from .settings import Settings
        from .device import gpu_idle
        with lock(Settings.load().data/'.gpu.lock',blocking=False):
            gpu_idle();run(root)
        state=read(root/'inspection.json');state.update(status='completed',finished_at=now());write(root/'inspection.json',state)
    except BaseException as error:
        state=read(root/'inspection.json');state.update(status='failed',error=type(error).__name__+': '+str(error),finished_at=now());write(root/'inspection.json',state)
        raise
