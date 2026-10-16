"""Conservative, CPU-only model preselection. No inference, imports or downloads."""
from pathlib import Path
import numpy as np
import yaml
from .model_hub import canonical


def point_features(frame):
    """Read media headers only. RGB is never substituted for intensity."""
    path=Path(frame['media']);declared=frame.get('feature_names')
    if declared is not None and (not isinstance(declared,list) or not all(isinstance(f,str) and f for f in declared)
                                 or len(declared)!=len(set(declared))):
        raise ValueError('Khai báo feature_names không hợp lệ')
    if path.suffix.lower()=='.pcd':
        fields=None
        with path.open('rb') as stream:
            for _ in range(64):
                line=stream.readline(4096).decode('ascii').strip().split()
                if not line:continue
                if line[0].upper()=='FIELDS':fields=line[1:]
                if line[0].upper()=='DATA':break
            else:raise ValueError('PCD header không hợp lệ')
        if not fields or len(fields)!=len(set(fields)):raise ValueError('PCD thiếu FIELDS hoặc trùng field')
        return fields
    if path.suffix.lower()=='.npy':
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if not isinstance(array,np.ndarray) or array.ndim!=2:raise ValueError('NPY cần ma trận point cloud')
        if declared and len(declared)==array.shape[1]:return list(declared)
        if not declared and array.shape[1]==3:return ['x','y','z']
        raise ValueError('NPY nhiều cột cần khai báo ý nghĩa từng cột')
    if path.suffix.lower()=='.bin' and declared:
        if len(declared)<3 or path.stat().st_size % (4*len(declared)):
            raise ValueError('BIN không khớp số cột đã khai báo')
        return list(declared)
    raise ValueError('Chưa xác định đặc trưng từng cột của '+path.suffix)


def dataset_profile(frames,limit=16):
    """A declared sample, not a claim that every frame has been validated."""
    if type(limit) is not int or limit<1:raise ValueError('Số frame mẫu phải > 0')
    rows=list(frames)
    sample=[rows[int(i)] for i in np.linspace(0,len(rows)-1,min(limit,len(rows)),dtype=int)] if rows else []
    common=None;errors=[];variants=set()
    for frame in sample:
        try:
            fields=tuple(point_features(frame));variants.add(fields)
            common=set(fields) if common is None else common & set(fields)
        except (OSError,ValueError,UnicodeError) as error:
            errors.append(dict(frame_id=frame['frame_id'],reason=str(error)))
    # Sequence IDs alone do not prove registered, timed multi-sweep input.
    return dict(features=sorted(common or []),checked=len(sample),total=len(rows),
                complete=bool(sample) and not errors,mixed=len(variants)>1,errors=errors,
                sweeps_available=False)


def requirements(model):
    if model.get('backend','registry_framework')=='native_st3d':
        try:
            cfg=yaml.safe_load(Path(model['config']).read_text())
            fields=cfg['DATA_CONFIG']['POINT_FEATURE_ENCODING']['used_feature_list']
            sweeps=cfg['DATA_CONFIG'].get('MAX_SWEEPS',1)
            if not isinstance(fields,list) or not fields or not all(isinstance(f,str) for f in fields):return None
            if type(sweeps) is not int or sweeps<1:return None
            return dict(features=fields,sweeps=sweeps)
        except (OSError,ValueError,KeyError,TypeError,yaml.YAMLError):return None
    # Catalog contracts must cite the pinned config evidence, not the model name.
    contract=model.get('input_contract')
    if not isinstance(contract,dict) or not isinstance(contract.get('features'),list) or not contract['features']:return None
    evidence=contract.get('evidence',[])
    if not evidence or any(p not in model.get('config_files',{}) for p in evidence):return None
    if not all(isinstance(f,str) and f for f in contract['features']):return None
    if type(contract.get('sweeps',1)) is not int or contract.get('sweeps',1)<1:return None
    return contract


def assess(model,classes,profile,source_counts=None):
    supported={canonical(c) for c in model.get('classes',[])}
    missing_classes=[c for c in classes if canonical(c) not in supported]
    reasons=[]
    if missing_classes:reasons.append('Thiếu class: '+', '.join(missing_classes))
    if source_counts is not None:
        for split in ('train','validation'):
            counts={canonical(c):v for c,v in source_counts.get(split,{}).items()}
            absent=[c for c in classes if counts.get(canonical(c),0)<=0]
            if absent:reasons.append('Source '+split+' thiếu nhãn: '+', '.join(absent))
    if model.get('modality','point_cloud')!='point_cloud':reasons.append('Cần dữ liệu camera hoặc modality khác')
    native=model.get('backend','registry_framework')=='native_st3d'
    if not native:reasons.append('Chưa có adapter adaptation train / calibration / signed-ignore')
    contract=requirements(model)
    missing=[]
    if contract is None:reasons.append('Chưa xác minh yêu cầu đặc trưng đầu vào của checkpoint')
    else:
        if not profile['complete']:reasons.append('Chưa đọc đủ đặc trưng của các frame mẫu')
        missing=[f for f in contract['features'] if f not in profile['features']]
        if missing:reasons.append('Target thiếu feature: '+', '.join(missing))
        if contract.get('sweeps',1)>1 and not profile['sweeps_available']:
            reasons.append(f"Cần {contract['sweeps']} sweep có timestamp / ego pose; chưa có manifest được hỗ trợ")
    return dict(eligible=not reasons,reasons=reasons,missing_classes=missing_classes,
                missing_features=missing,contract=contract,
                # Candidate eligibility never replaces per-project GPU verification.
                gpu_verified=False)
