"""Read pinned Python configs without executing them; narrow MMDet3D recipes."""
import ast
from copy import deepcopy
from pathlib import Path
from .io import digest

SUPPORTED_HEADS={'CenterHead','Anchor3DHead'}
SUPPORTED_MODELS={'CenterPoint','MVXFasterRCNN','DynamicMVXFasterRCNN','VoxelNet','DynamicVoxelNet'}


def merge(base,override):
    if not isinstance(base,dict) or not isinstance(override,dict):return deepcopy(override)
    if override.get('_delete_'):return {k:deepcopy(v) for k,v in override.items() if k!='_delete_'}
    result=deepcopy(base)
    for k,v in override.items():result[k]=merge(result.get(k),v)
    return result


def config_tree(path,root=None,stack=()):
    path=Path(path).resolve();root=Path(root or path.parent).resolve()
    if not path.is_relative_to(root) or path in stack:raise ValueError('Config base ngoài cache hoặc có chu kỳ')
    tree=ast.parse(path.read_text());base={};files={str(path):digest(path)}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_base_' for t in node.targets):
            names=ast.literal_eval(node.value)
            for name in ([names] if isinstance(names,str) else names):
                item,evidence=config_tree(path.parent/name,root,(*stack,path));base=merge(base,item);files.update(evidence)
    env=dict(base)
    def value(n):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return deepcopy(env[n.id])
        if isinstance(n,ast.Subscript):
            sub=n.slice
            key=slice(value(sub.lower) if sub.lower else None,value(sub.upper) if sub.upper else None,value(sub.step) if sub.step else None) if isinstance(sub,ast.Slice) else value(sub)
            return value(n.value)[key]
        if isinstance(n,(ast.List,ast.Tuple)):return [value(v) for v in n.elts]
        if isinstance(n,ast.Dict):return {value(k):value(v) for k,v in zip(n.keys,n.values)}
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and not n.args:
            if any(k.arg is None for k in n.keywords):raise ValueError('Config dict unpack chưa hỗ trợ')
            return {k.arg:value(k.value) for k in n.keywords}
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('sum','len') and len(n.args)==1 and not n.keywords:
            item=value(n.args[0]);return sum(item) if n.func.id=='sum' else len(item)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            return -value(n.operand) if isinstance(n.op,ast.USub) else value(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=value(n.left),value(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
            if isinstance(n.op,ast.Sub):return a-b
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='_base_':return deepcopy(base[n.attr])
        raise ValueError('Biểu thức config chưa hỗ trợ; không thực thi Python để suy đoán')
    unresolved=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name=='_base_':continue
            try:env[name]=value(node.value)
            except (KeyError,ValueError,TypeError,ZeroDivisionError):
                unresolved.append(name);env.pop(name,None)
    if any(n in unresolved for n in ('model','test_pipeline','test_dataloader','class_names')):
        raise ValueError('Không đọc tĩnh được config quan trọng: '+', '.join(unresolved))
    return merge(base,env),files


def describe(config,root):
    cfg,files=config_tree(config,root)
    model=cfg.get('model',{})
    if not isinstance(model,dict):raise ValueError('Model config chưa có recipe dictionary')
    head=model.get('pts_bbox_head',model.get('bbox_head',{}))
    if not isinstance(head,dict):raise ValueError('Head config chưa có recipe dictionary')
    kind=head.get('type','').split('.')[-1]
    pipeline=cfg.get('test_pipeline')
    if pipeline is None:
        data=cfg.get('test_dataloader',{}).get('dataset',{})
        while isinstance(data,dict) and 'dataset' in data:data=data['dataset']
        pipeline=data.get('pipeline',[])
    def walk(rows):
        for row in rows:
            if isinstance(row,list):yield from walk(row)
            elif isinstance(row,dict):
                yield row
                yield from walk(row.get('transforms',[]))
    stages=list(walk(pipeline or []));loaders=[s for s in stages if s.get('type','').split('.')[-1]=='LoadPointsFromFile']
    if len(loaders)!=1:raise ValueError('Cần đúng một LoadPointsFromFile trong test pipeline')
    loader=loaders[0];dims=loader.get('use_dim',loader.get('load_dim',6));dims=list(range(dims)) if isinstance(dims,int) else dims
    if loader.get('coord_type','LIDAR')!='LIDAR':raise ValueError('Recipe chỉ hỗ trợ hệ LiDAR')
    if dims not in ([0,1,2],[0,1,2,3],[0,1,2,3,4]):raise ValueError('Thứ tự feature chưa có recipe')
    multi=[s for s in stages if s.get('type','').split('.')[-1]=='LoadPointsFromMultiSweeps']
    sweeps=max([s.get('sweeps_num',10)+1 for s in multi] or [1])
    if multi:
        # MMDet3D's default sweep use_dim is XYZ + elapsed time, not XYZI.
        dims=multi[-1].get('use_dim',[0,1,2,4]);dims=list(range(dims)) if isinstance(dims,int) else dims
    if not isinstance(dims,list) or not all(type(i) is int and 0<=i<5 for i in dims):raise ValueError('Sweep feature indices chưa có recipe')
    features=[['x','y','z','intensity','time'][i] for i in dims]
    blockers=[]
    if kind not in SUPPORTED_HEADS:blockers.append('Chưa có loss/ignore recipe cho head '+kind)
    model_kind=model.get('type','').split('.')[-1]
    implemented=kind in SUPPORTED_HEADS and model_kind in SUPPORTED_MODELS and not model.get('roi_head') and not model.get('img_backbone')
    if not implemented and kind in SUPPORTED_HEADS:blockers.append('Chưa có recipe cho cấu trúc model '+model_kind+' hoặc nhánh ROI/camera')
    encoder=model.get('pts_voxel_encoder',model.get('voxel_encoder',{}))
    channels=encoder.get('in_channels',encoder.get('num_features'))
    if isinstance(channels,int) and channels!=len(features):blockers.append('Feature pipeline không khớp số kênh voxel encoder; không tự đổi weights/kênh')
    if any(loader.get(k) for k in ('shift_height','use_color','norm_intensity','norm_elongation')):
        blockers.append('Point loader có biến đổi feature chưa hỗ trợ trong recipe adaptation')
    if sweeps>1:blockers.append(f'Config yêu cầu {sweeps} sweep; chưa hỗ trợ manifest sweep/ego pose trong recipe adaptation')
    if 'time' in features:blockers.append('Model cần time thật; không tự thêm cột time cho KITTI XYZI')
    bounds=model.get('data_preprocessor',{}).get('voxel_layer',{}).get('point_cloud_range')
    return dict(version=1,head=kind,model_type=model_kind,features=features,sweeps=sweeps,point_cloud_range=bounds,config_files=files,
                implementation_supported=implemented,input_supported=not blockers,
                blockers=blockers,quality_semantics='rotation_consistency_iou',
                classes=cfg.get('class_names',cfg.get('metainfo',{}).get('classes',[])))


def current(recipe):
    try:return all(digest(p)==h for p,h in recipe['config_files'].items())
    except (KeyError,OSError):return False


def input_blockers(recipe,profile):
    reasons=list(recipe['blockers'])
    if not current(recipe):reasons.append('Config recipe đã thay đổi; chuẩn bị lại binding')
    if not profile['complete']:reasons.append('Chưa xác minh feature target')
    else:
        needed=set(recipe['features'])-({'time'} if recipe.get('input_mode')=='current_scan_only' else set())
        if not needed<=set(profile['features']):reasons.append('Target thiếu feature: '+', '.join(sorted(needed-set(profile['features']))))
    return reasons


def effective(recipe,mode='strict'):
    if mode not in ('strict','current_scan_only'):raise ValueError('MMDet input mode không hợp lệ')
    result=deepcopy(recipe);result['input_mode']=mode
    if mode=='current_scan_only':
        # No synthetic history or encoder surgery. Official elapsed-time
        # semantics: every point in the current reference scan has delta t=0.
        result['declared_sweeps']=recipe.get('declared_sweeps',recipe['sweeps']);result['sweeps']=1
        result['blockers']=[b for b in recipe['blockers'] if not b.startswith(('Config yêu cầu','Model cần time thật'))]
        result['input_supported']=not result['blockers']
    return result


LEGACY_CONVENTION=('Checkpoint MMDetection3D v0.x dùng quy ước box LiDAR cũ (yaw/l-w); '
    'đo trên source: yaw lệch ~90°, kích thước méo. Chọn checkpoint v1.0.0_models.')


_URLS = {}


def _catalog_urls():
    from .model_hub import load_catalog, CATALOG
    from .io import file_signature
    key = file_signature(CATALOG)
    if key not in _URLS:
        _URLS.clear(); _URLS[key] = {m['id']: m['weights_url'] for m in load_catalog()['models']}
    return _URLS[key]


def legacy_blocker(model):
    """v0.x weights predate the 1.x LiDAR box refactor; their boxes decode wrongly in 1.x."""
    import re
    url=model.get('weights_url')
    if url is None and model.get('model_id'):
        url=_catalog_urls().get(model['model_id'])
    return [LEGACY_CONVENTION] if url and re.search(r'/v0\.[0-9.]+_models/',url) else []


def with_legacy(model,recipe):
    rows=legacy_blocker(model)
    if not rows or rows[0] in recipe['blockers']:return recipe
    recipe=dict(recipe,blockers=list(recipe['blockers'])+rows,input_supported=False)
    return recipe


def cached_model(data,model):
    """Read downloaded input evidence for UI filtering, without model imports."""
    if model.get('framework')!='mmdetection3d':return model
    from .model_hub import model_root
    from .io import read
    root=model_root(data,model['id'])
    try:
        connection=read(root/'connection.json');installed=read(root/'installed.json')
        recipe=connection.get('mmdet_recipe')
        if (connection.get('model_id')==model['id'] and connection.get('checkpoint_sha256')==installed.get('sha256')
            and recipe and current(recipe)):return dict(model,mmdet_recipe=with_legacy(model,recipe))
    except (OSError,ValueError,KeyError):pass
    return static_model(root,model)


def static_model(root,model):
    """Downloaded but not yet connected: read the pinned config statically for filtering.

    Grants no capability; preparation still connects, strict-loads and verifies.
    """
    from .io import read,display_digest
    try:
        receipt=read(root/'installed.json')
        if receipt.get('model_id')!=model['id'] or receipt.get('weights_url')!=model['weights_url']:return model
        upstream=root/'upstream'
        for relative,spec in model['config_files'].items():
            if display_digest(upstream/relative)!=spec['sha256']:return model
        recipe=describe(upstream/model['config_path'],upstream)
    except (OSError,ValueError,KeyError,SyntaxError,TypeError):return model
    return dict(model,mmdet_recipe=with_legacy(model,recipe),mmdet_recipe_source='static_config_unconnected')
