"""Expose installed model-hub packages in adaptation without granting capability."""
from pathlib import Path
from .io import read
from .model_hub import load_catalog,model_root,canonical,installed


def candidates(data,classes):
    rows=[]
    for m in load_catalog()['models']:
        root=model_root(data,m['id']);receipt=root/'installed.json'
        if not receipt.is_file() or not (root/'weights.pth').is_file():continue
        config=root/'upstream'/m['config_path']
        if not config.is_file():continue
        try:
            meta=read(receipt)
            if meta.get('model_id')!=m['id'] or meta.get('weights_url')!=m['weights_url']:continue
        except (ValueError,OSError):continue
        supported={canonical(c) for c in m['classes']}
        connection=read(root/'connection.json') if (root/'connection.json').is_file() else {}
        recipe=connection.get('mmdet_recipe') if connection.get('checkpoint_sha256')==meta.get('sha256') else None
        rows.append(dict(path=str(root/'weights.pth'),config=str(config),runtime=m['framework'],
            classes=m['classes'],architecture=m['architecture'],intermediate=False,
            backend='registry_framework',model_id=m['id'],framework=m['framework'],dataset=m['dataset'],
            name=f"{m.get('family',m['architecture'])} · {m['dataset']} · {m['name']} · Kho model",
            adapter_status='prepared_unverified' if (root/'adapter.json').exists() else 'not_prepared',
            missing_classes=[c for c in classes if canonical(c) not in supported],adaptation_ready=False,
            **(dict(mmdet_recipe=recipe) if recipe else {})))
    return rows


def blockers(model,classes):
    if model.get('backend','native_st3d')=='native_st3d':return []
    if model.get('backend')!='registry_framework':return ['Backend model chưa được hỗ trợ']
    recipe=model.get('mmdet_recipe')
    if model.get('framework')=='mmdetection3d' and recipe and recipe.get('implementation_supported'):
        from .mmdet_recipe import current
        from .mmdet_recipe import legacy_blocker
        rows=list(dict.fromkeys(list(recipe['blockers'])+legacy_blocker(model)))
        if not current(recipe):rows.append('Config recipe đã thay đổi; chuẩn bị lại binding')
        missing=[c for c in classes if canonical(c) not in {canonical(n) for n in model['classes']}]
        if missing:rows.append('Checkpoint thiếu class adaptation: '+', '.join(missing))
        return rows
    identity=model.get('architecture','Model')+' / '+model.get('framework',model.get('runtime','framework'))
    rows=[identity+': chưa nối train target, signed-ignore và calibration vào vòng adaptation. '
          'Mới có handler raw predict, chưa chứng nhận chạy thật. Cần bổ sung recipe; bấm kiểm tra GPU lại chưa giải quyết được.']
    if model.get('adapter_status')=='not_prepared':rows.append('Chưa tạo adapter binding; mở Kho model → Chọn model & chuẩn bị adapter.')
    supported={canonical(c) for c in model['classes']}
    missing=[c for c in classes if canonical(c) not in supported]
    if missing:rows.append('Checkpoint thiếu class adaptation: '+', '.join(missing))
    return rows


def validate_cache(data,candidate):
    """Full integrity only on explicit save, not hashing every package on rerender."""
    model=next((m for m in load_catalog()['models'] if m['id']==candidate['model_id']),None)
    if model is None or not installed(data,model):raise ValueError('Gói model đã thiếu/thay đổi; kiểm tra lại cache trong Kho model')
    root=model_root(data,model['id'])
    if (root/'adapter.json').exists():
        from .adapter_factory import load_adapter
        load_adapter(data,model['id'],None)  # Binding validation only; no model import or GPU.
