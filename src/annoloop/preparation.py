"""Project-bound post-download checks. Viewing never dispatches model work."""
from pathlib import Path
import math
import uuid
from .io import read,write,digest,display_digest,now


def certificate_valid(root,certificate,config,hasher=digest):
    """Only native worker evidence can open the maintained native training route."""
    try:
        if certificate.get('preparation_protocol')!=1:return False
        classes=config.get('adaptation_classes',[])
        if not classes or certificate.get('adaptation_classes')!=classes:return False
        for role,model in config['models'].items():
            receipt=certificate['preparation_checks'][role]
            documents={}
            for key in ('verification','calibration'):
                path=Path(receipt[key]).resolve()
                if not path.is_relative_to(Path(root).resolve()) or hasher(path)!=receipt[key+'_sha256']:return False
                documents[key]=read(path)
            verified=documents['verification'];calibration=documents['calibration']
            if model.get('mmdet_recipe'):
                if verified.get('recipe_version')!=1 or verified.get('quality_semantics')!='rotation_consistency_iou':return False
                if verified.get('input_mode','strict')!=model['mmdet_recipe'].get('input_mode','strict'):return False
                from .mmdet_recipe import current
                if not current(model['mmdet_recipe']):return False
            if (verified['checkpoint_sha256']!=model['checkpoint_sha256']
                or verified['config_sha256']!=model['config_sha256']
                or verified.get('optimizer_steps')!=0 or verified.get('target_gold_read') is not False):return False
            if not all(verified.get(k) is True for k in ('separate_scores_checked','forward_backward_checked',
                'target_adapter_backward_checked','signed_ignore_effect_checked')):return False
            if not all(isinstance(verified.get(k),int) and verified[k]>0
                       for k in ('source_gradient_tensors','target_gradient_tensors')):return False
            if not all(math.isfinite(verified[k]) for k in ('source_loss','target_loss')):return False
            if (calibration.get('domain')!='source_validation'
                or calibration.get('checkpoint_sha256')!=model['checkpoint_sha256']
                or set(calibration.get('curves',{}))!=set(classes)):return False
            if any(set(calibration['curves'][name])!={'confidence','quality'} for name in classes):return False
        return set(config['models'])=={'A','B'}
    except (ValueError,TypeError,KeyError,OSError):return False


def overview(store,pid,fast=False):
    from . import adaptation
    from .model_hub import load_catalog,model_root
    from .model_filter import dataset_profile,assess
    root=store.path(pid);project=store.get(pid)
    config=read(root/'adaptation.json') if (root/'adaptation.json').exists() else {}
    inventory=read(root/'dataset.json');setup=read(root/'setup.json')
    profile=dataset_profile([f for f in inventory['frames'] if f['split']=='adaptation'])
    classes=config.get('adaptation_classes',project['schema']['classes'])
    registry={m['id']:m for m in load_catalog()['models']}
    check=adaptation.readiness(store,pid,fast=fast)
    certificate=read(root/'native_verified.json') if (root/'native_verified.json').exists() else {}
    complete=check['gpu_verified'] and certificate_valid(root,certificate,config,display_digest if fast else digest)
    stages=[]
    for role in ('A','B'):
        intent=config.get('model_selection',{}).get(role,{})
        bound=config.get('models',{}).get(role)
        model=registry.get(intent.get('id')) if intent.get('kind')=='hub' else bound
        title=(model or {}).get('architecture',adaptation.ARCHITECTURES[role])
        def add(stage,status,detail):stages.append(dict(role=role,model=title,stage=stage,status=status,detail=detail))
        if model is None:
            add('Checkpoint','pending','Chọn/tải model hoặc hoàn tất học nhãn mới từ source.');continue
        if intent.get('kind')!='hub':model=dict(model,backend=model.get('backend','native_st3d'))
        native=model.get('backend','registry_framework')=='native_st3d'
        if intent.get('kind')=='hub':
            cache=model_root(store.root.parent,intent['id'])
            downloaded=(cache/'installed.json').exists() and (cache/'weights.pth').exists()
            connection=read(cache/'connection.json') if (cache/'connection.json').exists() else {}
            bound_adapter=(cache/'adapter.json').exists()
        else:
            downloaded=bool(model.get('path') and Path(model['path']).is_file())
            connection={};bound_adapter=native
        recipe=(bound or {}).get('mmdet_recipe') or connection.get('mmdet_recipe')
        assessment=assess(model,classes,profile,setup['source_counts'])
        if recipe:
            from .mmdet_recipe import input_blockers
            reasons=input_blockers(recipe,profile)
            assessment.update(input_status='incompatible' if reasons else 'compatible',selectable=not reasons,selection_blockers=reasons)
        add('Class & đầu vào','passed' if assessment['input_status']=='compatible' else
            'failed' if not assessment['selectable'] else 'pending',
            ('Đầu vào theo chế độ '+recipe.get('input_mode','strict')+'.' if recipe else 'Đúng hợp đồng đầu vào.') if assessment['input_status']=='compatible' else
            ' · '.join(assessment['selection_blockers']) or 'Cần xác minh input contract sau tải; chưa được coi là tương thích.')
        add('Tải checkpoint','passed' if downloaded else 'pending','Có file local; hash được kiểm lại khi bấm kiểm tra.' if downloaded else 'Chưa tải hoặc học source chưa hoàn tất.')
        add('Chuẩn bị adapter','passed' if bound_adapter else 'pending','Recipe đã nối; chưa phải chứng nhận chạy.' if bound_adapter else 'Tải & chuẩn bị adapter cho model đã chọn.')
        capabilities=connection.get('capabilities',{})
        for name,capability in (('Predict source/target','raw_prediction_implemented'),
                                ('Forward/backward','target_training_implemented'),
                                ('Signed-ignore','signed_ignore_implemented'),('Calibration source','calibration_implemented')):
            report=check.get('calibration_reports',{}).get(role) if native or recipe else None
            if name=='Calibration source' and report and not report['ready']:
                missing=[r['class_name']+': '+str(r['samples'])+'/'+str(report['minimum_samples']) for r in report['classes'] if not r['enough_samples']]
                add(name,'failed','Thiếu dự đoán source validation: '+', '.join(missing))
            elif complete:
                add(name,'passed','Có bằng chứng kiểm chứng khớp checkpoint, source và dataset.')
            elif native or recipe and recipe.get('implementation_supported'):
                add(name,'pending','Bấm kiểm tra; model chạy tuần tự, không optimizer step.')
            elif capabilities.get(capability):
                add(name,'pending','Có code handler; chưa có chứng nhận chạy trên dự án này.')
            else:add(name,'missing','Recipe '+str(model.get('framework'))+' chưa triển khai '+name+'.')
    return dict(project_id=pid,stages=stages,ready=bool(check['ready'] and complete),
                blockers=check['blockers'],can_check=check['can_check'],checked_at=now(),
                dataset_sha256=project['dataset_sha256'],target_gold_read=False,busy=adaptation.execution_busy(store,pid))


def save_input_setup(store,pid,settings,coordinates_confirmed,bin_columns,mmdet_input_mode):
    """Save an explicit input choice without downloading, importing or dispatching."""
    from . import adaptation
    config=read(store.path(pid)/'adaptation.json')
    return adaptation.save_setup(store,pid,settings,{k:m['path'] for k,m in config.get('models',{}).items()},
        coordinates_confirmed,int(bin_columns),config.get('epochs',1),config.get('learning_rate',.0001),
        config.get('source_weight',.5),config.get('adaptation_classes'),config.get('memory_policy'),
        preparation_protocol=1,mmdet_input_mode=mmdet_input_mode)


def check_prepared(store,pid,settings,coordinates_confirmed,bin_columns,mmdet_input_mode=None):
    """Explicit action: bind/validate downloaded files, then dispatch implemented checks."""
    from . import adaptation
    from .model_hub import load_catalog
    from .connect_model import connect
    root=store.path(pid);config=read(root/'adaptation.json')
    selection=config.get('model_selection')
    if adaptation.execution_busy(store,pid):raise ValueError('Chờ job đang chạy dừng trước khi kiểm tra.')
    if selection:
        registry={m['id']:m for m in load_catalog()['models']}
        connected=set()
        for intent in selection.values():
            if intent['kind']=='hub' and intent['id'] not in connected:
                connect(settings.data,registry[intent['id']],pid);connected.add(intent['id'])
        config=adaptation.save_model_selection(store,pid,settings,selection)
    save_input_setup(store,pid,settings,coordinates_confirmed,bin_columns,mmdet_input_mode)
    report=overview(store,pid)
    if report['can_check']:
        report['job']=adaptation.start_job(store,pid,settings,mode='verify')
        report['status']='checking'
    else:report['status']='blocked'
    receipt=root/'preparation_checks'/uuid.uuid4().hex/'report.json'
    report['receipt']=str(receipt);write(receipt,report);write(root/'preparation_latest.json',report)
    return report
