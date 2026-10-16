"""Offline connection of an installed registry model to implemented adapters.

This connects maintained handlers; it does not generate missing training code.
No weights deserialization, upstream config execution, download or GPU action.
"""
import argparse
import json
from pathlib import Path
import uuid
from .io import read,write,now,lock
from .model_hub import load_catalog,model_root,installed
from .settings import Settings


def connect(data,model,project_id=None):
    from .adapter_factory import prepare_adapter,recipe_for
    root=model_root(data,model['id'])
    if not installed(data,model):
        raise ValueError('Model chưa tải đủ hoặc weights/config sai hash; không tạo kết nối')
    with lock(root/'connection.lock',blocking=False):
        receipt=read(root/'installed.json')
        report=dict(schema_version=1,model_id=model['id'],framework=model['framework'],
            architecture=model['architecture'],classes=model['classes'],checked_at=now(),
            checkpoint_sha256=receipt['sha256'],config_sha256=model['config_files'][model['config_path']]['sha256'],
            status='unsupported',adapter=None,runtime_verified=False,gpu_verified=False,
            strict_weights_load_verified=False,adaptation_ready=False,optimizer_steps=0,
            entrypoints={},missing_steps=[],target_gold_read=False)
        try:
            recipe=recipe_for(model)
        except ValueError as error:
            report['missing_steps']=[str(error)]
        else:
            profile=prepare_adapter(data,model)
            report.update(status='connected_unverified',adapter=str(root/'adapter.json'),
                capabilities=dict(raw_prediction_implemented=recipe['supports_raw_prediction'],
                    source_validation_implemented=False,calibration_implemented=False,
                    signed_ignore_implemented=False,target_training_implemented=False),
                entrypoints=dict(predict_raw='annoloop.adapter_factory:FrameworkAdapter.predict_raw',
                    worker='annoloop.framework_worker',predict_source_validation=None,
                    train_target=None,calibration=None),
                missing_steps=['Kiểm tra runtime và strict load weights.',
                    'Kiểm chứng predict / forward / backward trên source và target đúng feature.',
                    'Nối source validation và calibration vào DetectorAdapter.',
                    'Nối loss target và signed-ignore, checkpoint / resume vào DetectorAdapter.'])
            if profile['status']!='prepared_unverified':raise ValueError('Trạng thái binding không hợp lệ')
        if project_id:
            from .projects import ProjectStore
            from .model_filter import dataset_profile,assess
            store=ProjectStore(data);project=store.get(project_id)
            inventory=store.verify_dataset(project_id)
            setup=read(store.path(project_id)/'setup.json')
            config_path=store.path(project_id)/'adaptation.json'
            classes=read(config_path).get('adaptation_classes',project['schema']['classes']) if config_path.exists() else project['schema']['classes']
            profile=dataset_profile([f for f in inventory['frames'] if f['split']=='adaptation'])
            report['project_check']=dict(project_id=project_id,dataset_sha256=project['dataset_sha256'],
                input_profile=profile,assessment=assess(model,classes,profile,setup['source_counts']))
        # Preserve attempts; do not modify selection, verification or round receipts.
        attempt=root/'connections'/uuid.uuid4().hex/'report.json'
        report['receipt']=str(attempt)
        write(attempt,report);write(root/'connection.json',report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    choose=parser.add_mutually_exclusive_group(required=True)
    choose.add_argument('--model',help='ID model trong kho đã tải')
    choose.add_argument('--native-source',help='ID checkpoint ST3D/LSF đã pin')
    choose.add_argument('--all-installed',action='store_true',help='Chỉ xử lý model đã có installed.json, không tải mới')
    parser.add_argument('--data',type=Path,help='Thư mục lưu trạng thái AnnoLoop')
    parser.add_argument('--project',help='Kiểm tra thêm dữ liệu của dự án; không đổi A/B')
    parser.add_argument('--checkpoint',type=Path,help='Import file local đúng hash của --native-source, không tải mạng')
    parser.add_argument('--role',choices=['A','B'],help='Gắn native initializer vào cấu hình source của dự án')
    args=parser.parse_args(argv);data=args.data or Settings.load().data
    if args.native_source:
        if args.role and not args.project:parser.error('--role cần --project')
        if args.project and not args.role:parser.error('--project với --native-source cần --role A/B')
        from .native_sources import entry,install,ready,root_for
        try:
            model=entry(args.native_source)
            if args.checkpoint:install(data,model['id'],args.checkpoint)
            if not ready(data,model):raise ValueError('Chưa có native source đã xác minh hash; tải trong app hoặc import --checkpoint')
            if not (root_for(data,model['id'])/'checkpoint.metadata.json').exists():install(data,model['id'])
            report=dict(model_id=model['id'],metadata=str(root_for(data,model['id'])/'checkpoint.metadata.json'),
                status='native_source_registered',classes=model['classes'],adaptation_ready=False,gpu_verified=False)
            if args.project:
                from dataclasses import replace
                from .native_source_binding import bind
                from .projects import ProjectStore
                setup=bind(ProjectStore(data),args.project,replace(Settings.load(),data=data),args.role,model['id'])
                report.update(status='native_source_bound',plan=setup['plan']['models'][args.role])
            print(json.dumps(report,ensure_ascii=False,indent=2));return 0
        except (ValueError,OSError,KeyError) as error:
            print(json.dumps(dict(status='failed',error=str(error),adaptation_ready=False),ensure_ascii=False));return 1
    if args.checkpoint or args.role:parser.error('--checkpoint / --role chỉ dùng với --native-source')
    models=load_catalog()['models']
    if args.model:
        models=[m for m in models if m['id']==args.model]
        if not models:parser.error('Model không nằm trong registry')
    else:models=[m for m in models if (model_root(data,m['id'])/'installed.json').exists()]
    results=[];failed=False
    for model in models:
        try:results.append(connect(data,model,args.project))
        except (ValueError,OSError,KeyError,BlockingIOError) as error:
            failed=True;results.append(dict(model_id=model['id'],status='failed',error=str(error),adaptation_ready=False))
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
