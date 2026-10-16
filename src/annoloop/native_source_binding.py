"""Bind pinned source initializers to project source training, without dispatch."""
from pathlib import Path
import os
import uuid
import yaml
from .io import read,write,lock,digest,now
from .settings import ROOT
from .native_sources import entry,root_for,ready
from .compatibility import checkpoint_metadata
from .setup import inspect_source,make_plan
from .pretrain import source_config
from .schema import ClassSchema


def bind(store,pid,settings,key,identifier):
    from .adaptation import active_job,ARCHITECTURES
    if key not in ARCHITECTURES:raise ValueError('Vai trò cần A hoặc B')
    model=entry(identifier)
    if model['architecture']!=ARCHITECTURES[key]:raise ValueError('Native source không đúng kiến trúc vai trò')
    if not ready(settings.data,model):raise ValueError('Cần tải/import native source đã xác minh hash')
    metadata_path=root_for(settings.data,identifier)/'checkpoint.metadata.json'
    metadata=checkpoint_metadata(metadata_path)
    if metadata.get('native_source_id')!=identifier or metadata['checkpoint_sha256']!=model['sha256'] or metadata['classes']!=model['classes']:
        raise ValueError('Metadata native source không khớp catalog')
    root=store.path(pid)
    with lock(root/'.adaptation_dispatch.lock'),lock(root/'.setup.lock'):
        if active_job(store,pid) or store.get(pid)['completed_rounds']:raise ValueError('Giữ model của dự án đã chạy; tạo dự án mới để thay initializer')
        setup=read(root/'setup.json')
        if setup['launches'] and setup['launches'][-1]['status'] not in ('completed','completed_with_stops','failed','dispatch_failed','cancelled'):
            raise ValueError('Chờ lượt source training dừng trước khi đổi initializer')
        source=inspect_source(setup['source_root'])
        if source['evidence']!=setup['source_evidence']:raise ValueError('Source đã thay đổi; tạo dự án mới')
        refs=dict(setup['model_metadata']);refs[key]=str(metadata_path)
        models={k:checkpoint_metadata(p) if p else None for k,p in refs.items()}
        classes=setup['plan']['classes'];plan=make_plan(classes,source,models)
        filename='secondiou' if key=='A' else 'pvrcnn'
        template=ROOT/'configs/source_templates'/f'{filename}_nuscenes_car_pedestrian_xyz.yaml'
        config=yaml.safe_load(template.read_text());epochs=20;ros=(.9,1.1)
        previous=setup.get('specs',{}).get(key)
        if previous:
            old=yaml.safe_load(Path(previous['config']).read_text());epochs=old['OPTIMIZATION']['NUM_EPOCHS']
            scaling=next((a for a in old['DATA_CONFIG']['DATA_AUGMENTOR']['AUG_CONFIG_LIST'] if a['NAME']=='random_object_scaling'),None)
            ros=scaling['SCALE_UNIFORM_NOISE'] if scaling else None
        overrides=[];known={a['class_name'] for a in config['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG']}
        profile=yaml.safe_load((Path(__file__).parent/'catalog'/model['profile']).read_text())
        for name in classes:
            if name in known:continue
            if not plan['models'][key]['pretrain_required']:
                anchor=next(a for a in profile['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG'] if a['class_name']==name)
                overrides.append(anchor);continue
            stats=source['anchor_statistics'].get(name)
            if not stats:raise ValueError('Source thiếu geometry để tạo anchor: '+name)
            shift=config['DATA_CONFIG'].get('SHIFT_COOR',[0,0,0])[2]
            overrides.append(dict(class_name=name,anchor_sizes=[stats['size']],anchor_rotations=[0,1.57],
                anchor_bottom_heights=[stats['bottom_z']+shift],align_center=False,feature_map_stride=8,matched_threshold=.5,unmatched_threshold=.35))
        schema=ClassSchema('3d','cuboid',tuple(classes))
        config=source_config(template,schema,source['loader_root'],epochs,ros,overrides)
        config['DATA_CONFIG']['VERSION']=source['version']
        # Reuse the exact original head/anchors when no new taxonomy is learned.
        # Never silently replace the pretrained anchors with target statistics.
        if not plan['models'][key]['pretrain_required']:
            if classes!=profile['CLASS_NAMES']:raise ValueError('Native direct reuse cần giữ thứ tự class gốc')
            config['MODEL']=profile['MODEL'];config['DATA_CONFIG']['POINT_CLOUD_RANGE']=profile['POINT_CLOUD_RANGE']
            for processor in config['DATA_CONFIG']['DATA_PROCESSOR']:
                if processor['NAME']=='transform_points_to_voxels':processor['VOXEL_SIZE']=profile['VOXEL_SIZE']
        folder=root/'generated/native_sources'/key/uuid.uuid4().hex;folder.mkdir(parents=True)
        config_path=folder/'source.yaml';config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        copied=folder/'checkpoint.metadata.json'
        write(copied,dict(metadata,config=str(config_path),config_sha256=digest(config_path)))
        refs[key]=str(copied);setup['model_metadata']=refs
        runtime=str(Path(os.getenv('ANNOLOOP_ST3D_RUNTIME',ROOT/'vendor/ST3D')).resolve())
        setup['specs'][key]=dict(config=str(config_path),runtime=runtime,seed=42,checkpoint_metadata=str(copied),
            source_train_annotations=str(root/'generated/source/source_train.json'),
            source_validation_annotations=str(root/'generated/source/source_validation.json'))
        plan['config_blockers']={k:v for k,v in setup['plan'].get('config_blockers',{}).items() if k!=key}
        setup['plan']=plan
        setup.setdefault('native_source_bindings',{})[key]=dict(id=identifier,sha256=model['sha256'],at=now(),
            pretrain_required=plan['models'][key]['pretrain_required'])
        write(root/'setup.json',setup)
        return setup
