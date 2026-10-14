"""Folder-based onboarding, generated artifacts and conditional source training.

Only explicit source folders are inspected for labels. Target annotations are never
read. Files are generated in project storage, leaving input datasets untouched.
"""
from collections import Counter
from importlib import import_module
import os
from pathlib import Path
import pickle
import numpy as np
import yaml
from .compatibility import checkpoint_metadata
from .inventory import EXTENSIONS, source_counts
from .io import digest, read, write, lock, now
from .pretrain import source_config
from .schema import ClassSchema, discover
from .settings import ROOT, WORKSPACE


class SourceInfoUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            # NumPy 1.26 exposes an _core compatibility stub before importing
            # its children. Choose/import the actual module, not that stub.
            namespace = 'numpy._core' if int(np.__version__.split('.')[0]) >= 2 else 'numpy.core'
            return getattr(import_module(namespace + '.multiarray'), name)
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module == 'numpy.dtypes':
            value = getattr(np.dtypes, name, None)
            if isinstance(value, type) and issubclass(value, np.dtype):
                return value
        raise ValueError('Infos source chứa kiểu pickle chưa hỗ trợ: '+module+'.'+name)


def source_infos(path):
    with Path(path).open('rb') as stream:
        document = SourceInfoUnpickler(stream).load()
    if not isinstance(document, list) or not document:
        raise ValueError('Infos source cần danh sách frame có nhãn')
    return document


def inspect_source(folder):
    """Currently a nuScenes prepared-infos adapter. Never guesses train/val splits."""
    selected = Path(folder).expanduser().resolve()
    if not selected.is_dir():
        raise ValueError('Folder source không tồn tại')
    candidates = [selected] + sorted(p for p in selected.iterdir() if p.is_dir() and p.name.startswith('v1.0-'))
    valid = [p for p in candidates if (p/'nuscenes_infos_1sweeps_train.pkl').is_file() and (p/'nuscenes_infos_1sweeps_val.pkl').is_file()]
    if len(valid) != 1:
        raise ValueError('Chọn folder nuScenes đã chuẩn bị có một bộ infos 1-sweep train/val. Adapter folder source hiện hỗ trợ nuScenes; định dạng khác cần thêm adapter.')
    root = valid[0]
    if not root.name.startswith('v1.0-'):
        raise ValueError('Folder infos nuScenes cần nằm trong v1.0-mini/v1.0-trainval')
    samples_path = root/root.name/'sample.json'
    samples = {r['token']:r for r in read(samples_path)} if samples_path.is_file() else {}
    frames, evidence, counts, scenes = {}, {}, {}, {}
    geometry={}
    for split, suffix in [('train','train'),('validation','val')]:
        path = root/f'nuscenes_infos_1sweeps_{suffix}.pkl'
        rows = source_infos(path)
        converted, seen = [], set()
        counter, sequence_ids = Counter(), set()
        for row in rows:
            token = str(row['token'])
            if token in seen:
                raise ValueError('Infos source trùng frame')
            seen.add(token)
            names = [str(n) for n in row['gt_names']]
            scene = samples.get(token,{}).get('scene_token')
            if samples and scene is None:
                raise ValueError('Infos không khớp metadata scene nuScenes')
            if scene:
                sequence_ids.add(scene)
            boxes = [dict(class_name=n, ignore=n=='ignore') for n in names]
            if split=='train' and 'gt_boxes' in row:
                values=np.asarray(row['gt_boxes'])
                if values.ndim!=2 or len(values)!=len(names) or values.shape[1]<7:
                    raise ValueError('Source infos có geometry không khớp class')
                for name,value in zip(names,values):
                    if name!='ignore' and np.isfinite(value[:7]).all() and (value[3:6]>0).all():
                        geometry.setdefault(name,[]).append([*value[3:6],value[2]-value[5]/2])
            counter.update(n for n in names if n!='ignore')
            converted.append(dict(frame_id=token,sequence_id=scene,boxes=boxes))
        frames[split]=converted
        counts[split]=dict(counter)
        scenes[split]=sequence_ids
        evidence[split]=dict(path=str(path),sha256=digest(path))
    if {r['frame_id'] for r in frames['train']} & {r['frame_id'] for r in frames['validation']}:
        raise ValueError('Source train/validation trùng frame')
    if scenes['train'] & scenes['validation']:
        raise ValueError('Source train/validation trùng scene')
    names = sorted(set(counts['train']) | set(counts['validation']))
    schema = ClassSchema.parse(dict(data_type='3d', annotation_type='cuboid',classes=names))
    anchors={name:dict(size=np.median(np.array(values)[:,:3],axis=0).tolist(),bottom_z=float(np.median(np.array(values)[:,3])),samples=len(values)) for name,values in geometry.items()}
    return dict(adapter='nuscenes_1sweep',root=str(root),loader_root=str(root.parent),version=root.name,anchor_statistics=anchors,
                schema=schema.as_dict(),frames=frames,counts=counts,evidence=evidence,
                scene_split_verified=bool(samples),scene_metadata=dict(path=str(samples_path),sha256=digest(samples_path)) if samples else None)


def inspect_target(folder):
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Folder target không tồn tại')
    schema_files = any((root/name).is_file() for name in ('class_schema.json','manifest.json','dataset.json'))
    schema = None
    if schema_files:
        try:
            schema,_=discover(root)
        except ValueError as error:
            if not str(error).startswith('Thiếu schema'):
                raise
    counts={mode:sum(1 for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions) for mode,extensions in EXTENSIONS.items()}
    mode=schema.data_type if schema else '3d' if counts['3d'] else '2d'
    if not counts[mode]:
        raise ValueError('Target không có media 3D hoặc ảnh được hỗ trợ')
    return dict(root=str(root),data_type=mode,frame_count=counts[mode],classes=list(schema.classes) if schema else [])


def model_catalog(data, extra_folders=()):
    """A human-facing catalog; users select checkpoints instead of metadata JSON."""
    roots=[ROOT/'resources/checkpoints', Path(data)/'models', Path(data)/'jobs',*map(Path,extra_folders)]
    result=[];seen=set()
    for root in roots:
        paths = sorted(root.rglob('*.metadata.json')) + sorted(root.rglob('*.pth.metadata.json')) if root.is_dir() else []
        for path in paths:
            try:
                meta=read(path)
                if not meta.get('classes_verified') or meta.get('architecture') not in ('SECONDNetIoU','PVRCNN'):
                    continue
                weight=Path(meta['checkpoint']).expanduser()
                if not weight.is_absolute():weight=path.parent/weight
                if not weight.is_file() or str(weight.resolve()) in seen:continue
                seen.add(str(weight.resolve()))
                result.append(dict(metadata=str(path.resolve()),checkpoint=str(weight.resolve()),architecture=meta['architecture'],classes=meta['classes'],
                    name=f"{weight.stem} · {', '.join(meta['classes'])}"))
            except (ValueError,KeyError,OSError,TypeError):
                continue
    return result


def make_plan(classes, source, models):
    wanted=ClassSchema.parse(dict(data_type='3d',annotation_type='cuboid',classes=list(classes))).classes
    if set(models)!={'A','B'}:raise ValueError('Chọn hai detector A/B')
    plan={}
    source_blockers=[]
    for name in wanted:
        if source['counts']['train'].get(name,0)==0:source_blockers.append('Source train thiếu '+name)
        if source['counts']['validation'].get(name,0)==0:source_blockers.append('Source validation thiếu '+name)
    for key, expected in [('A','SECONDNetIoU'),('B','PVRCNN')]:
        model=models[key]
        if model is not None and (model.get('architecture')!=expected or not model.get('classes_verified')):
            raise ValueError('Checkpoint '+key+' chưa xác minh đúng architecture/class')
        missing=[c for c in wanted if model is None or c not in model['classes']]
        plan[key]=dict(architecture=expected,missing_classes=missing,pretrain_required=bool(missing),
                       action='pretrain' if missing else 'reuse',
                       reason='Cần học: '+', '.join(missing) if missing else 'Checkpoint đã hỗ trợ đủ class — bỏ qua pretrain')
    return dict(models=plan,classes=list(wanted),source_blockers=source_blockers,source_ready=not source_blockers)


def prepare_project(store, settings, name, source, target, classes, models, rounds=3, execution='manual',policy=None,epochs=20,ros=(.9,1.1)):
    """Generate JSON/YAML automatically; preparation never starts a trainer."""
    if target['data_type']!='3d':raise ValueError('Workflow hai detector hiện hỗ trợ 3D; 2D chưa có adapter model')
    verified={key:checkpoint_metadata(path) if path else None for key,path in models.items()}
    # Read source fresh to ensure counts/evidence were not stale from a UI preview.
    source=inspect_source(source['root'])
    plan=make_plan(classes,source,verified)
    schema=ClassSchema('3d','cuboid',tuple(classes))
    runtime=Path(os.getenv('ANNOLOOP_ST3D_RUNTIME',ROOT/'vendor/ST3D')).resolve()
    config_documents={};config_blockers={}
    for key,item in plan['models'].items():
        if not item['pretrain_required']:continue
        filename='secondiou' if key=='A' else 'pvrcnn'
        template=ROOT/'configs/source_templates'/f'{filename}_nuscenes_car_pedestrian_xyz.yaml'
        try:
            if plan['source_blockers']:raise ValueError('; '.join(plan['source_blockers']))
            base=yaml.safe_load(template.read_text())
            native={a['class_name'] for a in base['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG']}
            overrides=[]
            for name in classes:
                if name in native:continue
                stats=source['anchor_statistics'].get(name)
                if not stats:raise ValueError('Source infos thiếu geometry để tự tạo anchor cho '+name)
                shift=base['DATA_CONFIG'].get('SHIFT_COOR',[0,0,0])[2]
                overrides.append(dict(class_name=name,anchor_sizes=[stats['size']],anchor_rotations=[0,1.57],
                    anchor_bottom_heights=[stats['bottom_z']+shift],align_center=False,feature_map_stride=8,
                    matched_threshold=.5,unmatched_threshold=.35))
            config=source_config(template,schema,source['loader_root'],epochs,ros,anchor_overrides=overrides)
            config['DATA_CONFIG']['VERSION']=source['version']
            config_documents[key]=config
        except ValueError as error:
            config_blockers[key]=str(error)
    pid=store.create(name,target['root'],rounds,execution,policy,schema_override=schema)
    destination=store.path(pid)/'generated'
    source_dir=destination/'source'
    write(source_dir/'class_schema.json',source['schema'])
    inventories={}
    for split in ('train','validation'):
        path=source_dir/f'source_{split}.json'
        write(path,dict(role='source',split=split,frames=source['frames'][split],origin=source['evidence'][split],inventory_only=True))
        assert source_counts(path,ClassSchema.parse(source['schema']))['split']==split
        inventories[split]=str(path)
    model_refs={}
    for key,metadata in verified.items():
        if metadata is None:
            model_refs[key]=None
            continue
        copied=destination/'models'/f'{key}.metadata.json'
        write(copied,{**metadata,'checkpoint':metadata['checkpoint']})
        model_refs[key]=str(copied)
    specs={}
    for key,config in config_documents.items():
        path=destination/'configs'/f'{key}_source.yaml';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(config,sort_keys=False))
        specs[key]=dict(config=str(path),runtime=str(runtime),source_train_annotations=inventories['train'],
                        source_validation_annotations=inventories['validation'],seed=42)
        if model_refs[key]:specs[key]['checkpoint_metadata']=model_refs[key]
    plan['config_blockers']=config_blockers
    setup=dict(source_root=source['root'],loader_root=source['loader_root'],target_root=target['root'],
               source_counts=source['counts'],source_evidence=source['evidence'],scene_metadata=source['scene_metadata'],anchor_statistics=source['anchor_statistics'],
               model_metadata=model_refs,plan=plan,specs=specs,created_at=now(),launches=[])
    write(store.path(pid)/'setup.json',setup)
    write(destination/'job_specs.json',specs)
    return pid


def launch_required(store, pid, settings, detectors=None):
    """Launch only missing-class models, once; reruns and existing-class paths do nothing."""
    from . import jobs
    root=store.path(pid)
    with lock(root/'.setup.lock'):
        setup=read(root/'setup.json')
        required=[key for key,item in setup['plan']['models'].items() if item['pretrain_required']]
        if not required:return dict(action='reuse',jobs_started=0)
        if detectors is not None:
            if not detectors or len(detectors) != len(set(detectors)) or any(key not in required for key in detectors):
                raise ValueError('Chọn ít nhất một detector còn cần học nguồn')
            required=[key for key in ('A','B') if key in detectors]
        if setup['launches'] and setup['launches'][-1]['status'] not in ('failed','dispatch_failed','completed','completed_with_stops','cancelled'):
            raise ValueError('Source job đã được gửi; xem trạng thái, không khởi chạy lại')
        if setup['plan']['source_blockers'] or setup['plan']['config_blockers']:
            raise ValueError('Source chưa sẵn sàng; xem class thiếu hoặc profile chưa hỗ trợ')
        for evidence in setup['source_evidence'].values():
            if digest(evidence['path'])!=evidence['sha256']:raise ValueError('Source infos đã thay đổi; chuẩn bị dự án mới')
        for key in required:
            if setup['model_metadata'][key]:checkpoint_metadata(setup['model_metadata'][key])
        from .runtime import require_runtime
        # Fail early before registering a retry or GPU trainer.
        checked=set()
        for key in required:
            runtime=setup['specs'][key]['runtime']
            if runtime not in checked:
                require_runtime(setup['specs'][key],settings)
                checked.add(runtime)
        setup['launches'].append(dict(status='dispatching',at=now()))
        write(root/'setup.json',setup)
        try:
            if len(required)==2:directory=jobs.start_source_pair(setup['specs'],settings)
            else:directory=jobs.start_pretrain(setup['specs'][required[0]],settings)
            result=dict(action='pretrain',jobs_started=len(required),path=directory,detectors=required)
            setup['launches'][-1].update(status='submitted',**result)
        except BaseException as error:
            setup['launches'][-1].update(status='dispatch_failed',error=str(error))
            write(root/'setup.json',setup)
            raise
        write(root/'setup.json',setup)
        return result


def refresh_training_result(store, pid):
    """Promote only finished, validated source checkpoints; never starts/restarts work."""
    from . import jobs
    root=store.path(pid)
    with lock(root/'.setup.lock'):
        setup=read(root/'setup.json')
        if not setup['launches']:return setup
        launch=setup['launches'][-1]
        if launch['status']!='submitted':return setup
        directory=Path(launch['path'])
        required=[key for key,item in setup['plan']['models'].items() if item['pretrain_required']]
        if (directory/'pair.json').is_file():
            pair=read(directory/'pair.json') if (directory/'pair.json').is_file() else {}
            runs=pair['jobs']
            if pair.get('status') not in ('completed','completed_with_stops','cancelled'):
                from .reused.training_monitor import process_identity
                controller=read(directory/'controller.json') if (directory/'controller.json').is_file() else {}
                alive=controller.get('process_start') is not None and process_identity(controller.get('pid'))==controller['process_start']
                if not alive and all(not jobs.status(run)['process_alive'] for run in runs.values()):
                    error=pair.get('error') or 'Controller đã dừng trước khi hoàn tất; giữ artifacts, cần lượt chạy mới'
                    setup['launches'][-1].update(status='failed',error=error)
                    write(root/'setup.json',setup)
                return setup
        else:
            pair=None
            runs={launch.get('detectors',required)[0]:str(directory)}
        metadata={key:checkpoint_metadata(path) if path else None for key,path in setup['model_metadata'].items()}
        learned_runs={}
        stopped=[]
        for key,run in runs.items():
            if not (Path(run)/'job.json').is_file():return setup
            snapshot=jobs.status(run)
            if pair and not snapshot['process_alive'] and snapshot['training'].get('status')=='stopped':
                stopped.append(key)
                continue
            if not snapshot['process_alive'] and snapshot['training'].get('status') in ('failed','stopped','interrupted'):
                setup['launches'][-1].update(status='failed',error='Source job '+snapshot['training']['status'])
                write(root/'setup.json',setup)
                return setup
            if snapshot['process_alive'] or snapshot['training'].get('status')!='completed':return setup
            final=Path(run)/'checkpoint.metadata.json'
            learned=checkpoint_metadata(final)
            if learned.get('target_gold_read') is not False or learned.get('optimizer_steps',0)<=0:
                raise ValueError('Checkpoint nguồn chưa xác nhận run hoàn tất')
            if not set(setup['plan']['classes']).issubset(learned['classes']):
                raise ValueError('Checkpoint nguồn cuối chưa hỗ trợ đủ class đã yêu cầu')
            metadata[key]=learned
            learned_runs[key]=run
        source=dict(counts=setup['source_counts'])
        new_plan=make_plan(setup['plan']['classes'],source,metadata)
        new_plan['config_blockers']={}
        for key,run in learned_runs.items():setup['model_metadata'][key]=str(Path(run)/'checkpoint.metadata.json')
        setup['plan']=new_plan
        setup['launches'][-1]['status']=pair['status'] if pair else 'completed'
        if stopped:setup['launches'][-1]['stopped_detectors']=stopped
        write(root/'setup.json',setup)
        return setup
