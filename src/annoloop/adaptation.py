"""Persistent native adaptation setup, checks and background dispatch."""
from copy import deepcopy
import json
import os
import signal
from pathlib import Path
import subprocess
import uuid
import yaml
from .io import read, write, digest, display_digest, now, lock
from .inspection import checkpoint_choices
from .reused.training_monitor import process_identity
from .runtime import require_runtime, worker_environment
from .settings import ROOT


ARCHITECTURES = {'A': 'SECONDNetIoU', 'B': 'PVRCNN'}


def model_choices(store, pid, settings):
    result = checkpoint_choices(store, pid, settings)
    setup_path = store.path(pid) / 'setup.json'
    if setup_path.exists():
        setup = read(setup_path)
        for key, path in setup.get('model_metadata', {}).items():
            if not path:
                continue
            from .compatibility import checkpoint_metadata
            metadata = checkpoint_metadata(path)
            config = metadata.get('config') or setup.get('specs', {}).get(key, {}).get('config')
            if config and Path(config).is_file():
                cfg = yaml.safe_load(Path(config).read_text())
                if metadata['classes'] == cfg['CLASS_NAMES'] and metadata['architecture'] == cfg['MODEL']['NAME']:
                    result.append(dict(path=metadata['checkpoint'], config=config,
                        classes=metadata['classes'], runtime=setup.get('specs', {}).get(key, {}).get('runtime',
                        str(Path(os.getenv('ANNOLOOP_ST3D_RUNTIME', ROOT / 'vendor/ST3D')))),
                        name=metadata['architecture'] + ' · ' + Path(metadata['checkpoint']).stem, intermediate=False))
    classes = set(store.get(pid)['schema']['classes'])
    available = {key: [] for key in ARCHITECTURES}
    seen = set()
    for candidate in result:
        cfg = yaml.safe_load(Path(candidate['config']).read_text())
        arch = cfg['MODEL']['NAME']
        if candidate['path'] in seen or not classes.issubset(candidate['classes']):
            continue
        if cfg.get('SELF_TRAIN') or cfg.get('DATA_CONFIG_TAR'):
            continue
        if cfg['DATA_CONFIG'].get('DATASET') != 'NuScenesDataset':
            continue
        if cfg['DATA_CONFIG']['POINT_FEATURE_ENCODING']['used_feature_list'] != ['x', 'y', 'z']:
            continue
        for key, expected in ARCHITECTURES.items():
            if arch == expected:
                available[key].append(dict(candidate, architecture=arch))
                seen.add(candidate['path'])
    from .hub_selection import candidates as hub_candidates
    for candidate in hub_candidates(settings.data,classes):
        for key in available:
            if not any(c['path']==candidate['path'] for c in available[key]):
                available[key].append(dict(candidate))
    return available


def save_setup(store, pid, settings, selected, coordinates_confirmed, bin_columns=4,
               epochs=1, learning_rate=.0001, source_weight=.5, adaptation_classes=None, memory_policy=None, model_selection=None, preparation_protocol=None,mmdet_input_mode=None):
    if type(epochs) is not int or epochs < 1 or not 0 < learning_rate <= .01 or not 0 < source_weight <= 1:
        raise ValueError('Epoch/LR/source weight không hợp lệ')
    if bin_columns not in (3, 4, 5):
        raise ValueError('BIN cần 3/4/5 cột')
    schema_classes = store.get(pid)['schema']['classes']
    chosen = schema_classes if adaptation_classes is None else adaptation_classes
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen).issubset(schema_classes):
        raise ValueError('Class adaptation phải là tập con không rỗng của class dự án')
    chosen = [name for name in schema_classes if name in chosen]
    from dataclasses import asdict
    from .pseudo_memory import MemoryPolicy
    memory = asdict(MemoryPolicy(**(memory_policy or {})))
    choices = model_choices(store, pid, settings)
    old_config=read(store.path(pid)/'adaptation.json') if (store.path(pid)/'adaptation.json').exists() else {}
    input_mode=mmdet_input_mode or old_config.get('mmdet_input_mode','strict')
    if input_mode not in ('strict','current_scan_only'):raise ValueError('MMDet input mode không hợp lệ')
    models = {}
    for key in ARCHITECTURES:
        candidate = next((c for c in choices[key] if c['path'] == selected.get(key)), None)
        if candidate:
            if candidate.get('backend')=='registry_framework':
                from .hub_selection import validate_cache
                validate_cache(settings.data,candidate)
            models[key] = dict(candidate, checkpoint_sha256=digest(candidate['path']),
                               config_sha256=digest(candidate['config']))
            if models[key].get('mmdet_recipe'):
                from .mmdet_recipe import effective
                models[key]['mmdet_recipe']=effective(models[key]['mmdet_recipe'],input_mode)
        elif selected.get(key):
            raise ValueError('Model '+key+' không còn trong danh sách; chọn lại checkpoint')
    document = dict(models=models, coordinates_confirmed=bool(coordinates_confirmed),
                    bin_columns=bin_columns, epochs=epochs, learning_rate=learning_rate,
                    source_weight=source_weight, adaptation_classes=chosen, tracking_mode='off',
                    memory_policy=memory, saved_at=now())
    if input_mode!='strict' or 'mmdet_input_mode' in old_config:document['mmdet_input_mode']=input_mode
    with lock(store.path(pid) / '.adaptation_dispatch.lock'):
        if execution_busy(store, pid):
            raise ValueError('Chờ adaptation/check GPU hiện tại dừng trước khi sửa cấu hình')
        old_path = store.path(pid) / 'adaptation.json'
        if preparation_protocol is not None:document['preparation_protocol']=preparation_protocol
        elif old_path.exists() and 'preparation_protocol' in read(old_path):
            document['preparation_protocol']=read(old_path)['preparation_protocol']
        if model_selection is not None:
            document['model_selection']=model_selection
        elif old_path.exists() and 'model_selection' in read(old_path):
            document['model_selection']=read(old_path)['model_selection']
        if store.get(pid)['completed_rounds'] and old_path.exists():
            old = read(old_path)
            if any(old.get(k) != document.get(k) for k in ('models', 'coordinates_confirmed', 'bin_columns')):
                raise ValueError('Sau vòng đầu, giữ profile/checkpoint khởi tạo và hệ tọa độ. Chọn/rollback vòng ở So sánh; đổi profile cần dự án mới')
            if old.get('adaptation_classes', schema_classes) != chosen:
                raise ValueError('Sau vòng đầu, giữ class adaptation để so sánh cùng phạm vi; đổi class cần dự án mới')
        write(store.path(pid) / 'adaptation.json', document)
    return document


def save_model_selection(store,pid,settings,selection,mmdet_input_mode=None):
    """One durable model selection for choose/prepare/adapt; no download or worker."""
    if set(selection)!={'A','B'}:raise ValueError('Chọn model cho cả A và B')
    from .model_hub import load_catalog,model_root
    registry={m['id']:m for m in load_catalog()['models']}
    choices=model_choices(store,pid,settings);paths={}
    for key,value in selection.items():
        if value['kind']=='hub':
            if value['id'] not in registry:raise ValueError('Model không còn trong danh mục')
            path=str(model_root(settings.data,value['id'])/'weights.pth')
            paths[key]=path if any(c['path']==path for c in choices[key]) else None
        elif value['kind']=='local':
            if not any(c['path']==value['path'] and c.get('backend','native_st3d')=='native_st3d' for c in choices[key]):
                raise ValueError('Checkpoint local không còn hợp lệ')
            paths[key]=value['path']
        elif value['kind']=='source':
            setup=read(store.path(pid)/'setup.json')
            native=[c for c in choices[key] if c.get('backend','native_st3d')=='native_st3d']
            if value.get('source_id'):
                from .native_sources import entry
                source=entry(value['source_id'])
                if source['architecture']!=ARCHITECTURES[key]:raise ValueError('Native source không đúng vai trò '+key)
                bound=setup.get('native_source_bindings',{}).get(key,{})
                if bound.get('id')!=source['id']:
                    paths[key]=None;continue
                # Only checkpoints descended from the chosen initializer may
                # satisfy this intent. Never pick an unrelated old project run.
                matching=[]
                for candidate in native:
                    run=candidate.get('origin_run');job=Path(run)/'job.json' if run else None
                    if job and job.is_file():
                        ref=read(job)['spec'].get('checkpoint_metadata')
                        if ref and checkpoint_origin(ref,source):matching.append(candidate)
                    elif candidate['path']==models_path(setup,key) and checkpoint_origin(setup['model_metadata'][key],source):
                        matching.append(candidate)
                native=matching
            if native:
                import re
                latest_run=native[0].get('origin_run')
                pool=[c for c in native if c.get('origin_run')==latest_run]
                pick=max(pool,key=lambda c:int(re.search(r'epoch_(\d+)',Path(c['path']).stem)[1]) if re.search(r'epoch_(\d+)',Path(c['path']).stem) else 0)
                paths[key]=pick['path']
            elif setup['plan']['models'][key]['pretrain_required']:paths[key]=None
            else:raise ValueError('Chưa có checkpoint nguồn cho model '+key)
        else:raise ValueError('Nguồn model không hợp lệ')
    old_path=store.path(pid)/'adaptation.json';old=read(old_path) if old_path.exists() else {}
    previous=old.get('model_selection') or {k:(dict(kind='hub',id=m['model_id']) if m.get('backend')=='registry_framework' else dict(kind='local',path=m['path'])) for k,m in old.get('models',{}).items()}
    if store.get(pid)['completed_rounds'] and previous!=selection:
        raise ValueError('Dự án đã có vòng hoàn tất; tạo dự án mới để đổi model gốc')
    return save_setup(store,pid,settings,paths,old.get('coordinates_confirmed',False),
        old.get('bin_columns',4),old.get('epochs',1),old.get('learning_rate',.0001),old.get('source_weight',.5),
        old.get('adaptation_classes'),old.get('memory_policy'),model_selection=selection,
        preparation_protocol=old.get('preparation_protocol') if store.get(pid)['completed_rounds'] else 1,
        mmdet_input_mode=mmdet_input_mode)


def models_path(setup,key):
    path=setup.get('model_metadata',{}).get(key)
    return read(path).get('checkpoint') if path else None


def checkpoint_origin(path,source):
    from .compatibility import checkpoint_metadata
    metadata=checkpoint_metadata(path)
    return (metadata.get('native_source_id')==source['id'] and metadata['checkpoint_sha256']==source['sha256']
            or metadata.get('transfer',{}).get('initial_checkpoint_sha256')==source['sha256'])


_FINISHED_CHILDREN = {}


def _child_receipts(root, finished):
    """Child *.process.json receipts. A finished job with a dead controller creates no new
    children, so its listing is reused; every child's liveness is still checked by the caller."""
    key = str(root)
    if finished and key in _FINISHED_CHILDREN:
        return _FINISHED_CHILDREN[key]
    children = [read(p) for d in ('*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*') for p in root.glob(d + '.process.json')]
    if finished:
        if len(_FINISHED_CHILDREN) > 512:
            _FINISHED_CHILDREN.clear()
        _FINISHED_CHILDREN[key] = children
    else:
        _FINISHED_CHILDREN.pop(key, None)
    return children


def job_status(root):
    root = Path(root)
    state = read(root / 'state.json')
    alive = bool(state.get('process_start') and process_identity(state.get('pid')) == state['process_start'])
    children_alive = False
    if not alive:
        # Children only matter for orphan detection; a live controller already means a live job.
        finished = state.get('status') not in ('starting', 'running')
        children = _child_receipts(root, finished)
        children_alive = any(c.get('process_start') and process_identity(c.get('pid')) == c['process_start'] for c in children)
        from .jobs import process_alive
        work = root.parents[1] / 'work'
        children_alive = children_alive or any(process_alive(p.parent) for pattern in ('M*/attempts/*/[AB]/training.json', 'M*/[AB]/training.json')
                        for p in work.glob(pattern))
    if not alive and children_alive:
        state = dict(state, status='orphaned_worker', error='Controller đã dừng nhưng worker còn sống; cần ngắt worker trước khi chạy lại')
    elif state['status'] in ('starting', 'running') and not alive:
        state = dict(state, status='interrupted')
    return dict(state, process_alive=bool(alive or children_alive), controller_alive=alive)


def active_job(store, pid):
    for path in (store.path(pid) / 'adaptation_jobs').glob('*/state.json'):
        if job_status(path.parent)['process_alive']:
            return str(path.parent)
    return None


def execution_busy(store,pid):
    """Guard configuration/dispatch without changing A/B monitor's job contract."""
    from .local_ensemble import active as ensemble_active
    from .final_delivery import active as delivery_active
    from .preparation_jobs import active as preparation_active
    return bool(active_job(store,pid) or ensemble_active(store,pid) or delivery_active(store,pid) or preparation_active(store,pid))


def stop_job(root):
    snapshot = job_status(root)
    if snapshot['controller_alive']:
        os.kill(snapshot['pid'], signal.SIGTERM)
        return True
    if snapshot['process_alive']:
        root = Path(root)
        for path in root.glob('**/*.process.json'):
            child = read(path)
            if child.get('process_start') and process_identity(child.get('pid')) == child['process_start']:
                os.killpg(child['pid'], signal.SIGTERM)
        from .jobs import stop
        for path in (root.parents[1] / 'work').glob('M*/**/[AB]/training.json'):
            stop(path.parent)
        return True
    return False


_CALIBRATION = {}


def _calibration_summary(path, chosen):
    """Small summary of a source-validation result; multi-MB JSON is re-read only when it changes."""
    from .io import file_signature
    from .fusion import source_calibration_report
    key = (str(path), file_signature(path), file_signature(path.parent / 'spec.json'), tuple(chosen))
    if key not in _CALIBRATION:
        spec, result = read(path.parent / 'spec.json'), read(path)
        records = result.get('calibration_records')
        report = source_calibration_report(records, chosen) if records else None
        summary = {k: result.get(k) for k in ('checkpoint_sha256', 'domain')}
        summary['frames'] = len(result.get('frames', []))
        if len(_CALIBRATION) > 64:
            _CALIBRATION.clear()
        _CALIBRATION[key] = (spec, summary, report)
    spec, summary, report = _CALIBRATION[key]
    return deepcopy(spec), dict(summary), deepcopy(report)


def calibration_reports(store, pid, models, source_evidence):
    """Read completed source-only evidence matching the currently selected weights.

    Reuse existing native inference reports for diagnostics, never predictions
    from a different checkpoint/config, a one-frame verify, or target gold.
    This function does not run inference or modify project storage.
    """
    from .fusion import source_calibration_report
    root = store.path(pid)
    project = store.get(pid)
    expected = {k: m['checkpoint_sha256'] for k, m in models.items()}
    if project['selected_version']:
        selected = read(root / 'rounds' / project['selected_version'] / 'round.json')
        expected = {k: m['sha256'] for k, m in selected['checkpoints'].items()}
    reports = {}
    jobs = sorted((root / 'adaptation_jobs').glob('*/spec.json'),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for job_path in jobs:
        if read(job_path).get('source_evidence') != source_evidence:
            continue
        for key, model in models.items():
            if key in reports:
                continue
            results = sorted((job_path.parent / key / 'source').glob('*/result.json'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
            for path in results:
                chosen = read(root / 'adaptation.json').get('adaptation_classes', project['schema']['classes'])
                spec, result, report = _calibration_summary(path, chosen)
                if (spec.get('checkpoint_sha256') != expected.get(key)
                    or spec.get('config_sha256') != model['config_sha256']
                    or spec.get('runtime') != model['runtime']
                    or spec.get('classes') != model['classes']
                    or spec.get('mode') != 'validation' or spec.get('limit') is not None
                    or spec.get('score_threshold') != .05
                    or result.get('checkpoint_sha256') != expected.get(key)
                    or result.get('domain') != 'source_validation'):
                    continue
                if report is None:
                    continue
                reports[key] = dict(report, path=str(path), checkpoint_sha256=expected[key],
                                    source_frames=result['frames'], score_threshold=.05)
                break
        if len(reports) == len(models):
            break
    return reports


def _point_contract(root, project, config):
    from .point_format import verification_contract
    return verification_contract(project['dataset_sha256'], config,
        [f for f in read(root / 'dataset.json')['frames'] if f['split'] == 'adaptation'])


_CONTRACTS = {}


def _display_contract(root, project, config):
    """Page-view cache: per-frame stats are slow on mounted drives; dispatch recomputes."""
    from .io import file_signature
    fields = ('bin_columns', 'coordinates_confirmed', 'mmdet_input_mode')
    key = (str(root), project['dataset_sha256'], file_signature(root / 'dataset.json'),
           json.dumps({k: config.get(k) for k in fields}, sort_keys=True))
    cached = _CONTRACTS.get(key)
    if cached:
        contract, formats = cached
        try:
            if all(file_signature(path) == signature for path, signature in formats.items()):
                return deepcopy(contract)
        except OSError:
            pass
    contract = _point_contract(root, project, config)
    formats = {path: file_signature(path) for path in contract.get('input_formats', {})}
    if len(_CONTRACTS) >= 32:
        _CONTRACTS.clear()
    _CONTRACTS[key] = (deepcopy(contract), formats)
    return contract


def readiness(store, pid, fast=False):
    """fast=True is for page views: file hashes are reused until file identity changes."""
    hasher = display_digest if fast else digest
    root = store.path(pid)
    project = store.get(pid)
    path = root / 'adaptation.json'
    config = read(path) if path.exists() else dict(models={}, coordinates_confirmed=False)
    profile = None
    blockers = []
    non_native = False
    implementation_missing = False
    if project['schema']['data_type'] != '3d':
        blockers.append('Detector 2D chưa có adapter')
    for key, arch in ARCHITECTURES.items():
        model = config['models'].get(key)
        if not model:
            if config.get('model_selection',{}).get(key,{}).get('kind')=='hub':
                blockers.append(f'Model {key}: đã chọn từ danh mục, cần hoàn tất bước Chuẩn bị model rồi cập nhật binding')
            else:
                blockers.append(f'Detector {key}: chọn checkpoint tại bước Class & model hỗ trợ ' + ', '.join(project['schema']['classes']))
        else:
            from .hub_selection import blockers as hub_blockers
            reasons=hub_blockers(model,config.get('adaptation_classes',project['schema']['classes']))
            if model.get('backend','native_st3d')!='native_st3d':
                recipe=model.get('mmdet_recipe')
                implementation_missing=implementation_missing or not (model.get('framework')=='mmdetection3d' and recipe and recipe.get('implementation_supported'))
                if recipe:
                    from .model_filter import dataset_profile
                    from .mmdet_recipe import input_blockers
                    if profile is None:
                        profile=dataset_profile([f for f in read(root/'dataset.json')['frames'] if f['split']=='adaptation'])
                    reasons=input_blockers(recipe,profile)+reasons
                    reasons=list(dict.fromkeys(reasons))
            non_native=non_native or bool(reasons)
            blockers.extend(f'Detector {key}: '+reason for reason in reasons)
            for field, hash_key in (('path', 'checkpoint_sha256'), ('config', 'config_sha256')):
                if not Path(model[field]).is_file() or hasher(model[field]) != model[hash_key]:
                    blockers.append(f'Detector {key}: checkpoint/config đã thay đổi, chọn lại')
    if not config.get('coordinates_confirmed'):
        blockers.append('Xác nhận target: mét, X trước / Y trái / Z lên trong hệ LiDAR')
    checked = root / 'native_verified.json'
    verified = read(checked) if checked.exists() else {}
    evidence_path = root / 'setup.json'
    source_evidence = read(evidence_path).get('source_evidence', {}) if evidence_path.exists() else {}
    source_valid = bool(source_evidence)
    for evidence in source_evidence.values():
        if not Path(evidence['path']).is_file() or hasher(evidence['path']) != evidence['sha256']:
            source_valid = False
    if not source_valid:
        blockers.append('Source infos train/validation thiếu hoặc đã thay đổi; chuẩn bị lại nguồn')
    expected = {key: {k: value[k] for k in ('checkpoint_sha256', 'config_sha256', 'runtime')}
                for key, value in config['models'].items()}
    try:
        input_contract=(_display_contract if fast else _point_contract)(root,project,config)
    except (ValueError,OSError) as error:
        input_contract=None
        blockers.append('Point cloud target: '+str(error))
    gpu_verified = bool(not non_native and verified.get('models') == expected and len(expected) == 2
                        and verified.get('source_evidence') == source_evidence and source_valid
                        and input_contract is not None and verified.get('point_contract') == input_contract
                        and verified.get('optimizer_steps') == 0)
    if not gpu_verified and not non_native:
        blockers.append('Chạy kiểm tra adapter GPU A → B trước khi train adaptation')
    can_check = not [b for b in blockers if not b.startswith('Chạy kiểm tra')]
    if (config.get('preparation_protocol')==1 or any(m.get('mmdet_recipe') for m in config['models'].values())) and not non_native:
        from .preparation import certificate_valid
        if not gpu_verified or not certificate_valid(root,verified,config,hasher):
            blockers.append('Kiểm chứng sau chuẩn bị chưa đạt đủ predict / backward / ignore / calibration. Mở Chuẩn bị model để kiểm tra.')
    reports = calibration_reports(store, pid, {k:m for k,m in config['models'].items() if m.get('backend','native_st3d')=='native_st3d' or m.get('mmdet_recipe',{}).get('implementation_supported')}, source_evidence)
    for key, report in reports.items():
        for row in report['classes']:
            if not row['enough_samples']:
                blockers.append(f"Detector {key} · {row['class_name']}: chỉ {row['samples']}/{report['minimum_samples']} "
                                f"dự đoán source validation để calibration "
                                f"({row['matched_at_iou_0_3']} box có IoU ≥ 0.3). "
                                'Cần checkpoint tốt hơn hoặc thêm source validation độc lập.')
    return dict(config=config, blockers=blockers, gpu_verified=gpu_verified,
                can_check=can_check, ready=not blockers, source_evidence=source_evidence,
                calibration_reports=reports,implementation_missing=bool(implementation_missing))


def start_job(store, pid, settings, mode='run', retry=False, resume_from=None):
    if mode not in ('run', 'verify'):
        raise ValueError('Adaptation job mode không hợp lệ')
    if retry and mode != 'run':
        raise ValueError('Chỉ chạy lại vòng adaptation, không gắn retry vào kiểm tra GPU')
    if resume_from and (mode != 'run' or retry):
        raise ValueError('Tiếp tục và bắt đầu lại là hai thao tác riêng')
    root = store.path(pid)
    with lock(root / '.adaptation_dispatch.lock'):
        if execution_busy(store, pid):
            raise ValueError('Job hiện tại còn sống; không khởi chạy lại')
        check = readiness(store, pid)
        if not (check['ready'] if mode == 'run' else check['can_check']):
            raise ValueError('; '.join(check['blockers']))
        if mode == 'run':
            index = store.next_round(pid)
            if index is None:
                raise ValueError('Dự án đã kết thúc hoặc hết số vòng đã chọn')
            exists = (root / 'work' / f'M{index}').exists()
            if exists and not retry and not resume_from:
                raise ValueError(f'Vòng M{index} bị ngắt; chọn chạy lại vòng để tạo lần thử mới, giữ artifacts cũ')
            if retry and not exists:
                raise ValueError('Không có vòng bị ngắt cần chạy lại')
        resume_plan = None
        if resume_from:
            from .resume import plan
            resume_plan = plan(store, pid, resume_from)
        inventory = store.verify_dataset(pid)
        for model in check['config']['models'].values():
            require_runtime(model, settings)
        from .device import gpu_idle
        gpu_idle()
        with lock(settings.data / '.gpu.lock', blocking=False):
            for existing in settings.data.glob('projects/*/adaptation_jobs/*/state.json'):
                if job_status(existing.parent)['process_alive']:
                    raise ValueError('Một controller/worker adaptation còn sống; chờ hoặc ngắt job đó')
            job = root / 'adaptation_jobs' / uuid.uuid4().hex
            document = dict(project_id=pid, mode=mode, config=check['config'],
                            source_evidence=check['source_evidence'], dataset_sha256=digest(root / 'dataset.json'),
                            retry_interrupted_round=retry, attempt_id=job.name, resume_plan=resume_plan)
            write(job / 'spec.json', document)
            write(job / 'state.json', dict(status='starting', mode=mode, created_at=now()))
            environment = worker_environment(settings)
            environment.update(ANNOLOOP_DATA_DIR=str(settings.data), VINAI_PYTHON=str(settings.python),
                               ANNOLOOP_WORKER_PYTHON=str(settings.trainer_python),
                               ANNOLOOP_TRAINING_LAUNCHER=str(settings.training_launcher))
            with (job / 'controller.log').open('w') as stream:
                process = subprocess.Popen([str(settings.trainer_python), '-u', '-m', 'annoloop.adaptation_worker',
                                            '--run', str(job)], cwd=ROOT, env=environment, stdout=stream,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            state = read(job / 'state.json')
            state.update(pid=process.pid, process_start=process_identity(process.pid))
            write(job / 'state.json', state)
            return str(job)
