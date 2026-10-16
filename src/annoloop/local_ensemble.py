"""Local checkpoint discovery and explicit sequential inference ensemble jobs.

Discovery never imports Torch or reads target annotation instances. This path
does not train N detectors or change the existing A/B adaptation history.
"""
from dataclasses import asdict
from pathlib import Path
import os
import signal
import subprocess
import uuid
import yaml
from .io import read, write, digest, now, lock
from .settings import ROOT
from .reused.training_monitor import process_identity
from .ensemble_fusion import EnsemblePolicy


def configuration(store, pid):
    path = store.path(pid) / 'ensemble.json'
    return read(path) if path.exists() else {}


def _source_paths(cfg):
    data = cfg.get('DATA_CONFIG', {})
    base = Path(data.get('DATA_PATH', '')) / data.get('VERSION', '')
    return {str((base / p).resolve()) for values in data.get('INFO_PATH', {}).values() for p in values}


def discover(store, pid, settings):
    """Known local receipts/configs only; eligibility still requires real strict load."""
    project_root = store.path(pid)
    setup = read(project_root / 'setup.json') if (project_root / 'setup.json').exists() else {}
    evidence = setup.get('source_evidence', {})
    expected_source = {str(Path(e['path']).resolve()) for e in evidence.values()}
    wanted = configuration(store, pid).get('classes')
    if not wanted:
        adaptation = project_root / 'adaptation.json'
        wanted = read(adaptation).get('adaptation_classes') if adaptation.exists() else None
    wanted = wanted or store.get(pid)['schema']['classes']
    found = {};configs={}
    def add(path, model, origin, blocked=None):
        path = Path(path).absolute()
        if not path.is_file() or str(path) in found:
            return
        row = {key:model[key] for key in ('config','runtime','classes','architecture','backend','origin_run') if key in model}
        row.update(path=str(path), origin=origin, bytes=path.stat().st_size,
                   name=model.get('name', path.parent.name + ' / ' + path.stem))
        reasons = list(blocked or [])
        config_path = Path(row.get('config', ''))
        if not blocked:
            try:
                if config_path not in configs:
                    configs[config_path]=yaml.safe_load(config_path.read_text())
                cfg=configs[config_path]
                row['classes'] = list(cfg['CLASS_NAMES'])
                row['architecture'] = cfg['MODEL']['NAME']
                data = cfg['DATA_CONFIG']
                if row['architecture'] not in ('SECONDNetIoU', 'PVRCNN'):
                    reasons.append('Chưa có calibration native cho kiến trúc này')
                if cfg.get('SELF_TRAIN') or cfg.get('DATA_CONFIG_TAR'):
                    reasons.append('Profile self-training/DSNorm cần adapter chuyên biệt')
                if data.get('DATASET') != 'NuScenesDataset' or data.get('MAX_SWEEPS', 1) != 1:
                    reasons.append('Native calibration hiện cần NuScenes một sweep')
                if data.get('POINT_FEATURE_ENCODING', {}).get('used_feature_list') != ['x', 'y', 'z']:
                    reasons.append('Native ensemble hiện cần input XYZ đã khai báo')
                if not expected_source or _source_paths(cfg) != expected_source:
                    reasons.append('Config chưa gắn đúng source train/validation của dự án')
                if not Path(row.get('runtime', '')).is_dir():
                    reasons.append('Thiếu runtime model')
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
                reasons.append('Thiếu hoặc lỗi config: ' + str(error)[:120])
        row.setdefault('classes', [])
        row.setdefault('architecture', 'Chưa xác minh')
        row['covered_classes'] = [c for c in wanted if c in row['classes']]
        if not row['covered_classes']:
            reasons.append('Không có class yêu cầu được xác minh; không tự đổi Vehicle thành car')
        row['vote_group'] = row['architecture']
        row['reasons'] = reasons
        row['eligible'] = not reasons
        found[str(path)] = row
    # Source checkpoints from all local projects, not just the current A/B pair.
    for receipt in sorted((settings.data / 'jobs').glob('*/job.json')):
        try:
            doc = read(receipt)
            if doc.get('kind') != 'source_pretrain':
                continue
            for path in sorted((receipt.parent / 'checkpoints').glob('*.pth')):
                add(path, doc['spec'], 'source · ' + receipt.parent.name[:8])
        except (OSError, ValueError, KeyError):
            continue
    trained_receipts = list((settings.data / 'projects').glob('*/work/M*/attempts/*/*/trained.json'))
    trained_receipts += list((settings.data / 'experiments').rglob('trained.json'))
    for receipt in sorted(trained_receipts):
        try:
            trained = read(receipt); spec = read(receipt.parent / 'spec.json')
            if trained.get('target_gold_used') is not False:
                continue
            path = trained.get('checkpoint')
            if path:
                origin = 'thử nghiệm' if receipt.is_relative_to(settings.data / 'experiments') else 'adaptation'
                add(path, spec, origin + ' · ' + receipt.parent.name)
        except (OSError, ValueError, KeyError):
            continue
    for parent in sorted((settings.data / 'projects').glob('*')):
        config = parent / 'adaptation.json'
        if config.is_file():
            try:
                for model in read(config).get('models', {}).values():
                    add(model['path'], model, 'model dự án · ' + parent.name[:8])
            except (OSError, ValueError, KeyError):
                pass
    # Locally imported official weights must be visible even when not runnable.
    for receipt in sorted((settings.data / 'model_imports').glob('*/*/provenance.json')):
        try:
            model = read(receipt)
            add(receipt.parent / 'weights.pth', dict(name=model.get('name', receipt.parent.name),
                classes=model.get('class_names_from_config', []),
                config=str(receipt.parent / 'upstream_config.yaml'),
                architecture='PVRCNN' if 'pvrcnn' in receipt.parent.name else 'SECONDNetIoU',
                sha256=model.get('sha256')), 'ST3D chính thức · đã tải',
                ['Chưa strict-load weights, xác minh head/class và nối source calibration'])
        except (OSError, ValueError, KeyError):
            continue
    for receipt in sorted((settings.data / 'model_cache').glob('*/installed.json')):
        try:
            model = read(receipt)
            add(model['weights'], model, 'kho model · đã tải',
                ['Framework adapter chưa có native source calibration/quality contract'])
        except (OSError, ValueError, KeyError):
            continue
    return sorted(found.values(), key=lambda m: (not m['eligible'], m['architecture'], m['path']))


def status(root):
    state = read(Path(root) / 'state.json')
    alive = bool(state.get('process_start') and process_identity(state.get('pid')) == state['process_start'])
    if state['status'] in ('starting', 'running') and not alive:
        state = dict(state, status='interrupted')
    return dict(state, process_alive=alive)


def jobs(store, pid):
    return sorted((store.path(pid) / 'ensemble_jobs').glob('*/state.json'),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def active(store, pid):
    return any(status(p.parent)['process_alive'] for p in jobs(store, pid))


def save(store, pid, settings, paths, classes, coordinates_confirmed, bin_columns=4, policy=None):
    from .adaptation import execution_busy
    if store.get(pid)['schema']['data_type'] != '3d':
        raise ValueError('Ensemble 2D chưa có detector/calibration adapter')
    policy = asdict(EnsemblePolicy(**(policy or {})))
    if bin_columns not in (3, 4, 5) or not coordinates_confirmed:
        raise ValueError('Cần xác nhận XYZ theo mét, X trước/Y trái/Z lên và số cột BIN')
    if not classes or not set(classes).issubset(store.get(pid)['schema']['classes']):
        raise ValueError('Chọn class thuộc schema dự án')
    available = {m['path']: m for m in discover(store, pid, settings)}
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Chọn checkpoint không trùng')
    models, seen = [], set()
    for path in paths:
        model = available.get(path)
        if not model or not model['eligible']:
            raise ValueError('Checkpoint chưa đủ điều kiện native ensemble: ' + path)
        identity = digest(path)
        if identity in seen:
            continue  # Round snapshots/copies of the same weights are aliases.
        seen.add(identity)
        models.append(dict(model, checkpoint_sha256=identity, config_sha256=digest(model['config'])))
    if not set(classes).issubset({c for m in models for c in m['classes']}):
        raise ValueError('Các model đã chọn chưa phủ đủ class')
    with lock(store.path(pid) / '.adaptation_dispatch.lock'):
        if execution_busy(store, pid):
            raise ValueError('Chờ job hiện tại kết thúc trước khi đổi workflow')
        doc = dict(enabled=True, models=models, classes=classes, bin_columns=bin_columns,
                   coordinates_confirmed=True, policy=policy, saved_at=now())
        write(store.path(pid) / 'ensemble.json', doc)
    return doc


def disable(store, pid):
    from .adaptation import execution_busy
    with lock(store.path(pid) / '.adaptation_dispatch.lock'):
        if execution_busy(store, pid):
            raise ValueError('Chờ job hiện tại kết thúc')
        doc = configuration(store, pid)
        write(store.path(pid) / 'ensemble.json', dict(doc, enabled=False))


def start(store, pid, settings):
    from .runtime import require_runtime, worker_environment
    from .adaptation import execution_busy
    from .device import gpu_idle
    config = configuration(store, pid)
    if not config.get('enabled') or not config.get('models'):
        raise ValueError('Lưu lựa chọn ghép checkpoint trước')
    store.verify_dataset(pid)
    setup = read(store.path(pid) / 'setup.json')
    evidence = setup.get('source_evidence', {})
    if not evidence:
        raise ValueError('Thiếu source train/validation độc lập')
    for item in evidence.values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Source infos đã thay đổi')
    for model in config['models']:
        if digest(model['path']) != model['checkpoint_sha256'] or digest(model['config']) != model['config_sha256']:
            raise ValueError('Checkpoint/config đã thay đổi; chọn lại')
        require_runtime(model, settings)
    with lock(store.path(pid) / '.adaptation_dispatch.lock'), lock(settings.data / '.gpu.lock', blocking=False):
        if execution_busy(store, pid):
            raise ValueError('Một job dự án đang chạy')
        gpu_idle()
        root = store.path(pid) / 'ensemble_jobs' / uuid.uuid4().hex
        write(root / 'spec.json', dict(project_id=pid, config=config, source_evidence=evidence,
              dataset_sha256=digest(store.path(pid) / 'dataset.json')))
        write(root / 'state.json', dict(status='starting', stage='queued', created_at=now()))
        env = dict(worker_environment(settings), ANNOLOOP_DATA_DIR=str(settings.data))
        with (root / 'worker.log').open('w') as log:
            process = subprocess.Popen([str(settings.python), '-u', '-m', 'annoloop.ensemble_worker',
                '--run', str(root)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        write(root / 'state.json', dict(status='starting', stage='queued', created_at=now(),
              pid=process.pid, process_start=process_identity(process.pid)))
    return root


def stop(root):
    state = status(root)
    if state['process_alive']:
        # The controller handles SIGTERM and waits for its inference child to exit.
        os.kill(state['pid'], signal.SIGTERM)
