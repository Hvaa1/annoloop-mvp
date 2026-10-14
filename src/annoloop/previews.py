"""Read saved target predictions, including interrupted rounds; never run a model."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .io import digest, read
from .predictions import validate_predictions


@lru_cache(maxsize=64)
def _hashes(path, modified, size):
    # Old aggregate files also contain many boxes; retain only their hash map.
    return read(path).get('checkpoint_hashes', {})


def checkpoint_hashes(directory):
    for name in ('checkpoints.json', 'predictions.json'):
        path = directory / name
        if path.is_file():
            stat = path.stat()
            return _hashes(str(path), stat.st_mtime_ns, stat.st_size)
    return {}


def _target_run(spec_path, expected, frame_lookup):
    spec = read(spec_path)
    if (spec.get('mode') != 'target_sample'
        or spec.get('config_sha256') != expected['config_sha256']
        or spec.get('runtime') != expected['runtime']
        or spec.get('classes') != expected['classes']):
        return None
    frames = spec.get('frames', [])
    available = {}
    for index, frame in enumerate(frames):
        known = frame_lookup.get(frame['frame_id'])
        if (not known or known['split'] != 'adaptation'
            or known['sha256'] != frame['sha256'] or known['media'] != frame['media']):
            return None
        path = spec_path.parent / 'frames' / f'{index:06d}.json'
        if path.is_file():
            available[frame['frame_id']] = str(path)
    result_path = spec_path.parent / 'result.json'
    complete = False
    if result_path.is_file():
        result = read(result_path)
        complete = (result.get('domain') == 'target_unlabelled'
                    and result.get('checkpoint_sha256') == spec['checkpoint_sha256']
                    and len(available) == len(frames))
    return dict(path=str(spec_path.parent), checkpoint=spec['path'],
                checkpoint_sha256=spec['checkpoint_sha256'], classes=spec['classes'],
                available=available, total=len(frames), completed=len(available), complete=complete)


def saved_attempts(store, pid):
    """List before/after A/B artifacts bound to the project dataset and weights."""
    from .adaptation import job_status
    root = store.path(pid)
    project = store.get(pid)
    if digest(root / 'dataset.json') != project['dataset_sha256']:
        raise ValueError('Dataset snapshot đã thay đổi; không ghép prediction với dữ liệu khác')
    inventory = read(root / 'dataset.json')
    frame_lookup = {f['frame_id']: f for f in inventory['frames']}
    attempts = []
    for path in (root / 'adaptation_jobs').glob('*/spec.json'):
        spec = read(path)
        if spec.get('mode') != 'run' or spec.get('dataset_sha256') != project['dataset_sha256']:
            continue
        state = job_status(path.parent)
        works = list((root / 'work').glob(f'M*/attempts/{path.parent.name}'))
        if not works and state.get('work_directory'):
            work = Path(state['work_directory'])
            if work.is_relative_to(root / 'work') and work.is_dir():
                works = [work]
        if not works and 'attempt_id' not in spec and (root / 'work/M1').is_dir():
            works = [root / 'work/M1']
        for work in works:
            version = work.relative_to(root / 'work').parts[0]
            before_hashes = checkpoint_hashes(work / 'before')
            if not before_hashes and version == 'M1':
                before_hashes = {k: m['checkpoint_sha256'] for k, m in spec['config']['models'].items()}
            after_hashes, trained = {}, {}
            for key in ('A', 'B'):
                receipt = work / (key + '_trained.json')
                if receipt.is_file():
                    trained[key] = read(receipt)
                    after_hashes[key] = trained[key]['checkpoint_sha256']
            phases = {'before': {}, 'after': {}}
            for key, model in spec['config']['models'].items():
                runs = sorted((path.parent / key / 'target').glob('*/spec.json'),
                              key=lambda p: p.stat().st_mtime, reverse=True)
                for run_path in runs:
                    run = _target_run(run_path, model, frame_lookup)
                    if not run:
                        continue
                    for phase, hashes in (('before', before_hashes), ('after', after_hashes)):
                        if key not in phases[phase] and run['checkpoint_sha256'] == hashes.get(key):
                            phases[phase][key] = run
            if any(run['available'] for phase in phases.values() for run in phase.values()):
                attempts.append(dict(id=version + '_' + path.parent.name, version=version,
                    job=str(path.parent), work=str(work), state=state, phases=phases, trained=trained,
                    adaptation_classes=spec['config'].get('adaptation_classes', project['schema']['classes']),
                    bin_columns=spec['config']['bin_columns']))
    return sorted(attempts, key=lambda a: (a['state'].get('created_at', ''), int(a['version'][1:])), reverse=True)


def prediction(run, frame_id):
    path = run['available'].get(frame_id)
    if path is None:
        return None
    document = read(path)
    if document.get('frame_id') != frame_id:
        raise ValueError('Frame của prediction không khớp point cloud')
    validate_predictions(document, run['classes'])
    return document


def point_preview(frame, bin_columns, max_points=12000):
    from .inspection import load_xyz
    if digest(frame['media']) != frame['sha256']:
        raise ValueError('Point cloud đã thay đổi so với dataset của lượt predict')
    points = load_xyz(frame['media'], bin_columns)
    indices = np.linspace(0, len(points) - 1, min(max_points, len(points)), dtype=int)
    return points[indices], len(points)


def inference_progress(job, state):
    key = state.get('active_detector')
    kind = {'predict_target': 'target', 'calibrate_source_validation': 'source'}.get(state.get('stage'))
    if key not in ('A', 'B') or kind is None:
        return None
    paths = sorted((Path(job) / key / kind).glob('*/inspection.json'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        return None
    progress = read(paths[0])
    completed = progress.get('completed_frames', 0)
    total = progress.get('total_frames', 0)
    return dict(detector=key, kind=kind, completed=completed, total=total,
                live=state['process_alive'], path=str(paths[0].parent))
