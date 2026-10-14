"""Validate interrupted-round artifacts before explicitly resuming in a new folder."""
from pathlib import Path
from .io import read, digest
from .predictions import assert_no_gold, validate_predictions


def plan(store, pid, work):
    root = store.path(pid)
    work = Path(work).absolute()
    index = len(store.get(pid)['completed_rounds']) + 1
    if not work.is_relative_to(root / 'work' / f'M{index}' / 'attempts') or not work.is_dir():
        raise ValueError('Lượt cần tiếp tục không thuộc vòng chưa hoàn tất của dự án')
    job = root / 'adaptation_jobs' / work.name
    from .adaptation import job_status
    if job_status(job)['process_alive']:
        raise ValueError('Worker của lượt cũ còn sống; chưa thể tiếp tục')
    old = read(job / 'spec.json')
    current = read(root / 'adaptation.json')
    from dataclasses import asdict
    from .pseudo_memory import MemoryPolicy, validate as validate_memory
    memory_policy = asdict(MemoryPolicy(**current.get('memory_policy', {})))
    if asdict(MemoryPolicy(**old['config'].get('memory_policy', {}))) != memory_policy:
        raise ValueError('Thiết lập memory đã đổi; không tiếp tục nhãn/checkpoint cũ')
    project = store.get(pid)
    if old.get('mode') != 'run' or old['dataset_sha256'] != project['dataset_sha256']:
        raise ValueError('Dataset/profile của lượt cũ không khớp')
    evidence = read(root / 'setup.json')['source_evidence']
    if old['source_evidence'] != evidence:
        raise ValueError('Source infos đã đổi; không tiếp tục nhãn/checkpoint cũ')
    for field in ('coordinates_confirmed', 'bin_columns', 'epochs', 'learning_rate', 'source_weight', 'tracking_mode'):
        if old['config'].get(field) != current.get(field):
            raise ValueError('Thiết lập đã đổi: ' + field)
    classes = project['schema']['classes']
    if old['config'].get('adaptation_classes', classes) != current.get('adaptation_classes', classes):
        raise ValueError('Class adaptation đã đổi')
    expected = {k: m['checkpoint_sha256'] for k, m in current['models'].items()}
    if project['selected_version']:
        prior = root / 'rounds' / project['selected_version']
        previous = read(prior / 'round.json')
        expected = {k: m['sha256'] for k, m in previous['checkpoints'].items()}
        labels_path = prior / 'labels.json'
    else:
        labels_path = work / 'initial_shared.json'
    for key, model in current['models'].items():
        if any(model[f] != old['config']['models'][key][f] for f in
               ('checkpoint_sha256', 'config_sha256', 'runtime', 'classes', 'architecture')):
            raise ValueError('Checkpoint/config model đã đổi: ' + key)
    cache = work / 'before/predictions.json'
    cache_hash = None
    if cache.is_file():
        data = read(cache); assert_no_gold(data)
        if data['checkpoint_hashes'] != expected or set(data['models']) != {'A', 'B'}:
            raise ValueError('Cache trước train khác checkpoint khởi tạo')
        frame_ids = {f['frame_id'] for f in read(root / 'dataset.json')['frames'] if f['split'] == 'adaptation'}
        for frames in data['models'].values():
            if set(frames) != frame_ids:
                raise ValueError('Cache trước train thiếu frame')
            for boxes in frames.values():
                validate_predictions({'boxes': boxes}, classes)
        cache_hash = digest(cache)
    labels_hash = None
    if labels_path.is_file():
        labels = read(labels_path); assert_no_gold(labels)
        if labels.get('target_gold_read') is not False:
            raise ValueError('Nhãn cũ chưa xác nhận không dùng target gold')
        labels_hash = digest(labels_path)
    if labels_hash and not cache_hash:
        raise ValueError('Có nhãn cũ nhưng thiếu cache trước train để kiểm chứng')
    completed, checkpoints = {}, {}
    for key in ('A', 'B'):
        receipt_path = work / (key + '_trained.json')
        leaf = work / key
        if not receipt_path.is_file() and (leaf / 'trained.json').is_file():
            receipt_path = leaf / 'trained.json'
        if receipt_path.is_file():
            receipt = read(receipt_path)
            if (not cache_hash or not labels_hash or receipt.get('target_gold_used') is not False
                or receipt.get('source_assisted') is not True or receipt.get('optimizer_steps', 0) <= 0
                or receipt.get('epochs') != current['epochs']
                or receipt.get('classes') != current['models'][key]['classes']
                or receipt.get('architecture') != current['models'][key]['architecture']
                or receipt['shared_labels_sha256'] != labels_hash
                or digest(receipt['checkpoint']) != receipt['checkpoint_sha256']):
                raise ValueError('Receipt/checkpoint hoàn tất không hợp lệ: ' + key)
            completed[key] = receipt
        elif (leaf / 'spec.json').is_file():
            training_spec = read(leaf / 'spec.json')
            resume = training_spec.get('policy', {}).get('resume_checkpoint')
            original_resume = (old.get('resume_plan') or {}).get('checkpoints', {}).get(key)
            valid_weights = training_spec['checkpoint_sha256'] == expected[key]
            if resume:
                valid_weights = (resume == original_resume
                                 and training_spec['checkpoint_sha256'] == resume['sha256'])
            if (not cache_hash or training_spec.get('shared_labels_sha256') != labels_hash
                or not valid_weights
                or training_spec['config_sha256'] != current['models'][key]['config_sha256']):
                raise ValueError('Trainer dở không khớp nhãn/config: ' + key)
            available = list((leaf / 'checkpoints').glob('checkpoint_epoch_*.pth'))
            available = [p for p in available if 0 < int(p.stem.rsplit('_', 1)[1]) <= current['epochs']]
            if available:
                checkpoint = max(available, key=lambda p: int(p.stem.rsplit('_', 1)[1]))
                checkpoints[key] = dict(path=str(checkpoint), sha256=digest(checkpoint),
                                       epoch=int(checkpoint.stem.rsplit('_', 1)[1]))
            elif resume:
                # A resumed run may stop before its next save boundary. Keep the
                # last verified optimizer checkpoint instead of resetting it.
                if digest(resume['path']) != resume['sha256']:
                    raise ValueError('Checkpoint resume của lượt trước đã đổi')
                checkpoints[key] = resume
    memory_path = work / 'memory_before.json'
    memory_hash = None
    if memory_path.is_file():
        scope = current.get('adaptation_classes', classes)
        frame_ids = {f['frame_id'] for f in read(root / 'dataset.json')['frames'] if f['split']=='adaptation'}
        memory = validate_memory(read(memory_path), project['dataset_sha256'], scope, frame_ids)
        if memory['round_index'] != index - 1 or memory['policy'] != memory_policy:
            raise ValueError('Memory của lượt cũ khác vòng/thiết lập')
        memory_hash = digest(memory_path)
    return dict(work=str(work), job=str(job), index=index, completed=completed, checkpoints=checkpoints,
                labels_path=str(labels_path) if labels_hash else None, labels_sha256=labels_hash,
                before_cache=str(cache) if cache_hash else None, before_sha256=cache_hash,
                memory_before=str(memory_path) if memory_hash else None, memory_before_sha256=memory_hash,
                source_spec_sha256=digest(job / 'spec.json'))


def options(store, pid):
    index = len(store.get(pid)['completed_rounds']) + 1
    result = []
    for work in (store.path(pid) / 'work' / f'M{index}/attempts').glob('*'):
        try:
            result.append(plan(store, pid, work))
        except (ValueError, KeyError, OSError):
            continue
    return sorted(result, key=lambda p: (len(p['completed']), len(p['checkpoints']),
                  Path(p['work']).stat().st_mtime), reverse=True)
