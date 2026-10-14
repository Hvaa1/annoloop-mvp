"""Load/rebuild memory from immutable machine-label snapshots, never target annotations."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from .io import read, digest
from .predictions import assert_no_gold
from .pseudo_memory import empty, validate, update_frame, summary


def context(index, hashes, origin='round_prediction', **inputs):
    return dict(round_index=index, checkpoint_hashes=hashes, origin=origin, inputs=inputs)


def _legacy_source(root, version, frames, classes):
    directory = root / 'rounds' / version
    receipt = read(directory / 'round.json')
    path = directory / 'labels.json'
    if digest(path) != receipt['labels_sha256']:
        raise ValueError('Nhãn snapshot đã đổi; không dựng memory: ' + version)
    labels = read(path); assert_no_gold(labels)
    rows = {f['frame_id']: f for f in labels['frames']}
    if len(rows) != len(labels['frames']) or set(rows) != set(frames):
        raise ValueError('Snapshot memory khác adaptation frames')
    hashes = {k: r['sha256'] for k, r in receipt['checkpoints'].items()}
    work = Path(receipt['work_directory']) if receipt.get('work_directory') else root / 'work' / version
    cache = work / 'after/predictions.json'
    raw = {}
    if cache.is_file():
        document = read(cache); assert_no_gold(document)
        if document['checkpoint_hashes'] != hashes or set(document['models']) != {'A', 'B'}:
            raise ValueError('Predict snapshot khác checkpoint khi dựng memory')
        raw = document['models']
        if any(set(model) != set(frames) for model in raw.values()):
            raise ValueError('Predict snapshot memory thiếu frame')
    return rows, raw, context(int(version[1:]), hashes, 'legacy_snapshot',
        version=version, labels_sha256=digest(path), raw_sha256=digest(cache) if cache.is_file() else None), work


def load(store, pid, frames, classes, policy, point_loader):
    """Latest observations include rejected rounds, but training still uses selected labels."""
    project = store.get(pid)
    root = store.path(pid)
    by_frame = {f['frame_id']: f for f in frames}
    memory = empty(project['dataset_sha256'], classes, policy)
    versions = project['completed_rounds']
    start = 0
    for position in range(len(versions)-1, -1, -1):
        directory = root / 'rounds' / versions[position]
        receipt = read(directory / 'round.json')
        if receipt.get('memory_sha256'):
            path = directory / 'memory.json'
            if digest(path) != receipt['memory_sha256']:
                raise ValueError('Memory snapshot đã thay đổi')
            memory = validate(read(path), project['dataset_sha256'], classes, by_frame)
            if memory['round_index'] != int(versions[position][1:]):
                raise ValueError('Memory khác vòng snapshot')
            start = position + 1
            break
    sources = [_legacy_source(root, v, by_frame, classes) for v in versions[start:]]
    if not memory['frames'] and sources and sources[0][2]['round_index'] == 1:
        # Use an actual initial consensus only when both training receipts bind it.
        work = sources[0][3]
        path = work / 'initial_shared.json'; cache = work / 'before/predictions.json'
        if path.is_file() and cache.is_file():
            receipts = [work / (k + '_trained.json') for k in ('A', 'B')]
            if all(p.is_file() and read(p).get('shared_labels_sha256') == digest(path) for p in receipts):
                initial = read(path); before = read(cache)
                assert_no_gold(initial); assert_no_gold(before)
                rows = {f['frame_id']: f for f in initial['frames']}
                if len(rows) != len(initial['frames']) or set(rows) != set(by_frame):
                    raise ValueError('Initial pseudo-label memory thiếu frame')
                if set(before['models']) != {'A', 'B'} or any(set(m) != set(by_frame) for m in before['models'].values()):
                    raise ValueError('Initial predict memory thiếu frame')
                sources.insert(0, (rows, before['models'], context(0, before['checkpoint_hashes'],
                    'legacy_initial_pseudo', labels_sha256=digest(path), raw_sha256=digest(cache)), work))
    if sources:
        old = {f['frame_id']: f['objects'] for f in memory['frames']}
        rebuilt = []
        # Read each point cloud once, then replay all saved observations of that frame.
        for fid, frame in by_frame.items():
            objects = old.get(fid, [])
            points = point_loader(frame)
            for rows, raw, ctx, _ in sources:
                row = dict(deepcopy(rows[fid]), boxes=[b for b in rows[fid]['boxes'] if b['class_name'] in classes])
                current = {k: [b for b in model[fid] if b['class_name'] in classes] for k, model in raw.items()}
                objects, _ = update_frame(objects, row, current, ctx, classes, policy, points)
            rebuilt.append(dict(frame_id=fid, objects=objects))
        memory.update(frames=rebuilt, round_index=sources[-1][2]['round_index'],
                      sources=memory['sources'] + [s[2] for s in sources], policy=asdict(policy))
    elif not memory['frames'] and versions:
        raise ValueError('Không có memory/snapshot để khôi phục')
    memory['policy'] = asdict(policy)
    memory['summary'] = summary(memory)
    if memory['frames']:
        validate(memory, project['dataset_sha256'], classes, by_frame)
    return memory
