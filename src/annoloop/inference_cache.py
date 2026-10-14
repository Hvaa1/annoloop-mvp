"""Copy verified real target outputs into a fresh inference run, never invent boxes."""
from pathlib import Path
import shutil
import numpy as np
from .io import read, write, digest
from .predictions import validate_predictions


def matches(old, spec):
    fields = ('mode', 'checkpoint_sha256', 'config_sha256', 'runtime', 'classes',
              'score_threshold', 'iou_threshold', 'bin_columns', 'coordinates_confirmed', 'require_components')
    return old.get('limit') is None and all(old.get(k) == spec.get(k) for k in fields)


def seed_target(old_job, key, spec, root):
    manifest = {}
    wanted = {f['frame_id']: (i, f) for i, f in enumerate(spec['frames'])}
    paths = sorted((Path(old_job) / key / 'target').glob('*/spec.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        old = read(path)
        if not matches(old, spec):
            continue
        for index, frame in enumerate(old['frames']):
            item = wanted.get(frame['frame_id'])
            if not item or str(item[0]) in manifest or frame != item[1]:
                continue
            source = path.parent / 'frames' / f'{index:06d}.json'
            if not source.is_file():
                continue
            document = read(source)
            if document['frame_id'] != frame['frame_id']:
                raise ValueError('Cached prediction khác frame')
            validate_predictions(document, spec['classes'])
            for box in document['boxes']:
                for field in ('cls_score', 'iou_score'):
                    if not np.isfinite(box[field]) or not 0 <= box[field] <= 1:
                        raise ValueError('Cached prediction thiếu score hợp lệ')
            destination = Path(root) / 'frames' / f'{item[0]:06d}.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            manifest[str(item[0])] = dict(sha256=digest(destination), source=str(source),
                                          frame_id=frame['frame_id'], points_sha256=frame['sha256'])
    return manifest


def cached_frame(root, spec, index):
    item = spec.get('reused_frames', {}).get(str(index))
    if item is None:
        return None
    frame = spec['frames'][index]
    path = Path(root) / 'frames' / f'{index:06d}.json'
    if (item['frame_id'] != frame['frame_id'] or item['points_sha256'] != frame['sha256']
        or digest(path) != item['sha256'] or digest(frame['media']) != frame['sha256']):
        raise ValueError('Cache/frame đã thay đổi trước khi tiếp tục predict')
    document = read(path)
    if document['frame_id'] != frame['frame_id']:
        raise ValueError('Cached prediction khác frame')
    validate_predictions(document, spec['classes'])
    return document


def finish_cached_target(root, spec):
    frames = [dict(frame_id=f['frame_id'], file=f'frames/{i:06d}.json') for i, f in enumerate(spec['frames'])]
    for i in range(len(frames)):
        if cached_frame(root, spec, i) is None:
            raise ValueError('Không được công bố cache thiếu frame là hoàn tất')
    result = dict(mode='target_sample', domain='target_unlabelled', checkpoint=spec['path'],
                  checkpoint_sha256=spec['checkpoint_sha256'], classes=spec['classes'], frames=frames,
                  reused_frames=len(frames), new_forward_frames=0, target_gold_read=False,
                  score_threshold=spec['score_threshold'], iou_threshold=spec['iou_threshold'])
    write(Path(root) / 'result.json', result)
    write(Path(root) / 'inspection.json', dict(status='completed', total_frames=len(frames),
                                             completed_frames=len(frames), reused_frames=len(frames)))
    return result
