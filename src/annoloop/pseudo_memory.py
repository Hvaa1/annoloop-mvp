"""Per-frame pseudo-label memory across model rounds; no target gold or GPU."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .predictions import assert_no_gold, validate_predictions
from .reused.geometry import cuboid_iou
from .reused.core import points_in_box

FORMAT = 'annoloop.pseudo_memory.v1'


@dataclass(frozen=True)
class MemoryPolicy:
    enabled: bool = True
    match_iou: float = .3
    stable_iou: float = .5
    min_confidence: float = .7
    min_quality: float = .5
    min_points: int = 5
    stable_hits: int = 2
    keep_misses: int = 1
    expire_misses: int = 3
    max_score_drop: float = .1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('Memory enabled phải bool')
        for name in ('match_iou', 'stable_iou', 'min_confidence', 'min_quality', 'max_score_drop'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Ngưỡng memory sai: ' + name)
        if not 0 < self.match_iou <= self.stable_iou <= 1:
            raise ValueError('Memory cần 0 < match IoU <= stable IoU')
        for name in ('min_points', 'stable_hits', 'keep_misses', 'expire_misses'):
            if type(getattr(self, name)) is not int:
                raise ValueError('Memory counter phải integer')
        if self.min_points < 1 or self.stable_hits < 2 or not 0 <= self.keep_misses < self.expire_misses:
            raise ValueError('Memory cần đủ điểm, >=2 xác nhận và thời hạn giữ hữu hạn')


def empty(dataset_sha256, classes, policy):
    return dict(format=FORMAT, dataset_sha256=dataset_sha256, classes=list(classes),
                policy=asdict(policy), round_index=-1, target_gold_read=False, frames=[], sources=[])


def validate(document, dataset_sha256, classes, frame_ids=None):
    assert_no_gold(document)
    if (document.get('format') != FORMAT or document.get('target_gold_read') is not False
        or document['dataset_sha256'] != dataset_sha256 or document['classes'] != list(classes)):
        raise ValueError('Memory khác dataset/class hoặc chưa xác nhận không dùng gold')
    MemoryPolicy(**document['policy'])
    if type(document.get('round_index')) is not int or document['round_index'] < -1:
        raise ValueError('Memory round index sai')
    ids = [f['frame_id'] for f in document['frames']]
    if len(ids) != len(set(ids)) or (frame_ids is not None and set(ids) != set(frame_ids)):
        raise ValueError('Memory thiếu/trùng frame')
    seen = set()
    for frame in document['frames']:
        for entry in frame['objects']:
            if entry['id'] in seen or entry['class_name'] not in classes:
                raise ValueError('Memory ID trùng hoặc sai class')
            seen.add(entry['id'])
            _quality(entry['box'], classes)
            if entry['box']['class_name'] != entry['class_name'] or entry['state'] not in ('positive', 'ignore', 'expired'):
                raise ValueError('Memory state/class sai')
            for key in ('hit_count', 'stable_hits', 'miss_count', 'conflict_count'):
                if type(entry[key]) is not int or entry[key] < 0:
                    raise ValueError('Memory counter sai')
            rounds = [h['round_index'] for h in entry['history']]
            if not rounds or rounds != sorted(set(rounds)) or rounds[-1] > document['round_index']:
                raise ValueError('Memory history trùng/sai vòng')
            for event in entry['history']:
                for name in ('confidence', 'quality', 'match_iou'):
                    value = event.get(name)
                    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError('Memory history score sai')
    if 'summary' in document and document['summary'] != summary(document):
        raise ValueError('Memory summary khác counters thực')
    return document


def _quality(box, classes):
    validate_predictions({'boxes': [box]}, classes)
    for key in ('confidence', 'quality'):
        value = box.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Memory cần confidence/quality đã calibration')


def _clean(box):
    result = deepcopy(box)
    for key in ('memory', 'memory_id', 'memory_status'):
        result.pop(key, None)
    return result


def _strong(box, policy):
    return box['confidence'] >= policy.min_confidence and box['quality'] >= policy.min_quality


def _near(box, raw, threshold):
    result = []
    for key, boxes in raw.items():
        for b in boxes:
            overlap = cuboid_iou(box['box'], b['box'])
            if overlap >= threshold:
                result.append((key, b, overlap))
    return result


def _annotate(entry):
    box = deepcopy(entry['box'])
    box['native_box_id'] = box.get('native_box_id', box['id'])
    box['id'] = box['memory_id'] = entry['id']
    box['training_policy'] = 'positive' if entry['state'] == 'positive' else 'ignore_region'
    box['reason'] = entry['reason']
    box['point_count'] = entry['point_count']
    measured = [h for h in entry['history'] if h['confidence'] is not None]
    box['memory'] = dict(status=entry['reason'], hit_count=entry['hit_count'],
        stable_hits=entry['stable_hits'], miss_count=entry['miss_count'], conflict_count=entry['conflict_count'],
        first_round=entry['first_round'], last_seen_round=entry['last_seen_round'],
        confidence_mean=sum(h['confidence'] for h in measured) / len(measured),
        quality_mean=sum(h['quality'] for h in measured) / len(measured))
    return box


def update_frame(objects, row, raw, context, classes, policy, points):
    """Hungarian association is within one frame/class, never a temporal track."""
    assert_no_gold(row); assert_no_gold(raw)
    entries = deepcopy(objects)
    candidates = [_clean(b) for b in row['boxes'] if b['class_name'] in classes]
    for b in candidates:
        _quality(b, classes)
    for boxes in raw.values():
        for b in boxes:
            _quality(b, classes)
    overlaps = np.array([[cuboid_iou(e['box']['box'], b['box']) if e['class_name'] == b['class_name'] else 0
                          for b in candidates] for e in entries]).reshape(len(entries), len(candidates))
    matched, used = {}, set()
    if entries and candidates:
        rows, cols = linear_sum_assignment(np.where(overlaps >= policy.match_iou, 1 - overlaps, 1e6))
        for i, j in zip(rows, cols):
            if overlaps[i, j] >= policy.match_iou:
                matched[int(i)] = int(j); used.add(int(j))
    for j, candidate in enumerate(candidates):
        if j in used:
            continue
        identity = hashlib.sha256(json.dumps([row['frame_id'], candidate['class_name'],
            context['round_index'], candidate['id'], candidate['box']], sort_keys=True).encode()).hexdigest()[:24]
        entries.append(dict(id='mem_' + identity, class_name=candidate['class_name'], box=candidate,
            first_round=context['round_index'], last_seen_round=context['round_index'], hit_count=0,
            stable_hits=0, miss_count=0, conflict_count=0, history=[], evidence_ids=[],
            state='ignore', reason='new', point_count=0))
        matched[len(entries)-1] = j
    evidence_id = hashlib.sha256(json.dumps(context['checkpoint_hashes'], sort_keys=True).encode()).hexdigest()
    for i, entry in enumerate(entries):
        candidate = candidates[matched[i]] if i in matched else None
        old = entry['box']
        overlap = cuboid_iou(old['box'], candidate['box']) if candidate is not None else None
        near = _near(candidate or old, raw, policy.match_iou)
        evidence = [dict(detector=k, class_name=b['class_name'], confidence=b['confidence'],
                         quality=b['quality'], match_iou=float(iou)) for k, b, iou in near]
        independent = evidence_id not in entry['evidence_ids']
        count = int(points_in_box(points, (candidate or old)['box']).sum())
        class_conflict = any(b['class_name'] != entry['class_name'] and _strong(b, policy) for _, b, _ in near)
        if candidate is not None:
            class_conflict = class_conflict or any(e is not entry and e['state'] != 'expired'
                and e['class_name'] != entry['class_name'] and _strong(e['box'], policy)
                and cuboid_iou(e['box']['box'], candidate['box']) >= policy.match_iou for e in entries)
        if candidate is not None:
            previous_positive = entry['state'] == 'positive'
            large_drop = (entry['hit_count'] > 0 and
                (candidate['confidence'] < old['confidence'] - policy.max_score_drop or
                 candidate['quality'] < old['quality'] - policy.max_score_drop))
            good = _strong(candidate, policy) and count >= policy.min_points and not class_conflict and not large_drop
            if good:
                if independent:
                    stable = previous_positive and entry['miss_count'] <= policy.keep_misses and overlap >= policy.stable_iou
                    entry['stable_hits'] = entry['stable_hits'] + 1 if stable else 1
                    entry['hit_count'] += 1
                    entry['evidence_ids'].append(evidence_id)
                entry.update(box=candidate, state='positive', reason='memory_observed',
                             miss_count=0, last_seen_round=context['round_index'])
            else:
                entry.update(state='ignore', reason='memory_conflict' if class_conflict or large_drop else 'memory_weak')
                entry['conflict_count'] += 1
                entry['stable_hits'] = 0
                entry['box'] = candidate
        else:
            entry['miss_count'] += 1
            strong_models = {k for k, b, _ in near if b['class_name'] == entry['class_name'] and _strong(b, policy)}
            conflict = class_conflict or (bool(near) and (not strong_models or len(strong_models) >= 2))
            if entry['miss_count'] >= policy.expire_misses:
                entry.update(state='expired', reason='memory_expired', stable_hits=0)
            elif conflict:
                entry.update(state='ignore', reason='memory_conflict', stable_hits=0)
                entry['conflict_count'] += 1
            elif (entry['state'] == 'positive' and entry['stable_hits'] >= policy.stable_hits
                  and entry['miss_count'] <= policy.keep_misses and count >= policy.min_points and _strong(old, policy)):
                entry.update(state='positive', reason='memory_retained_supported' if strong_models else 'memory_retained_missing')
            else:
                entry.update(state='ignore', reason='memory_missing' if not near else 'memory_partial_support')
        entry['point_count'] = count
        entry['history'].append(dict(round_index=context['round_index'], origin=context['origin'],
            checkpoint_hashes=context['checkpoint_hashes'], observed=candidate is not None,
            independent_evidence=independent if candidate is not None else False,
            confidence=candidate['confidence'] if candidate else None,
            quality=candidate['quality'] if candidate else None, match_iou=float(overlap) if overlap is not None else None,
            box=deepcopy(candidate['box']) if candidate else None,
            score=candidate['score'] if candidate else None,
            point_count=count, state=entry['state'], reason=entry['reason'], evidence=evidence))
    for entry in entries:
        if entry['state']=='positive' and entry['reason'].startswith('memory_retained') and any(
            e is not entry and e['state']=='positive' and e['reason']=='memory_observed'
            and e['class_name']==entry['class_name']
            and cuboid_iou(e['box']['box'], entry['box']['box']) >= policy.match_iou for e in entries):
            entry.update(state='ignore', reason='memory_duplicate', stable_hits=0)
            entry['history'][-1].update(state='ignore', reason='memory_duplicate')
    positives = [_annotate(e) for e in entries if e['state'] == 'positive']
    # Signed ignores must not suppress a retained positive for the same class.
    ignored = [deepcopy(b) for b in row.get('ignored', []) if not any(
        b['class_name'] == p['class_name'] and cuboid_iou(b['box'], p['box']) >= policy.match_iou for p in positives)]
    ignored = {b['id']: b for b in ignored}
    ignored.update((e['id'], _annotate(e)) for e in entries if e['state'] == 'ignore')
    result = dict(deepcopy(row), boxes=positives, ignored=list(ignored.values()))
    return entries, result


def summary(document):
    entries = [e for f in document['frames'] for e in f['objects']]
    return dict(objects=len(entries), observed=sum(e['state']=='positive' and e['reason']=='memory_observed' for e in entries),
        retained=sum(e['state']=='positive' and e['reason'].startswith('memory_retained') for e in entries),
        ignored=sum(e['state']=='ignore' for e in entries), expired=sum(e['state']=='expired' for e in entries),
        stable=sum(e['state']=='positive' and e['stable_hits']>=document['policy']['stable_hits'] for e in entries))


def positives(document):
    return {f['frame_id']: [_annotate(e) for e in f['objects'] if e['state']=='positive'] for f in document['frames']}


def update(document, labels, raw, context, point_loader, policy):
    if not policy.enabled:
        return None, deepcopy(labels)
    classes = document['classes']
    assert_no_gold(labels)
    if context['round_index'] != document['round_index'] + 1:
        raise ValueError('Memory chỉ cập nhật theo thứ tự, không ghi hai lần cùng vòng')
    old = {f['frame_id']: f['objects'] for f in document['frames']}
    frame_ids = [f['frame_id'] for f in labels['frames']]
    if len(frame_ids) != len(set(frame_ids)) or (old and set(old) != set(frame_ids)):
        raise ValueError('Memory/nhãn khác tập frame')
    frames, output = [], []
    for row in labels['frames']:
        current = {k: [b for b in model[row['frame_id']] if b['class_name'] in classes] for k, model in raw.items()}
        entries, result = update_frame(old.get(row['frame_id'], []), row, current, context, classes, policy, point_loader(row))
        frames.append(dict(frame_id=row['frame_id'], objects=entries)); output.append(result)
    memory = dict(deepcopy(document), frames=frames, round_index=context['round_index'], policy=asdict(policy),
                  sources=document['sources'] + [deepcopy(context)])
    memory['summary'] = summary(memory)
    validate(memory, document['dataset_sha256'], classes, frame_ids)
    return memory, dict(deepcopy(labels), frames=output, pseudo_memory=dict(format=FORMAT, **memory['summary']))
