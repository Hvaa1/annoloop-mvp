"""Class-aware fusion of calibrated predictions; checkpoint weights stay separate."""
from copy import deepcopy
from dataclasses import dataclass
import math
from .predictions import validate_predictions
from .reused.geometry import cuboid_iou
from .reused.core import points_in_box


@dataclass(frozen=True)
class EnsemblePolicy:
    match_iou: float = .5
    confidence: float = .7
    quality: float = .5
    singleton_confidence: float = .9
    singleton_quality: float = .6
    vote_fraction: float = .5
    min_points: int = 5

    def __post_init__(self):
        for field in ('match_iou', 'confidence', 'quality', 'singleton_confidence',
                      'singleton_quality', 'vote_fraction'):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('Ngưỡng ensemble không hợp lệ: ' + field)
        if type(self.min_points) is not int or self.min_points < 1:
            raise ValueError('Cần ít nhất một point hỗ trợ')
        if self.singleton_confidence < self.confidence or self.singleton_quality < self.quality:
            raise ValueError('Model đơn lẻ cần ngưỡng ít nhất bằng ngưỡng chung')


def fuse_frame(documents, providers, classes, points, policy=None):
    """One vote per lineage/group, with complete coverage and no synthetic boxes."""
    policy = policy or EnsemblePolicy()
    if not providers or set(documents) != set(providers):
        raise ValueError('Thiếu kết quả model; không coi frame chưa predict là frame rỗng')
    hashes = [p['checkpoint_sha256'] for p in providers.values()]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Checkpoint trùng hash không phải model độc lập')
    entries = []
    for key, doc in documents.items():
        validate_predictions(doc, providers[key]['classes'])
        for raw in doc['boxes']:
            if raw['class_name'] not in classes:
                continue
            for score in ('confidence', 'quality'):
                if score not in raw or not math.isfinite(raw[score]) or not 0 <= raw[score] <= 1:
                    raise ValueError('Cần confidence và quality đã calibration theo checkpoint/class')
            entries.append(dict(box=deepcopy(raw), provider=key,
                                group=providers[key]['vote_group']))
    # Seed matching avoids transitive A~B~C chains and duplicate votes in one model.
    entries.sort(key=lambda e: (-e['box']['confidence'], -e['box']['quality'],
                                e['provider'], str(e['box']['id'])))
    clusters = []
    for entry in entries:
        for cluster in clusters:
            seed = cluster[0]['box']
            if seed['class_name'] == entry['box']['class_name'] and \
               cuboid_iou(seed['box'], entry['box']['box']) >= policy.match_iou:
                cluster.append(entry)
                break
        else:
            clusters.append([entry])
    positive, ignored = [], []
    for index, cluster in enumerate(clusters):
        name = cluster[0]['box']['class_name']
        available = {p['vote_group'] for p in providers.values() if name in p['classes']}
        single = len(available) == 1
        conf = policy.singleton_confidence if single else policy.confidence
        quality = policy.singleton_quality if single else policy.quality
        required = 1 if single else max(2, math.ceil(len(available) * policy.vote_fraction))
        evidence = []
        for entry in cluster:
            box = entry['box']
            support = int(points_in_box(points, box['box']).sum())
            evidence.append(dict(provider=entry['provider'], vote_group=entry['group'],
                                 checkpoint_sha256=providers[entry['provider']]['checkpoint_sha256'],
                                 box_id=box['id'], confidence=box['confidence'], quality=box['quality'],
                                 point_count=support))
        good = [i for i, row in enumerate(evidence) if row['confidence'] >= conf and
                row['quality'] >= quality and row['point_count'] >= policy.min_points]
        votes = {evidence[i]['vote_group'] for i in good}
        winner = max(good or range(len(cluster)),
                     key=lambda i: (cluster[i]['box']['quality'], cluster[i]['box']['confidence']))
        box = dict(cluster[winner]['box'], id='ensemble-' + str(index), origin='local_ensemble',
                   evidence=evidence, independent_votes=len(votes), available_groups=len(available),
                   required_votes=required, point_count=evidence[winner]['point_count'])
        if len(votes) >= required:
            box.update(training_policy='positive', reason='single_group_strict' if single else 'independent_group_agreement')
            positive.append(box)
        else:
            box.update(training_policy='ignore_region', reason='insufficient_calibrated_evidence')
            ignored.append(box)
    # Cross-class conflicts remain visible, never silently relabelled.
    conflict = set()
    for i, first in enumerate(positive):
        for j in range(i + 1, len(positive)):
            second = positive[j]
            if first['class_name'] != second['class_name'] and \
               cuboid_iou(first['box'], second['box']) >= policy.match_iou:
                conflict.update((i, j))
    ignored.extend(dict(b, training_policy='ignore_region', reason='cross_class_conflict')
                   for i, b in enumerate(positive) if i in conflict)
    result = dict(boxes=[b for i, b in enumerate(positive) if i not in conflict],
                  ignored=ignored, target_gold_read=False)
    validate_predictions(result, classes)
    return result
