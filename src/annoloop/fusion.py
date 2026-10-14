"""Calibrate on source, compare versions and fuse detectors class by class."""
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from .predictions import validate_predictions
from .reused.calibration import isotonic
from .reused.geometry import cuboid_iou


MIN_CALIBRATION_SAMPLES = 10


class SourceCalibrationError(ValueError):
    def __init__(self, report):
        self.report = report
        missing = [f"{row['class_name']}: {row['samples']}/{report['minimum_samples']}"
                   for row in report['classes'] if not row['enough_samples']]
        super().__init__('Chưa đủ dự đoán source validation để calibration class ' + '; '.join(missing)
                         + '. Đây là số box model dự đoán, không phải số nhãn nguồn. '
                           'Cần checkpoint tốt hơn hoặc thêm source validation độc lập; '
                           'không dùng gold target và không tự hạ số mẫu tối thiểu.')


def source_calibration_report(records, classes):
    if records.get("domain") != "source_validation":
        raise ValueError("Calibration cần source validation có nhãn")
    rows = []
    for name in classes:
        boxes = [b for b in records["boxes"] if b["class_name"] == name]
        for box in boxes:
            for key in ("cls_score", "iou_score", "source_gt_iou"):
                value = box[key]
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Source calibration score sai")
        rows.append(dict(class_name=name, samples=len(boxes),
                         matched_at_iou_0_3=int(sum(b['source_gt_iou'] >= .3 for b in boxes)),
                         enough_samples=len(boxes) >= MIN_CALIBRATION_SAMPLES))
    return dict(domain='source_validation', minimum_samples=MIN_CALIBRATION_SAMPLES,
                classes=rows, ready=all(row['enough_samples'] for row in rows))


def calibrate_source(records, checkpoint_sha256, classes):
    report = source_calibration_report(records, classes)
    if not report['ready']:
        raise SourceCalibrationError(report)
    curves = {}
    for name in classes:
        boxes = [b for b in records["boxes"] if b["class_name"] == name]
        curves[name] = dict(confidence=isotonic([b["cls_score"] for b in boxes],
                              [float(b["source_gt_iou"] >= .3) for b in boxes]),
                            quality=isotonic([b["iou_score"] for b in boxes],
                              [b["source_gt_iou"] for b in boxes]))
    return dict(domain="source_validation", checkpoint_sha256=checkpoint_sha256, curves=curves)


def normalize(document, calibration, checkpoint_sha256, classes, adaptation_classes=None):
    if calibration.get("domain") != "source_validation" or calibration["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Calibration không khớp checkpoint/source")
    validate_predictions(document, classes)
    result = deepcopy(document["boxes"])
    active = set(classes if adaptation_classes is None else adaptation_classes)
    for box in result:
        if box['class_name'] not in active:
            box.pop('confidence', None)
            box.pop('quality', None)
            box.update(training_policy='ignore_region', reason='excluded_adaptation_class')
            continue
        curve = calibration["curves"][box["class_name"]]
        for output, key in (("confidence", "cls_score"), ("quality", "iou_score")):
            value = box[key]
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Cần cls_score và iou_score riêng, hợp lệ")
            box[output] = float(np.interp(value, curve[output]["x"], curve[output]["y"]))
    return result


def dominates(first, second):
    return first["confidence"] >= second["confidence"] and first["quality"] >= second["quality"]


def compare(first, second, min_iou=.3, min_confidence=.7, min_quality=.5):
    """Accept a dominating matched box; disagreement/unmatched regions remain review."""
    if not 0 < min_iou <= 1:
        raise ValueError("IoU threshold không hợp lệ")
    for box in first + second:
        validate_predictions({"boxes": [box]})
        for key in ("confidence", "quality"):
            if key not in box or not np.isfinite(box[key]) or not 0 <= box[key] <= 1:
                raise ValueError("Cần confidence/quality đã calibration")
    positive, review = [], []
    if not first or not second:
        return [], [dict(b, training_policy="ignore_region", reason="unmatched") for b in first + second]
    ious = np.array([[cuboid_iou(a["box"], b["box"]) if a["class_name"] == b["class_name"] else 0
                      for b in second] for a in first])
    rows, cols = linear_sum_assignment(np.where(ious >= min_iou, 1 - ious, 1e6))
    used_a, used_b = set(), set()
    for i, j in zip(rows, cols):
        if ious[i, j] < min_iou:
            continue
        used_a.add(i); used_b.add(j)
        a, b = first[i], second[j]
        winner = b if dominates(b, a) else a if dominates(a, b) else None
        if winner is not None and (winner["confidence"] < min_confidence or winner["quality"] < min_quality):
            review.extend(dict(box, training_policy="ignore_region", reason="low_calibrated_quality") for box in (a, b))
            continue
        if winner:
            positive.append(dict(winner, match_iou=float(ious[i, j]),
                                 training_policy="positive", matched_ids=[a["id"], b["id"]]))
        else:
            review.extend(dict(box, training_policy="ignore_region", reason="quality_conflict") for box in (a, b))
    review.extend(dict(b, training_policy="ignore_region", reason="unmatched") for i, b in enumerate(first) if i not in used_a)
    review.extend(dict(b, training_policy="ignore_region", reason="unmatched") for i, b in enumerate(second) if i not in used_b)
    return positive, review


def consensus_round(previous_a, current_a, previous_b, current_b, min_iou=.3):
    """M(n-1) vs Mn independently, then A vs B. Calibrated scores required."""
    a, review_a = compare(previous_a, current_a, min_iou) if previous_a is not None else (current_a, [])
    b, review_b = compare(previous_b, current_b, min_iou) if previous_b is not None else (current_b, [])
    positive, review = compare(a, b, min_iou)
    return dict(boxes=positive, ignored=review_a + review_b + review, target_gold_read=False)
