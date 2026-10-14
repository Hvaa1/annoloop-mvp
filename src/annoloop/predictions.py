"""Version-independent prediction contract for detector adapters and exporters."""
import math


def assert_no_gold(value):
    if isinstance(value, dict):
        if any(key in value for key in ("gold", "source_gt_iou", "target_gt_iou", "ground_truth")):
            raise ValueError("Target artifact không được chứa gold")
        if value.get("target_gold_read", False) is not False:
            raise ValueError("Target artifact đã dùng gold")
        for item in value.values():
            assert_no_gold(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_gold(item)


def validate_predictions(document, classes=None, modality="3d"):
    assert_no_gold(document)
    boxes = document.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("Prediction cần boxes[]")
    seen = set()
    for box in boxes:
        identity = box.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Box ID rỗng/trùng")
        seen.add(identity)
        name = box.get("class_name")
        if not isinstance(name, str) or not name or (classes is not None and name not in classes):
            raise ValueError("Prediction class nằm ngoài schema")
        geometry = box.get("box", [])
        length = 7 if modality == "3d" else 4
        if len(geometry) != length or any(type(x) not in (int, float) or not math.isfinite(x) for x in geometry):
            raise ValueError("Hình học box không hợp lệ")
        if modality == "3d" and any(x <= 0 for x in geometry[3:6]):
            raise ValueError("Kích thước cuboid phải > 0")
        if modality == "2d" and (geometry[2] <= geometry[0] or geometry[3] <= geometry[1]):
            raise ValueError("BBox cần x1 < x2, y1 < y2")
        score = box.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Score phải trong [0,1]")
        if "source_gt_iou" in box or "gold" in box:
            raise ValueError("Target prediction không được chứa gold")
    return boxes
