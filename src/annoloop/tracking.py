"""Reuse temporal geometry per class, keeping interpolations as ignore/review."""
from copy import deepcopy
from .predictions import validate_predictions
from .reused.tracking import Settings, refine_sequence


def refine(frames, point_loader, classes, mode="pose", fixed_sensor_confirmed=False, settings=None):
    if mode == "fixed_sensor" and not fixed_sensor_confirmed:
        raise ValueError("Cần xác nhận cảm biến cố định")
    for frame in frames:
        validate_predictions(frame, classes)
    result = deepcopy(frames)
    for frame in result:
        frame.update(boxes=[], proposals=[])
    tracks = []
    for class_index, name in enumerate(classes):
        subset = deepcopy(frames)
        for frame in subset:
            # Legacy tracker processes one category; this substitution is internal geometry only.
            frame["boxes"] = [dict(b, class_name="car") for b in frame["boxes"] if b["class_name"] == name]
        refined = refine_sequence(subset, point_loader, settings or Settings(), mode)
        prefix = f"c{class_index}:"
        for original, output in zip(result, refined["frames"]):
            for key in ("boxes", "proposals"):
                for box in output[key]:
                    box["class_name"] = name
                    if box.get("track_id"):
                        box["track_id"] = prefix + box["track_id"]
                    if key == "proposals":
                        box["id"] = prefix + box["id"]
                    original[key].append(box)
        tracks.extend(dict(track, track_id=prefix + track["track_id"], class_name=name)
                      for track in refined["tracks"])
    return dict(frames=result, tracks=tracks, mode=mode, target_gold_read=False,
                interpolations_are_positive=False)
