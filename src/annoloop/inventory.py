"""Read input media and declared classes; source annotation counting is explicit."""
from collections import Counter
import math
from pathlib import Path
from .io import digest, inside, read
from .schema import discover

EXTENSIONS = {"3d": {".pcd", ".npy", ".bin"}, "2d": {".png", ".jpg", ".jpeg"}}


def inspect_dataset(root, schema_override=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Dataset đầu vào phải là folder")
    if schema_override is None:
        schema, evidence = discover(root)
    else:
        schema, evidence = schema_override, {"origin": "user_selected_classes"}
    manifest = read(root / "manifest.json") if (root / "manifest.json").is_file() else {}
    media_key = "points" if schema.data_type == "3d" else "image"
    rows = manifest.get("frames")
    if rows is None:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS[schema.data_type])
        rows = [{"frame_id": p.relative_to(root).with_suffix("").as_posix(),
                 media_key: p.relative_to(root).as_posix()} for p in paths]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset không có frame")
    frames, seen, sequences = [], set(), {}
    for row in rows:
        fid = row.get("frame_id")
        if not isinstance(fid, str) or not fid or fid in seen:
            raise ValueError("Frame id rỗng hoặc bị trùng")
        seen.add(fid)
        path = inside(root, row[media_key])
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS[schema.data_type]:
            raise ValueError("Media không tồn tại hoặc không đúng modality")
        split = row.get("split", "adaptation")
        if split not in ("adaptation", "holdout", "delivery"):
            raise ValueError("Target split phải là adaptation/holdout/delivery")
        sequence = row.get("sequence_id")
        if sequence is not None:
            if not isinstance(sequence, str) or not sequence:
                raise ValueError("Sequence id không hợp lệ")
            if sequence in sequences and sequences[sequence] != split:
                raise ValueError("Cùng sequence không được chia adaptation và holdout")
            sequences[sequence] = split
        timestamp = row.get("timestamp")
        if timestamp is not None and (type(timestamp) not in (int, float) or not math.isfinite(timestamp)):
            raise ValueError("Timestamp phải hữu hạn")
        difficulty = row.get("difficulty", "unknown")
        if difficulty not in ("easy", "hard", "unknown"):
            raise ValueError("Difficulty phải là easy/hard/unknown")
        frames.append(dict(frame_id=fid, media=str(path), split=split, sequence_id=sequence,
                           timestamp=timestamp, difficulty=difficulty, sha256=digest(path)))
        frames[-1].update({key: row[key] for key in ("lidar_to_world", "pose_source", "pose_quality_passed") if key in row})
    return dict(root=str(root), schema=schema.as_dict(), class_evidence=evidence,
                frames=frames, target_gold_read=False)


def source_counts(annotation_path, schema):
    """Canonical labelled source JSON: split='train'|'validation', frames[].boxes[].class_name.

    Also supports COCO source JSON. Never called during target inspection.
    Counts are box instances, not number of distinct objects.
    """
    path = Path(annotation_path)
    doc = read(path)
    if doc.get("role") != "source" or doc.get("split") not in ("train", "validation"):
        raise ValueError("Đếm nhãn chỉ nhận annotation role=source, split=train/validation")
    if "categories" in doc:
        mapping = {c["id"]: c["name"] for c in doc["categories"]}
        names = [mapping[b["category_id"]] for b in doc.get("annotations", []) if not b.get("ignore", False)]
    else:
        names = [b["class_name"] for f in doc.get("frames", []) for b in f.get("boxes", [])
                 if not b.get("ignore", False)]
    unknown = set(names) - set(schema.classes)
    if unknown:
        raise ValueError("Annotation có class ngoài schema: " + ", ".join(sorted(unknown)))
    counts = Counter(names)
    return dict(split=doc["split"], counts={c: counts[c] for c in schema.classes},
                annotation_sha256=digest(path), path=str(path.absolute()))


def fixed_samples(inventory, count=8):
    """Stable, stratified samples from adaptation; held-out gold is never used."""
    if type(count) is not int or count < 1:
        raise ValueError("Sample count phải > 0")
    groups = {}
    for frame in inventory["frames"]:
        if frame["split"] == "adaptation":
            key = (frame["difficulty"], frame.get("sequence_id") or "")
            groups.setdefault(key, []).append(frame["frame_id"])
    groups = [sorted(group) for _, group in sorted(groups.items())]
    selected = []
    while groups and len(selected) < count:
        remaining = []
        for group in groups:
            if len(selected) == count:
                break
            selected.append(group.pop(0))
            if group:
                remaining.append(group)
        groups = remaining
    return selected
