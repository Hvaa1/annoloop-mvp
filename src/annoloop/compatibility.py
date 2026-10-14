"""Explicit, scoped mappings. Broad classes are never split by renaming."""
from pathlib import Path
from .io import digest, read


def checkpoint_metadata(path):
    metadata = read(path)
    classes = metadata.get("classes")
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
        raise ValueError("Metadata cần class checkpoint theo đúng thứ tự đầu ra")
    if any(not isinstance(c, str) or not c.strip() for c in classes):
        raise ValueError("Tên class checkpoint không hợp lệ")
    checkpoint = Path(metadata["checkpoint"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(path).parent / checkpoint
    if not checkpoint.is_file() or metadata.get("checkpoint_sha256") != digest(checkpoint):
        raise ValueError("Checkpoint thiếu hoặc không khớp hash metadata")
    if metadata.get("classes_verified") is not True:
        raise ValueError("Class checkpoint chưa được xác minh từ config/head")
    return {**metadata, "checkpoint": str(checkpoint.absolute())}


def relation(wanted, available, namespace, rules=()):
    if wanted in available:
        return dict(status="exact", native=wanted, usable=True)
    for rule in rules:
        if (rule.get("namespace") == namespace and rule.get("target") == wanted
            and rule.get("native") in available and rule.get("verified") is True):
            kind = rule.get("relation")
            if kind not in ("exact", "broader", "narrower", "related"):
                raise ValueError("Quan hệ mapping không hợp lệ")
            return dict(status=kind, native=rule["native"], usable=kind == "exact")
    return dict(status="missing", native=None, usable=False)


def matrix(schema, sources, models, rules=()):
    result = []
    for wanted in schema.classes:
        row = dict(class_name=wanted, sources={}, models={})
        for source in sources:
            mapping = relation(wanted, source["classes"], source["namespace"], rules)
            native = mapping["native"]
            mapping.update(train_count=source["train_counts"].get(native, 0),
                           validation_count=source["validation_counts"].get(native, 0))
            mapping["train_ready"] = mapping["usable"] and mapping["train_count"] > 0
            mapping["calibration_ready"] = mapping["usable"] and mapping["validation_count"] > 0
            row["sources"][source["id"]] = mapping
        for model in models:
            mapping = relation(wanted, model["classes"], model["namespace"], rules)
            mapping["usable"] = mapping["usable"] and model.get("classes_verified") is True
            row["models"][model["id"]] = mapping
        result.append(row)
    return result
