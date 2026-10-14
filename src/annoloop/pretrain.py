"""Materialize a source-only 3D training config with explicit class/anchor checks."""
from copy import deepcopy
import math
from pathlib import Path
import yaml


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError("Config inheritance có vòng lặp")
    seen.add(path)
    document = yaml.safe_load(path.read_text())
    def materialize(node):
        if isinstance(node, list):
            return [materialize(value) for value in node]
        if not isinstance(node, dict):
            return node
        base = node.get("_BASE_CONFIG_")
        values = {key: materialize(value) for key, value in node.items() if key != "_BASE_CONFIG_"}
        if base:
            # Paths are relative to this config, never the shell's working directory.
            original = load_config(path.parent / base, seen)
            return merge(original, values)
        return values
    return materialize(document)


def merge(first, second):
    result = deepcopy(first)
    for key, value in second.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def source_config(template, schema, source, epochs, ros=(.9, 1.1), anchor_overrides=None):
    if schema.data_type != "3d":
        raise ValueError("Trainer nguồn ST3D hiện chỉ hỗ trợ 3D; adapter 2D chưa triển khai")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Epoch phải >0")
    if not Path(source).is_dir():
        raise ValueError("Folder source không tồn tại")
    config = load_config(template)
    if config["MODEL"]["NAME"] not in ("SECONDNetIoU", "PVRCNN"):
        raise ValueError("Profile trainer chỉ hỗ trợ SECOND-IoU/PV-RCNN")
    anchors = config["MODEL"]["DENSE_HEAD"].get("ANCHOR_GENERATOR_CONFIG", [])
    by_class = {a["class_name"]: a for a in anchors}
    if anchor_overrides:
        by_class.update({a["class_name"]: deepcopy(a) for a in anchor_overrides})
    if any(c not in by_class for c in schema.classes):
        raise ValueError("Template thiếu anchor cho class cần học; cần profile multiclass tương ứng")
    config["CLASS_NAMES"] = list(schema.classes)
    config["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"] = [by_class[c] for c in schema.classes]
    config.pop("SELF_TRAIN", None)
    config.pop("DATA_CONFIG_TAR", None)
    data = config["DATA_CONFIG"]
    data["DATA_PATH"] = str(Path(source).resolve())
    augmentor = data.setdefault("DATA_AUGMENTOR", {})
    aug = [a for a in augmentor.get("AUG_CONFIG_LIST", []) if a["NAME"] not in ("random_object_scaling", "normalize_object_size", "gt_sampling")]
    # Disable inherited fixed object normalization; GT sampler configs often contain old class groups.
    augmentor["DISABLE_AUG_LIST"] = [name for name in augmentor.get("DISABLE_AUG_LIST", [])
                                     if name not in ("random_object_scaling", "normalize_object_size", "gt_sampling")]
    if ros is not None:
        if len(ros) != 2 or any(not math.isfinite(x) or x <= 0 for x in ros) or ros[0] > ros[1]:
            raise ValueError("Khoảng ROS không hợp lệ")
        aug.insert(0, dict(NAME="random_object_scaling", SCALE_UNIFORM_NOISE=list(ros)))
    augmentor["AUG_CONFIG_LIST"] = aug
    config["OPTIMIZATION"].update(NUM_EPOCHS=epochs, BATCH_SIZE_PER_GPU=1)
    return config
