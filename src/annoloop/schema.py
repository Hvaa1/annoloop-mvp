"""Dataset-declared classes, never a list inferred from unlabelled sensor data."""
from dataclasses import asdict, dataclass
from pathlib import Path
from .io import digest, read


@dataclass(frozen=True)
class ClassSchema:
    data_type: str
    annotation_type: str
    classes: tuple[str, ...]

    @classmethod
    def parse(cls, value):
        if value.get("schema_version", "1.0") != "1.0":
            raise ValueError("Schema version chưa hỗ trợ")
        modality = value.get("data_type")
        kind = value.get("annotation_type")
        if (modality, kind) not in {("3d", "cuboid"), ("2d", "bbox")}:
            raise ValueError("Schema cần data_type 2d/bbox hoặc 3d/cuboid")
        entries = value.get("classes", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError("Dataset cần danh sách class không rỗng")
        names = [entry.get("name") if isinstance(entry, dict) else entry for entry in entries]
        if any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names):
            raise ValueError("Tên class phải là chuỗi không rỗng, không có khoảng trắng ở hai đầu")
        if len(set(names)) != len(names):
            raise ValueError("Class bị trùng")
        return cls(modality, kind, tuple(names))

    def as_dict(self):
        value = asdict(self)
        value.update(schema_version="1.0", classes=list(self.classes))
        return value


def discover(root):
    """Read ONLY class metadata for target. Never read target annotation instances."""
    root = Path(root).resolve()
    for name in ("class_schema.json", "manifest.json", "dataset.json"):
        path = root / name
        if path.is_file():
            document = read(path)
            value = document.get("class_schema", document)
            if "classes" in value:
                return ClassSchema.parse(value), {"path": str(path), "sha256": digest(path),
                                                  "origin": "dataset_schema"}
    raise ValueError("Thiếu schema class: thêm class_schema.json hoặc schema export từ CVAT")


def from_coco_source(path):
    """Explicit SOURCE-only import. Category order follows numeric COCO IDs."""
    document = read(path)
    categories = sorted(document["categories"], key=lambda row: row["id"])
    if len({row["id"] for row in categories}) != len(categories):
        raise ValueError("COCO category id bị trùng")
    return ClassSchema.parse(dict(data_type="2d", annotation_type="bbox",
                                  classes=[row["name"] for row in categories]))
