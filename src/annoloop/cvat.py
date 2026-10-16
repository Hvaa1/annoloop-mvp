"""Pure shape/track packages for local CVAT. No credentials persisted."""
from copy import deepcopy
from pathlib import Path
import zipfile
import numpy as np
from .io import digest, write
from .predictions import validate_predictions
from .schema import ClassSchema

ATTRIBUTES = {"origin": "detector", "training_policy": "positive", "review_status": "unreviewed"}


def check_login(host, username, password):
    """Explicit authentication + users/self only; no task/export required.

    Return no password, cookie or token, and do not expose SDK response bodies
    in error messages. The caller may keep the non-secret receipt in UI memory.
    """
    from urllib.parse import urlparse
    from .io import now
    host=host.strip().rstrip('/');username=username.strip()
    parsed=urlparse(host)
    if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('Địa chỉ CVAT cần http/https, tài khoản và mật khẩu nhập riêng')
    if not username or not password:
        raise ValueError('Nhập tài khoản và mật khẩu CVAT trước khi kiểm tra')
    from cvat_sdk import make_client
    try:
        with make_client(host,credentials=(username,password)) as client:
            account=client.users.retrieve_current_user()
            result=dict(host=host,username=account.username,user_id=account.id,checked_at=now())
    except Exception as error:
        status=getattr(error,'status',None)
        if status in (400,401,403):
            raise ValueError(f'CVAT từ chối đăng nhập (HTTP {status}). Kiểm tra tài khoản, mật khẩu và địa chỉ server.') from None
        if status:
            raise ValueError(f'Không xác minh được tài khoản CVAT (HTTP {status}).') from None
        raise ValueError('Không kiểm tra được kết nối CVAT ('+type(error).__name__+').') from None
    return result


def label_specs(schema):
    return [dict(name=name, type="cuboid" if schema.data_type == "3d" else "rectangle",
                 attributes=[dict(name=key, input_type="text", mutable=True, default_value=value, values=[])
                             for key, value in ATTRIBUTES.items()]
                 + [dict(name="track_id", input_type="text", mutable=False, default_value="unassigned", values=[])])
            for name in schema.classes]


def cuboid_points(box):
    # CVAT THREE Euler XYZ, yaw around Z; same convention as reviewed demo exporter.
    return [*box[:3], 0., 0., box[6], *box[3:6], *([0.] * 7)]


def package(frames, schema, kind="shape", include_review=False):
    if kind not in ("shape", "track"):
        raise ValueError("Export kind phải shape/track")
    shapes, tracks = [], {}
    for index, frame in enumerate(frames):
        validate_predictions(frame, schema.classes, schema.data_type)
        for box in frame["boxes"] + (frame.get("proposals", []) if include_review else []):
            name = box["class_name"]
            if name not in schema.classes:
                raise ValueError("Class export ngoài schema")
            shape = dict(frame=index, type="cuboid" if schema.data_type == "3d" else "rectangle",
                         points=cuboid_points(box["box"]) if schema.data_type == "3d" else box["box"],
                         outside=False, occluded=False, z_order=0, rotation=0.,
                         class_name=name, attribute_values={key: str(box.get(key, value)) for key, value in ATTRIBUTES.items()})
            if kind == "shape":
                shapes.append(shape)
            else:
                tid = f'{frame.get("sequence_id", "")}|{name}|{box.get("track_id") or "singleton:" + str(index) + ":" + box["id"]}'
                track = tracks.setdefault(tid, dict(class_name=name, visible={}))
                if index in track["visible"]:
                    raise ValueError("Hai box trong cùng track/frame")
                track["visible"][index] = shape
    output_tracks = []
    for tid, track in sorted(tracks.items()):
        visible = track["visible"]
        first, last = min(visible), max(visible)
        values, previous = [], visible[first]
        for index in range(first, min(last + 2, len(frames))):
            if index in visible:
                previous = visible[index]
                values.append(deepcopy(previous))
            else:
                values.append(dict(deepcopy(previous), frame=index, outside=True))
        output_tracks.append(dict(track_id=tid, class_name=track["class_name"], frame=first,
                                  source="auto", attributes=[], shapes=values))
    return dict(annotation_kind=kind, schema=schema.as_dict(), shapes=shapes, tracks=output_tracks,
                frame_ids=[f["frame_id"] for f in frames])


def payload(document, labels):
    expected = ClassSchema.parse(document["schema"])
    mapping = {label["name"]: label["id"] for label in labels}
    specs = {label["name"]: {a["name"]: a["id"] for a in label.get("attributes", [])} for label in labels}
    if not set(expected.classes).issubset(mapping):
        raise ValueError("CVAT thiếu class dataset")
    output = dict(version=0, tags=[], shapes=[], tracks=[])
    def shape(raw):
        result = deepcopy(raw)
        name = result.pop("class_name")
        result["label_id"] = mapping[name]
        result["attributes"] = [dict(spec_id=specs[name][key], value=value)
                                for key, value in result.pop("attribute_values").items()]
        return result
    output["shapes"] = [shape(s) for s in document["shapes"]]
    for raw in document["tracks"]:
        track = deepcopy(raw)
        tid = track.pop("track_id")
        name = track.pop("class_name")
        track["label_id"] = mapping[name]
        track["attributes"] = [dict(spec_id=specs[name]["track_id"], value=tid)]
        track["shapes"] = []
        for item in raw["shapes"]:
            result = shape(item)
            result.pop("label_id")
            track["shapes"].append(result)
        output["tracks"].append(track)
    if document["annotation_kind"] == "shape" and output["tracks"] or document["annotation_kind"] == "track" and output["shapes"]:
        raise ValueError("Không trộn shape và track")
    return output


def binary_pcd(points):
    points = np.asarray(points, dtype="<f4")
    if points.ndim != 2 or points.shape[1] < 3 or not len(points) or not np.isfinite(points[:, :3]).all():
        raise ValueError("PCD cần XYZ hữu hạn")
    points = np.ascontiguousarray(points[:, :3])
    header = f"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n"
    return header.encode("ascii") + points.tobytes()


def export_media(inventory, frame_ids, output, point_loader=None):
    """Media order has indexed filenames; no guessed channels or units."""
    mapping = {f["frame_id"]: f for f in inventory["frames"]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    modality = inventory["schema"]["data_type"]
    if modality == "3d" and point_loader is None:
        raise ValueError("3D export cần point_loader đã chuẩn hóa XYZ theo đơn vị/trục đã xác minh")
    names = []
    with zipfile.ZipFile(output / "media.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for index, fid in enumerate(frame_ids):
            frame = mapping[fid]
            source = Path(frame["media"])
            if digest(source) != frame["sha256"]:
                raise ValueError("Media đã thay đổi")
            name = f"{index:08d}" + (".pcd" if modality == "3d" else source.suffix.lower())
            if modality == "3d":
                archive.writestr(name, binary_pcd(point_loader(frame)))
            else:
                archive.write(source, name)
            names.append(name)
    write(output / "media.json", dict(frame_ids=frame_ids, frame_names=names,
                                      media_sha256=digest(output / "media.zip")))
    return names


def verify_readback(actual, expected):
    def attributes(value):
        return {a["spec_id"]: a["value"] for a in value.get("attributes", [])}
    def check_shape(a, b):
        if any(a.get(k) != b.get(k) for k in ("frame", "type", "outside")):
            raise ValueError("CVAT read-back sai frame/type/outside")
        if len(a["points"]) != len(b["points"]) or not np.allclose(a["points"], b["points"], atol=1e-5, rtol=0):
            raise ValueError("CVAT read-back sai geometry")
        if any(attributes(a).get(k) != v for k, v in attributes(b).items()):
            raise ValueError("CVAT read-back sai provenance")
    if actual.get("tags") or any(len(actual[k]) != len(expected[k]) for k in ("shapes", "tracks")):
        raise ValueError("CVAT read-back sai số lượng")
    order = lambda x: (x["frame"], x.get("label_id", -1), x.get("outside", False), tuple(x["points"]))
    for a, b in zip(sorted(actual["shapes"], key=order), sorted(expected["shapes"], key=order)):
        if a["label_id"] != b["label_id"]:
            raise ValueError("CVAT read-back sai class")
        check_shape(a, b)
    tracks = {tuple(sorted(attributes(t).items())): t for t in actual["tracks"]}
    if len(tracks) != len(actual["tracks"]):
        raise ValueError("CVAT track id trùng")
    for b in expected["tracks"]:
        a = tracks.get(tuple(sorted(attributes(b).items())))
        if a is None or a["label_id"] != b["label_id"] or len(a["shapes"]) != len(b["shapes"]):
            raise ValueError("CVAT read-back sai track")
        for sa, sb in zip(sorted(a["shapes"], key=order), sorted(b["shapes"], key=order)):
            check_shape(sa, sb)


def upload(directory, document, host, username, password, name):
    """Create a new task; verify exact media order and annotation read-back.

    Explicit invocation only. Never overwrite a task containing unrelated human review.
    """
    from urllib.parse import urlparse
    from cvat_sdk import make_client
    from cvat_sdk.core.proxies.tasks import ResourceType
    from .io import lock, read
    parsed = urlparse(host)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        raise ValueError("Host CVAT phải http/https, credentials truyền riêng")
    directory = Path(directory)
    with lock(directory / ".upload.lock", blocking=False):
        media = read(directory / "media.json")
        if digest(directory / "media.zip") != media["media_sha256"] or media["frame_ids"] != document["frame_ids"]:
            raise ValueError("Media không khớp annotation")
        signature = directory / "annotations.json"
        if signature.exists():
            if read(signature) != document:
                raise ValueError("Annotation export đã thay đổi; tạo export mới")
        else:
            write(signature, document)
        receipt_path = directory / "upload.json"
        receipt = read(receipt_path) if receipt_path.exists() else dict(host=host.rstrip("/"), status="new")
        if receipt["host"] != host.rstrip("/"):
            raise ValueError("Export đã gắn CVAT host khác")
        with make_client(host, credentials=(username, password)) as client:
            if receipt.get("status") == "verified":
                return receipt  # Preserve human edits after verified upload.
            if "task_id" not in receipt:
                task = client.tasks.create(dict(name=name, labels=label_specs(ClassSchema.parse(document["schema"]))))
                receipt.update(task_id=task.id, status="task_created")
                write(receipt_path, receipt)
            else:
                task = client.tasks.retrieve(receipt["task_id"])
            if receipt["status"] == "task_created":
                receipt["status"] = "media_upload_started"
                write(receipt_path, receipt)
                task.upload_data([directory / "media.zip"], resource_type=ResourceType.LOCAL,
                                 params=dict(image_quality=70, sorting_method="lexicographical"))
            frames = task.get_frames_info()
            if [Path(f.name).name for f in frames] != media["frame_names"]:
                raise ValueError("Thứ tự media CVAT khác export hoặc media chưa sẵn sàng")
            expected = payload(document, [label.to_dict() for label in task.get_labels()])
            actual = task.get_annotations().to_dict()
            if actual["shapes"] or actual["tracks"] or actual["tags"]:
                if receipt["status"] != "annotations_write_started":
                    raise ValueError("Task đã có nhãn khác; không ghi đè")
                verify_readback(actual, expected)
            else:
                receipt["status"] = "annotations_write_started"
                write(receipt_path, receipt)
                expected["version"] = actual["version"]
                task.set_annotations(expected)
                verify_readback(task.get_annotations().to_dict(), expected)
            receipt.update(status="verified", url=f'{host.rstrip("/")}/tasks/{task.id}')
            write(receipt_path, receipt)
            return receipt
