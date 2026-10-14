"""Persistent projects, fixed samples and immutable label snapshots."""
from dataclasses import asdict
from pathlib import Path
import re
import shutil
import uuid
from .inventory import fixed_samples, inspect_dataset
from .io import digest, inside, lock, now, read, write
from .predictions import assert_no_gold, validate_predictions
from .schema import ClassSchema
from .selection import SelectionPolicy, decide, validate_metrics


class ProjectStore:
    def __init__(self, data):
        self.root = Path(data) / "projects"

    def path(self, project_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", project_id):
            raise ValueError("Project ID không hợp lệ")
        return self.root / project_id

    def create(self, name, dataset, max_rounds=3, execution_mode="manual", policy=None, samples=8, selected_classes=None, schema_override=None):
        if not name.strip() or type(max_rounds) is not int or max_rounds < 0:
            raise ValueError("Tên project/số vòng không hợp lệ")
        if execution_mode not in ("manual", "automatic"):
            raise ValueError("Execution mode không hợp lệ")
        inventory = inspect_dataset(dataset, schema_override)
        if selected_classes is not None:
            available = inventory["schema"]["classes"]
            if (not selected_classes or len(set(selected_classes)) != len(selected_classes)
                    or any(c not in available for c in selected_classes)):
                raise ValueError("Chọn ít nhất một class có trong schema dataset")
            inventory["schema"]["classes"] = [c for c in available if c in selected_classes]
        if not any(f["split"] == "adaptation" for f in inventory["frames"]):
            raise ValueError("Project cần ít nhất một frame adaptation")
        policy = policy or SelectionPolicy()
        project_id = uuid.uuid4().hex[:16]
        path = self.path(project_id)
        path.mkdir(parents=True)
        if schema_override is not None:
            schema_path = path / "generated" / "target" / "class_schema.json"
            write(schema_path, inventory["schema"])
            inventory["class_evidence"] = dict(origin="user_selected_classes", path=str(schema_path), sha256=digest(schema_path))
        write(path / "dataset.json", inventory)
        write(path / "project.json", dict(id=project_id, name=name, created_at=now(),
              schema=inventory["schema"], dataset_sha256=digest(path / "dataset.json"),
              sample_frame_ids=fixed_samples(inventory, samples), max_rounds=max_rounds,
              execution_mode=execution_mode, selection_policy=asdict(policy),
              completed_rounds=[], selected_version=None, reviewed_round=None, stop_after_round=False,
              target_gold_read=False))
        return project_id

    def get(self, project_id):
        return read(self.path(project_id) / "project.json")

    def list(self):
        return [read(p) for p in sorted(self.root.glob("*/project.json"))]

    def configure(self, project_id, name, max_rounds, execution_mode, policy=None):
        """Change run budget/settings without touching the dataset or snapshots."""
        if not name.strip() or type(max_rounds) is not int or max_rounds < 0:
            raise ValueError('Tên dự án/số vòng không hợp lệ')
        if execution_mode not in ('manual', 'automatic'):
            raise ValueError('Chế độ chạy không hợp lệ')
        root = self.path(project_id)
        from .adaptation import active_job
        with lock(root / '.adaptation_dispatch.lock'):
            if active_job(self, project_id):
                raise ValueError('Chờ job hiện tại hoàn tất trước khi sửa thiết lập dự án')
            with lock(root / '.lock'):
                project = self.get(project_id)
                if max_rounds < len(project['completed_rounds']):
                    raise ValueError('Số vòng không được ít hơn số vòng đã hoàn tất')
                project.update(name=name.strip(), max_rounds=max_rounds, execution_mode=execution_mode)
                if policy is not None:
                    project['selection_policy'] = asdict(policy)
                write(root / 'project.json', project)
        return project

    def verify_dataset(self, project_id):
        root = self.path(project_id)
        project = self.get(project_id)
        if digest(root / "dataset.json") != project["dataset_sha256"]:
            raise ValueError("Dataset snapshot bị thay đổi")
        inventory = read(root / "dataset.json")
        if digest(inventory["class_evidence"]["path"]) != inventory["class_evidence"]["sha256"]:
            raise ValueError("Schema/manifest đầu vào đã thay đổi")
        for frame in inventory["frames"]:
            if digest(frame["media"]) != frame["sha256"]:
                raise ValueError("Media đầu vào đã thay đổi: " + frame["frame_id"])
        return inventory

    def record_round(self, project_id, index, predictions, metrics, checkpoints, sample_predictions, work_directory=None,
                     memory=None):
        root = self.path(project_id)
        with lock(root / ".lock"):
            project = self.get(project_id)
            if type(index) is not int or not 1 <= index <= project["max_rounds"]:
                raise ValueError("Vòng ngoài ngân sách")
            if index != len(project["completed_rounds"]) + 1:
                raise ValueError("Ghi vòng theo thứ tự, không ghi đè")
            inventory = self.verify_dataset(project_id)
            schema = ClassSchema.parse(project["schema"])
            expected = {f["frame_id"] for f in inventory["frames"] if f["split"] == "adaptation"}
            ids = [f["frame_id"] for f in predictions["frames"]]
            if set(ids) != expected or len(ids) != len(expected) or predictions.get("target_gold_read") is not False:
                raise ValueError("Snapshot thiếu/trùng frame hoặc dùng holdout/gold")
            validate_metrics(metrics)
            assert_no_gold(predictions)
            count = 0
            for frame in predictions["frames"]:
                count += len(validate_predictions(frame, schema.classes, schema.data_type))
            if count != metrics["box_count"]:
                raise ValueError("Box count proxy không khớp predictions")
            if memory is not None:
                from .pseudo_memory import validate as validate_memory, positives as memory_positives
                active = predictions.get('adaptation_classes', project['schema']['classes'])
                validate_memory(memory, project['dataset_sha256'], active, expected)
                if memory['round_index'] != index:
                    raise ValueError('Memory không khớp vòng snapshot')
                memory_boxes = memory_positives(memory)
                if any(sorted(f['boxes'], key=lambda b:b['id']) !=
                       sorted(memory_boxes[f['frame_id']], key=lambda b:b['id']) for f in predictions['frames']):
                    raise ValueError('Memory positives không khớp nhãn snapshot')
            if set(sample_predictions) != set(project["sample_frame_ids"]):
                raise ValueError("Sample predict cần cùng frame cố định mỗi vòng")
            for document in sample_predictions.values():
                validate_predictions(document, schema.classes, schema.data_type)
            by_frame = {f["frame_id"]: f["boxes"] for f in predictions["frames"]}
            if any(document["boxes"] != by_frame[fid] for fid, document in sample_predictions.items()):
                raise ValueError("Sample predict không khớp snapshot nhãn")
            if len(checkpoints) != 2 or set(checkpoints) != {"A", "B"}:
                raise ValueError("Vòng cần checkpoint của hai detector A/B")
            checkpoint_refs = {key: dict(path=str(Path(path).absolute()), sha256=digest(path))
                               for key, path in checkpoints.items()}
            version = f"M{index}"
            destination = root / "rounds" / version
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                raise ValueError("Snapshot đã tồn tại; giữ lại để kiểm tra")
            temporary = destination.with_name(".pending_" + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                write(temporary / "labels.json", predictions)
                write(temporary / "samples.json", sample_predictions)
                if memory is not None:
                    write(temporary / 'memory.json', memory)
                for key, reference in checkpoint_refs.items():
                    copied = temporary / (key + ".pth")
                    shutil.copyfile(reference["path"], copied)
                    if digest(copied) != reference["sha256"]:
                        raise ValueError("Checkpoint thay đổi trong lúc snapshot")
                    reference["path"] = str(destination / (key + ".pth"))
                previous = (read(root / "rounds" / project["selected_version"] / "round.json")["metrics"]
                            if project["selected_version"] else None)
                decision = decide(metrics, previous, SelectionPolicy(**project["selection_policy"]))
                write(temporary / "round.json", dict(version=version, completed_at=now(), metrics=metrics,
                      checkpoints=checkpoint_refs, decision=decision,
                      selection_policy=project['selection_policy'],
                      work_directory=str(Path(work_directory).absolute()) if work_directory else None,
                      labels_sha256=digest(temporary / "labels.json"),
                      samples_sha256=digest(temporary / "samples.json")))
                if memory is not None:
                    receipt = read(temporary / 'round.json')
                    receipt.update(memory_sha256=digest(temporary / 'memory.json'), memory_summary=memory['summary'])
                    write(temporary / 'round.json', receipt)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            project["completed_rounds"].append(version)
            if decision["action"] == "accept":
                project["selected_version"] = version
            if decision["action"] != "review":
                project["reviewed_round"] = version
            write(root / "project.json", project)
            return decision

    def select(self, project_id, version):
        root = self.path(project_id)
        with lock(root / ".lock"):
            project = self.get(project_id)
            if version not in project["completed_rounds"]:
                raise ValueError("Chỉ chọn vòng đã hoàn tất")
            directory = inside(root, "rounds/" + version)
            receipt = read(directory / "round.json")
            for name in ("labels", "samples"):
                if digest(directory / (name + ".json")) != receipt[name + "_sha256"]:
                    raise ValueError("Snapshot bị thay đổi")
            for reference in receipt["checkpoints"].values():
                if digest(reference["path"]) != reference["sha256"]:
                    raise ValueError("Checkpoint bị thay đổi")
            if receipt.get('memory_sha256') and digest(directory / 'memory.json') != receipt['memory_sha256']:
                raise ValueError('Memory snapshot bị thay đổi')
            project["selected_version"] = version
            project["reviewed_round"] = project["completed_rounds"][-1]
            write(root / "project.json", project)

    def request_finish(self, project_id):
        root = self.path(project_id)
        with lock(root / ".lock"):
            project = self.get(project_id)
            project["stop_after_round"] = True
            write(root / "project.json", project)

    def allow_continue(self, project_id):
        """Clear a finish request; never dispatch or resume a trainer."""
        root = self.path(project_id)
        from .adaptation import active_job
        with lock(root / '.adaptation_dispatch.lock'):
            if active_job(self, project_id):
                raise ValueError('Chờ job hiện tại dừng trước khi cho phép chạy tiếp')
            with lock(root / '.lock'):
                project = self.get(project_id)
                if len(project['completed_rounds']) >= project['max_rounds']:
                    raise ValueError('Đã hết số vòng; tăng số vòng tối đa trong Thiết lập dự án trước')
                project['stop_after_round'] = False
                write(root / 'project.json', project)
        return project

    def next_round(self, project_id):
        project = self.get(project_id)
        if project["stop_after_round"] or len(project["completed_rounds"]) >= project["max_rounds"]:
            return None
        if project["completed_rounds"] and project["selected_version"] is None:
            raise ValueError("Chọn phiên bản nhãn trước khi train vòng tiếp")
        if project["completed_rounds"] and project["reviewed_round"] != project["completed_rounds"][-1]:
            raise ValueError("Cần duyệt vòng mới nhất trước khi train tiếp")
        return len(project["completed_rounds"]) + 1
