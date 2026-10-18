"""Sequential two-detector rounds; execution policy and selection policy are separate."""
from copy import deepcopy
from pathlib import Path
import re
import shutil
import numpy as np
from .device import gpu_idle
from .fusion import calibrate_source, consensus_round, normalize, source_calibration_report
from .io import digest, lock, now, read, write
from .predictions import validate_predictions
from .schema import ClassSchema
from .tracking import refine
from .pseudo_memory import MemoryPolicy, update as update_memory, validate as validate_memory
from .memory_store import load as load_memory, context as memory_context

MIN_POSITIVE_POINTS = 5
# Agreement = accepted A/B pairs / confident boxes of the busier detector per frame.
# Raw score>=.05 boxes (thousands per round) made the old ratio ~5% and unreachable.
AGREEMENT_CONFIDENCE = .7
AGREEMENT_DEFINITION = 'accepted_pairs_over_confident_boxes_v2'


class RoundController:
    def __init__(self, store, project_id, adapters, initial_checkpoints, point_loader,
                 epochs=1, tracking_mode="off", fixed_sensor_confirmed=False, gpu_guard=gpu_idle,
                 training_options=None, on_stage=None, attempt_id=None, adaptation_classes=None, resume_plan=None,
                 memory_policy=None):
        if set(adapters) != {"A", "B"} or set(initial_checkpoints) != {"A", "B"}:
            raise ValueError("Cần hai adapter A/B và hai checkpoint khởi tạo")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("Epoch phải >0")
        if tracking_mode not in ("off", "pose", "fixed_sensor"):
            raise ValueError("MVP không dùng tracking chưa bù ego để chọn nhãn train")
        self.store, self.project_id, self.adapters = store, project_id, adapters
        self.root = store.path(project_id)
        self.initial = {key: str(Path(value).absolute()) for key, value in initial_checkpoints.items()}
        self.point_loader, self.gpu_guard = point_loader, gpu_guard
        self.epochs, self.tracking_mode = epochs, tracking_mode
        self.fixed_sensor_confirmed = fixed_sensor_confirmed
        self.training_options = training_options or {}
        self.on_stage = on_stage or (lambda stage, key=None: None)
        if attempt_id is not None and not re.fullmatch(r'[a-zA-Z0-9_-]+', attempt_id):
            raise ValueError('Attempt ID không hợp lệ')
        self.attempt_id = attempt_id
        self.active_work = None
        schema_classes = store.get(project_id)['schema']['classes']
        chosen = schema_classes if adaptation_classes is None else adaptation_classes
        if not chosen or len(set(chosen)) != len(chosen) or not set(chosen).issubset(schema_classes):
            raise ValueError('Class adaptation phải là tập con không rỗng của class dự án')
        self.adaptation_classes = [name for name in schema_classes if name in chosen]
        self.resume_plan = resume_plan
        self.memory_policy = MemoryPolicy(**(memory_policy or {}))

    def gpu(self, function, *args):
        # Shared with source jobs; adapter callbacks are blocking leaf operations.
        with lock(self.root.parents[1] / ".gpu.lock", blocking=False):
            self.gpu_guard()
            result = function(*args)
            self.gpu_guard()  # Verify prior worker stopped before releasing the lease.
            return result

    def predict(self, checkpoints, frames, directory, classes):
        if directory.exists():
            cache = read(directory / "predictions.json")
            if cache["checkpoint_hashes"] != {k: digest(v) for k, v in checkpoints.items()}:
                raise ValueError("Prediction cache khác checkpoint")
            return cache["models"]
        directory.mkdir(parents=True)
        write(directory / 'checkpoints.json', dict(checkpoint_hashes={k: digest(v) for k, v in checkpoints.items()}))
        calibrations = {}
        # Check source evidence for both models before spending GPU time on
        # target prediction. A scarce class must never fall back to raw scores.
        for key in ('A', 'B'):
            checkpoint_hash = digest(checkpoints[key])
            self.on_stage('calibrate_source_validation', key)
            source = self.gpu(self.adapters[key].predict_source_validation, checkpoints[key])
            report = source_calibration_report(source, self.adaptation_classes)
            write(directory / (key + '_calibration_report.json'),
                  dict(report, checkpoint_sha256=checkpoint_hash))
            calibrations[key] = calibrate_source(source, checkpoint_hash, self.adaptation_classes)
            write(directory / (key + '_calibration.json'), calibrations[key])
        result = {}
        for key in ("A", "B"):
            checkpoint_hash = digest(checkpoints[key])
            self.on_stage('predict_target', key)
            raw = self.gpu(self.adapters[key].predict_target, checkpoints[key], frames)
            if set(raw) != {f["frame_id"] for f in frames}:
                raise ValueError("Adapter predict thiếu/trùng frame")
            result[key] = {fid: normalize(value, calibrations[key], checkpoint_hash, classes, self.adaptation_classes)
                           for fid, value in raw.items()}
            for fid, boxes in result[key].items():
                for box in boxes:
                    box["native_box_id"] = box["id"]
                    box["id"] = f'{key}:{checkpoint_hash[:12]}:{fid}:{box["id"]}'
        write(directory / "predictions.json", dict(models=result,
              checkpoint_hashes={k: digest(v) for k, v in checkpoints.items()}, target_gold_read=False))
        return result

    def shared(self, frames, current, previous=None):
        output = []
        matched, selected, supported, trained = 0, 0, 0, 0
        for frame in frames:
            fid = frame["frame_id"]
            def active(boxes):
                return [b for b in boxes if b['class_name'] in self.adaptation_classes]
            consensus = consensus_round(active(previous["A"][fid]) if previous else None, active(current["A"][fid]),
                                        active(previous["B"][fid]) if previous else None, active(current["B"][fid]))
            excluded = {box['id']: dict(deepcopy(box), training_policy='ignore_region', reason='excluded_adaptation_class')
                        for state in ([previous, current] if previous else [current])
                        for key in ('A', 'B') for box in state[key][fid]
                        if box['class_name'] not in self.adaptation_classes}
            row = dict(deepcopy(frame), boxes=consensus["boxes"], ignored=consensus["ignored"] + list(excluded.values()))
            points = self.point_loader(frame)
            from .reused.core import points_in_box
            for box in row["boxes"]:
                box["point_count"] = int(points_in_box(points, box["box"]).sum())
                supported += box["point_count"] >= MIN_POSITIVE_POINTS
            selected += len(row["boxes"])
            matched += len(row["boxes"])
            # Same rule as final delivery and memory: an almost empty box is not taught as an object.
            empty = [b for b in row["boxes"] if b["point_count"] < MIN_POSITIVE_POINTS]
            row["boxes"] = [b for b in row["boxes"] if b["point_count"] >= MIN_POSITIVE_POINTS]
            row["ignored"].extend(dict(b, training_policy="ignore_region", reason="insufficient_points") for b in empty)
            trained += len(row["boxes"])
            output.append(row)
        if self.tracking_mode != "off":
            tracked = refine(output, self.point_loader, self.store.get(self.project_id)["schema"]["classes"],
                             mode=self.tracking_mode, fixed_sensor_confirmed=self.fixed_sensor_confirmed)
            for row, refined in zip(output, tracked["frames"]):
                row["boxes"] = refined["boxes"]
                # Only geometrically supported interpolations become ignores; none become positives.
                row["ignored"].extend(p for p in refined["proposals"] if p["training_policy"] == "ignore_region")
                row["proposals"] = refined["proposals"]
        denominator = sum(max(sum(b['class_name'] in self.adaptation_classes
                                  and b.get('confidence', 0) >= AGREEMENT_CONFIDENCE for b in current[k][f['frame_id']])
                              for k in ('A', 'B')) for f in frames)
        excluded_classes = [c for c in self.store.get(self.project_id)['schema']['classes'] if c not in self.adaptation_classes]
        proxies = dict(domain="target_unlabelled_proxy", target_gold_read=False, box_count=selected,
                       trained_box_count=trained, min_positive_points=MIN_POSITIVE_POINTS,
                       agreement=matched / max(1, denominator), supported_fraction=supported / max(1, selected),
                       agreement_definition=AGREEMENT_DEFINITION, agreement_confidence=AGREEMENT_CONFIDENCE,
                       confident_box_denominator=denominator,
                       adaptation_classes=self.adaptation_classes, excluded_classes=excluded_classes)
        return dict(target_gold_read=False, frames=output, adaptation_classes=self.adaptation_classes,
                    excluded_classes=excluded_classes), proxies

    def run_one(self):
        with lock(self.root / ".controller.lock", blocking=False):
            index = self.store.next_round(self.project_id)
            if index is None:
                return dict(status="finished")
            inventory = self.store.verify_dataset(self.project_id)
            schema = ClassSchema.parse(inventory["schema"])
            if schema.data_type != "3d":
                raise ValueError("Controller ST3D hiện chỉ chạy 3D; chưa có detector adapter 2D")
            frames = [f for f in inventory["frames"] if f["split"] == "adaptation"]
            project = self.store.get(self.project_id)
            work = self.root / "work" / f"M{index}"
            if self.attempt_id:
                work = work / 'attempts' / self.attempt_id
            if work.exists():
                raise ValueError("Giữ artifacts vòng bị ngắt. Không tự chạy lại; tạo attempt sau khi kiểm tra")
            work.mkdir(parents=True)
            self.active_work = work
            write(work / "state.json", dict(status="running", started_at=now(), target_gold_read=False))
            try:
                continuation = self.resume_plan
                if continuation and continuation['index'] != index:
                    raise ValueError('Lượt cần tiếp tục không thuộc vòng hiện tại')
                if project["selected_version"]:
                    selected = self.root / "rounds" / project["selected_version"]
                    trained = read(selected / "round.json")
                    if digest(selected / 'labels.json') != trained['labels_sha256'] or any(
                            digest(value['path']) != value['sha256'] for value in trained['checkpoints'].values()):
                        raise ValueError('Phiên bản nhãn/checkpoint đang chọn đã thay đổi')
                    checkpoints = {key: value["path"] for key, value in trained["checkpoints"].items()}
                    shared_labels = selected / "labels.json"
                else:
                    checkpoints = self.initial
                    shared_labels = work / "initial_shared.json"
                if continuation and continuation['before_cache']:
                    source = Path(continuation['before_cache'])
                    if digest(source) != continuation['before_sha256']:
                        raise ValueError('Cache cần tiếp tục đã thay đổi')
                    (work / 'before').mkdir()
                    shutil.copyfile(source, work / 'before/predictions.json')
                previous = self.predict(checkpoints, frames, work / "before", schema.classes)
                if not project["selected_version"]:
                    if continuation and continuation['labels_path']:
                        if digest(continuation['labels_path']) != continuation['labels_sha256']:
                            raise ValueError('Pseudo-label của lượt cũ đã đổi')
                        shutil.copyfile(continuation['labels_path'], shared_labels)
                    else:
                        initial_labels, _ = self.shared(frames, previous)
                        write(shared_labels, initial_labels)
                memory_before = None
                if self.memory_policy.enabled:
                    self.on_stage('load_pseudo_memory')
                    saved_memory = continuation.get('memory_before') if continuation else None
                    if saved_memory:
                        if digest(saved_memory) != continuation['memory_before_sha256']:
                            raise ValueError('Memory cần tiếp tục đã đổi')
                        memory_before = validate_memory(read(saved_memory), project['dataset_sha256'],
                                                        self.adaptation_classes, [f['frame_id'] for f in frames])
                    else:
                        memory_before = load_memory(self.store, self.project_id, frames, self.adaptation_classes,
                                                    self.memory_policy, self.point_loader)
                    if memory_before['round_index'] == -1:
                        initial = read(shared_labels)
                        write(work / 'initial_consensus.json', initial)
                        memory_before, remembered = update_memory(memory_before, initial, previous,
                            memory_context(0, {k: digest(v) for k,v in checkpoints.items()}, 'initial_pseudo',
                                           labels_sha256=digest(work / 'initial_consensus.json')),
                            self.point_loader, self.memory_policy)
                        # Legacy resumed trainers bind the original pseudo-label hash.
                        if not continuation:
                            write(shared_labels, remembered)
                    if memory_before['round_index'] != index - 1:
                        raise ValueError('Memory không thuộc vòng trước')
                    write(work / 'memory_before.json', memory_before)
                if not any(f["boxes"] for f in read(shared_labels)["frames"]):
                    raise ValueError("Chưa có pseudo-label dương đủ bằng chứng để train")
                next_checkpoints = {}
                for key in ("A", "B"):
                    if continuation and key in continuation['completed']:
                        receipt = continuation['completed'][key]
                        if digest(receipt['checkpoint']) != receipt['checkpoint_sha256']:
                            raise ValueError('Checkpoint model đã hoàn tất bị đổi')
                        next_checkpoints[key] = receipt['checkpoint']
                        write(work / (key + '_trained.json'), receipt)
                        continue
                    self.on_stage('train_target', key)
                    # Per-detector run directory is distinct and must be registered by its adapter.
                    output = work / key
                    policy = dict(self.training_options, epochs=self.epochs, ckpt_interval=10, save_final=True)
                    if continuation and key in continuation['checkpoints']:
                        policy['resume_checkpoint'] = continuation['checkpoints'][key]
                    receipt = self.gpu(self.adapters[key].train_target, checkpoints[key], str(shared_labels), str(output), policy)
                    if (receipt.get("target_gold_used") is not False or receipt.get("optimizer_steps", 0) <= 0
                        or receipt.get("shared_labels_sha256") != digest(shared_labels)
                        or receipt.get("checkpoint_sha256") != digest(receipt["checkpoint"])):
                        raise ValueError("Trainer receipt không chứng minh đã train đúng nhãn chung/checkpoint")
                    next_checkpoints[key] = receipt["checkpoint"]
                    write(work / (key + "_trained.json"), receipt)
                current = self.predict(next_checkpoints, frames, work / "after", schema.classes)
                self.on_stage('fuse_and_snapshot')
                predictions, proxies = self.shared(frames, current, previous)
                memory_after = None
                if memory_before is not None:
                    self.on_stage('update_pseudo_memory')
                    memory_after, predictions = update_memory(memory_before, predictions, current,
                        memory_context(index, {k: digest(v) for k,v in next_checkpoints.items()},
                                       labels_sha256=digest(shared_labels)),
                        self.point_loader, self.memory_policy)
                    write(work / 'memory_after.json', memory_after)
                    # Historical retention is not fresh A/B agreement.
                    proxies.update(current_consensus_box_count=proxies['box_count'],
                        box_count=sum(len(f['boxes']) for f in predictions['frames']),
                        memory=memory_after['summary'])
                    proxies['supported_fraction'] = (sum(b['point_count'] >= 5 for f in predictions['frames']
                                                       for b in f['boxes']) / max(1, proxies['box_count']))
                sample_ids = project["sample_frame_ids"]
                samples = {f["frame_id"]: dict(boxes=f["boxes"]) for f in predictions["frames"] if f["frame_id"] in sample_ids}
                write(work / "model_samples.json", {key: {fid: {
                    'boxes': [b for b in values[fid] if b['class_name'] in self.adaptation_classes],
                    'ignored': [b for b in values[fid] if b['class_name'] not in self.adaptation_classes]} for fid in sample_ids}
                                                     for key, values in current.items()})
                decision = self.store.record_round(self.project_id, index, predictions, proxies, next_checkpoints, samples,
                                                   work_directory=work, memory=memory_after)
                write(work / "state.json", dict(status="completed", finished_at=now(), version=f"M{index}", decision=decision))
                return dict(status="completed", version=f"M{index}", decision=decision)
            except BaseException as error:
                write(work / "state.json", dict(status="interrupted_or_failed", error=str(error), finished_at=now()))
                raise

    def run(self):
        results = []
        while self.store.next_round(self.project_id) is not None:
            result = self.run_one()
            results.append(result)
            # Recovery belongs to the interrupted round only. Further automatic
            # rounds use their selected snapshot and train both models normally.
            if result.get('status') == 'completed':
                self.resume_plan = None
            if self.store.get(self.project_id)["execution_mode"] == "manual" or result.get("decision", {}).get("action") == "review":
                break
            if self.store.get(self.project_id)['selected_version'] is None:
                break
        return results
