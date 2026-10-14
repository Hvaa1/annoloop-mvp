"""Target pseudo-label conversion. No target annotations are opened here."""
from copy import deepcopy
import numpy as np
from .io import digest
from .inspection import load_xyz
from .predictions import assert_no_gold, validate_predictions


def signed_labels(frame, classes):
    assert_no_gold(frame)
    rows, names, scores = [], [], []
    for key, sign in (("boxes", 1), ("ignored", -1)):
        boxes = frame.get(key, [])
        validate_predictions({"boxes": boxes}, classes)
        for box in boxes:
            expected = "positive" if sign == 1 else "ignore_region"
            if box.get("training_policy") != expected:
                raise ValueError("Pseudo-label thiếu training_policy đúng: " + expected)
            rows.append([*box["box"], sign * (classes.index(box["class_name"]) + 1)])
            names.append(box["class_name"])
            scores.append(box.get("quality", box["score"]))
    return (np.asarray(rows, dtype=np.float32).reshape(-1, 8),
            np.asarray(names), np.asarray(scores, dtype=np.float32))


def make_target_dataset(config, classes, frames, bin_columns, labels=None, logger=None):
    """Import native dataset only in the model worker's compatible environment."""
    from pcdet.datasets.dataset import DatasetTemplate
    from .reused.core import points_in_box

    class TargetDataset(DatasetTemplate):
        def __init__(self):
            target_config = deepcopy(config)
            target_config.pop('USE_PSEUDO_LABEL', None)
            target_config.pop('TARGET', None)
            target_config.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z']
            target_config.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z']
            # World transforms preserve signed class IDs/scores. No GT sampling
            # or object augmentor that can change array alignment is used.
            if labels is not None:
                target_config.DATA_AUGMENTOR.AUG_CONFIG_LIST = [
                    a for a in target_config.DATA_AUGMENTOR.AUG_CONFIG_LIST
                    if a.NAME in ('random_world_flip', 'random_world_rotation', 'random_world_scaling')]
            super().__init__(target_config, classes, labels is not None, logger=logger)
            self.frames = frames
            self.labels = labels
            self.depth = 0
            if labels is not None:
                assert_no_gold(labels)
                by_id = {f['frame_id']: f for f in labels['frames']}
                if len(by_id) != len(labels['frames']) or set(by_id) != {f['frame_id'] for f in frames}:
                    raise ValueError('Shared labels không khớp adaptation frames')
                self.by_id = by_id
                self.frames = []
                for frame in frames:
                    row = by_id[frame['frame_id']]
                    boxes, _, _ = signed_labels(row, classes)
                    points = self.points(frame)
                    shift = np.asarray(config.get('SHIFT_COOR', [0, 0, 0]))
                    bounds = np.asarray(config.POINT_CLOUD_RANGE)
                    usable = [b for b in boxes if b[-1] > 0
                              and np.all(b[:3] + shift >= bounds[:3])
                              and np.all(b[:3] + shift < bounds[3:])
                              and points_in_box(points, b[:7]).sum() >= config.get('MIN_POINTS_OF_GT', 1)]
                    if usable:
                        self.frames.append(frame)
                if not self.frames:
                    raise ValueError('Không có pseudo-label dương trong range với đủ điểm; không train nền rỗng')

        def points(self, frame):
            if digest(frame['media']) != frame['sha256']:
                raise ValueError('Target media thay đổi')
            return load_xyz(frame['media'], bin_columns)

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, index):
            self.depth += 1
            try:
                if self.depth > 20:
                    raise ValueError('Pseudo-label bị loại sau augmentation/range 20 lần; dừng thay vì resample vô hạn')
                frame = self.frames[index]
                shift = np.asarray(config.get('SHIFT_COOR', [0, 0, 0]), dtype=np.float32)
                data = dict(points=self.points(frame) + shift, frame_id=frame['frame_id'])
                if self.labels is not None:
                    boxes, names, scores = signed_labels(self.by_id[frame['frame_id']], classes)
                    data.update(gt_boxes=boxes[:, :7].copy(), gt_classes=boxes[:, 7].copy(),
                                gt_names=names, gt_scores=scores)
                    data['gt_boxes'][:, :3] += shift
                result = self.prepare_data(data)
                if self.labels is not None and not (result['gt_boxes'][:, -1] > 0).any():
                    return self.__getitem__(int(np.random.randint(len(self))))
                return result
            finally:
                self.depth -= 1

    return TargetDataset()
