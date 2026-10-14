"""Offline tracking. Generated boxes are proposals, never gold labels."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import numpy as np

from .core import points_in_box, validate_predictions


@dataclass
class Settings:
    min_score: float = .5
    min_supported_score: float = .2
    max_missing_frames: int = 2
    min_points: int = 10
    base_gate_m: float = 2.
    max_speed_mps: float = 25.
    max_gap_seconds: float = .65
    max_size_ratio: float = 1.5
    max_yaw_change: float = np.pi / 3
    ambiguity_margin_m: float = .4
    extrapolate_one_frame: bool = False
    allow_estimated_pose_for_ignore: bool = False

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Threshold phải hữu hạn')
        if not 0 <= self.min_score <= 1 or self.min_points < 0:
            raise ValueError('min_score/min_points không hợp lệ')
        if not 0 <= self.min_supported_score <= self.min_score:
            raise ValueError('min_supported_score phải nằm giữa 0 và min_score')
        if isinstance(self.max_missing_frames, bool) or not isinstance(self.max_missing_frames, int) or self.max_missing_frames < 0:
            raise ValueError('max_missing_frames phải là số nguyên >= 0')
        if any(getattr(self, k) <= 0 for k in ('base_gate_m', 'max_speed_mps',
                                               'max_gap_seconds', 'max_yaw_change')):
            raise ValueError('Gate/thời gian phải > 0')
        if self.max_size_ratio < 1 or self.ambiguity_margin_m < 0:
            raise ValueError('Size ratio/ambiguity không hợp lệ')


def angle_delta(a, b):
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def box_yaw_delta(a, b):
    """Cuboid axes repeat after pi; retain directed detector yaw separately."""
    return float((a - b + np.pi/2) % np.pi - np.pi/2)


def validate_pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('lidar_to_world phải là ma trận 4x4 hữu hạn')
    r = pose[:3, :3]
    if not (np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError('Pose phải là rigid transform hợp lệ')
    # Current detector schema has yaw only, not a full 3D box orientation.
    if not np.allclose(r[2], [0, 0, 1], atol=1e-4):
        raise ValueError('Pilot chỉ hỗ trợ pose yaw + translation; roll/pitch cần adapter full 3D')
    return pose


def transform_box(box, pose):
    result = np.array(box, dtype=float, copy=True)
    result[:3] = pose[:3, :3] @ result[:3] + pose[:3, 3]
    result[6] = angle_delta(result[6] + np.arctan2(pose[1, 0], pose[0, 0]), 0)
    return result


def refine_sequence(frames, point_loader, settings=None, mode='pose'):
    """Frames: frame_id, sequence_id, timestamp, boxes, optional lidar_to_world.

    mode='unaligned_diagnostic' explicitly permits exploratory local-frame matching.
    mode='fixed_sensor' requires caller confirmation that sensor coordinates are fixed.
    """
    cfg = settings or Settings()
    if mode not in ('pose', 'fixed_sensor', 'unaligned_diagnostic'):
        raise ValueError('Coordinate mode không hợp lệ')
    aligned = mode != 'unaligned_diagnostic'
    output = deepcopy(frames)
    if not frames:
        raise ValueError('Sequence rỗng')
    ids, previous = set(), {}
    poses = []
    for frame in frames:
        fid, seq, t = frame['frame_id'], frame['sequence_id'], frame['timestamp']
        if fid in ids or not seq or not np.isfinite(t):
            raise ValueError('Frame ID trùng / sequence hoặc timestamp không hợp lệ')
        if seq in previous and t <= previous[seq]:
            raise ValueError('Timestamp phải tăng nghiêm ngặt trong mỗi sequence')
        ids.add(fid)
        previous[seq] = t
        validate_predictions({'boxes': frame['boxes']})
        poses.append(validate_pose(frame['lidar_to_world']) if mode == 'pose' else np.eye(4))
    tracks = []
    world_boxes = []
    for i, frame in enumerate(output):
        frame['proposals'] = []
        for b in frame['boxes']:
            b.update(track_id=None, origin='detector', review_status='unreviewed', temporal_supported=False)
            b['temporal_flags'] = ([] if aligned else ['EGO_MOTION_UNCOMPENSATED'])
            if frame.get('pose_source','').startswith('estimated'):
                b['temporal_flags'].append('EGO_POSE_ESTIMATED')
        world = [transform_box(b['box'], poses[i]) for b in frame['boxes']]
        world_boxes.append(world)
        candidates = []
        for ti, track in enumerate(tracks):
            last_i, last_b = track['observations'][-1]
            dt = frame['timestamp'] - frames[last_i]['timestamp']
            if (track['sequence_id'] != frame['sequence_id'] or i - last_i > cfg.max_missing_frames + 1
                    or not 0 < dt <= cfg.max_gap_seconds):
                continue
            last = world_boxes[last_i][last_b]
            predicted = last[:3].copy()
            if len(track['observations']) >= 2:
                prev_i, prev_b = track['observations'][-2]
                velocity = (last[:3] - world_boxes[prev_i][prev_b][:3]) / (
                    frames[last_i]['timestamp'] - frames[prev_i]['timestamp'])
                predicted += velocity * dt
            for bi, b in enumerate(frame['boxes']):
                if b['class_name'] != 'car' or b['score'] < cfg.min_score or not b.get('tracking_eligible', True):
                    continue
                cur = world[bi]
                ratio = np.maximum(cur[3:6] / last[3:6], last[3:6] / cur[3:6])
                distance = float(np.linalg.norm(cur[:3] - predicted))
                displacement = float(np.linalg.norm(cur[:3] - last[:3]))
                if distance <= cfg.base_gate_m:
                    if ratio.max() > cfg.max_size_ratio:
                        b['temporal_flags'].append('TEMPORAL_SIZE_CHANGE')
                    if abs(box_yaw_delta(cur[6], last[6])) > cfg.max_yaw_change:
                        b['temporal_flags'].append('TEMPORAL_YAW_CHANGE')
                    elif abs(angle_delta(cur[6], last[6])) > cfg.max_yaw_change:
                        b['temporal_flags'].append('TEMPORAL_HEADING_FLIP')
                if (distance <= cfg.base_gate_m + cfg.max_speed_mps * dt
                        and displacement <= cfg.base_gate_m + cfg.max_speed_mps * dt
                        and ratio.max() <= cfg.max_size_ratio
                        and abs(box_yaw_delta(cur[6], last[6])) <= cfg.max_yaw_change):
                    candidates.append((distance, ti, bi))
        # Reject near ties instead of silently switching IDs at crossings.
        ambiguous = set()
        for d, ti, bi in candidates:
            if any((tj == ti or bj == bi) and (tj, bj) != (ti, bi)
                   and abs(d - dj) < cfg.ambiguity_margin_m for dj, tj, bj in candidates):
                ambiguous.add((ti, bi))
                frame['boxes'][bi]['temporal_flags'].append('AMBIGUOUS_ASSOCIATION')
        used_t, used_b = set(), set()
        for _, ti, bi in sorted(candidates):
            if ti in used_t or bi in used_b or (ti, bi) in ambiguous:
                continue
            tracks[ti]['observations'].append((i, bi))
            frame['boxes'][bi]['track_id'] = tracks[ti]['track_id']
            used_t.add(ti)
            used_b.add(bi)
        for bi, b in enumerate(frame['boxes']):
            b['temporal_flags'] = sorted(set(b['temporal_flags']))
            if (b['class_name'] == 'car' and b['score'] >= cfg.min_score
                    and b.get('tracking_eligible', True) and bi not in used_b):
                tid = f'car_{len(tracks) + 1:06d}'
                tracks.append(dict(track_id=tid, sequence_id=frame['sequence_id'], observations=[(i, bi)]))
                b['track_id'] = tid
    points_cache, support_cache = {}, {}

    def support_count(target, box, key=None):
        if key is not None and (target, key) in support_cache:
            return support_cache[target, key]
        if target not in points_cache:
            pts = np.asarray(point_loader(frames[target]), dtype=float)
            if pts.ndim != 2 or pts.shape[1] < 3 or not np.isfinite(pts).all():
                raise ValueError('Point cloud không hợp lệ')
            points_cache[target] = pts
        count = int(points_in_box(points_cache[target], box).sum())
        if key is not None:
            support_cache[target, key] = count
        return count

    def request(track_index, target, left, right, kind):
        track = tracks[track_index]
        li, lb = left
        ri, rb = right
        f = output[target]
        if f['sequence_id'] != track['sequence_id']:
            return
        t0, t1, t = frames[li]['timestamp'], frames[ri]['timestamp'], f['timestamp']
        if t1 - t0 > cfg.max_gap_seconds or min(abs(t-t0), abs(t-t1)) > cfg.max_gap_seconds:
            return
        a, b = world_boxes[li][lb], world_boxes[ri][rb]
        alpha = (t - t0) / (t1 - t0)
        box = (1-alpha) * a + alpha * b
        box[3:6] = (a[3:6] + b[3:6]) / 2
        box[6] = angle_delta(a[6] + alpha * box_yaw_delta(b[6], a[6]), 0)
        local = transform_box(box, np.linalg.inv(poses[target]))
        estimated = any(frames[k].get('pose_source','').startswith('estimated') for k in (li,ri,target))
        pose_ok = all(frames[k].get('pose_quality_passed', True) for k in (li,ri,target))
        return dict(track_index=track_index, target=target, left=left, right=right, kind=kind,
                    local=local, estimated=estimated, pose_ok=pose_ok,
                    source_frames=[frames[li]['frame_id'], frames[ri]['frame_id']],
                    source_box_ids=[frames[li]['boxes'][lb]['id'], frames[ri]['boxes'][rb]['id']],
                    temporal_quality_hint=float(min(frames[li]['boxes'][lb]['score'],
                                                    frames[ri]['boxes'][rb]['score']) * (.8 if kind == 'interpolated' else .5)))

    # Freeze high-confidence anchors before adding any weak observations.
    requests = []
    for track_index, track in enumerate(tracks):
        track['anchor_observations'] = list(track['observations'])
        obs = track['anchor_observations']
        for left, right in zip(obs, obs[1:]):
            if 1 <= right[0] - left[0] - 1 <= cfg.max_missing_frames:
                for target in range(left[0] + 1, right[0]):
                    candidate = request(track_index, target, left, right, 'interpolated')
                    if candidate is not None:
                        requests.append(candidate)
        if cfg.extrapolate_one_frame and len(obs) >= 2:
            for target, left, right, kind in ((obs[0][0]-1, obs[0], obs[1], 'extrapolated_backward'),
                                             (obs[-1][0]+1, obs[-2], obs[-1], 'extrapolated_forward')):
                if 0 <= target < len(frames):
                    candidate = request(track_index, target, left, right, kind)
                    if candidate is not None:
                        requests.append(candidate)

    # Resolve weak-box support jointly so one detection cannot acquire two IDs.
    candidates = []
    for ri, req in enumerate(requests):
        if req['kind'] != 'interpolated':
            continue
        target, local = req['target'], req['local']
        for bi, b in enumerate(output[target]['boxes']):
            if b['class_name'] != 'car' or b['score'] >= cfg.min_score or b['track_id'] is not None:
                continue
            cur = np.asarray(b['box'])
            distance = float(np.linalg.norm(cur[:3] - local[:3]))
            if distance >= cfg.base_gate_m:
                continue
            flags = b['temporal_flags']
            flags.append('TEMPORAL_SUPPORT_EXISTING_BOX')
            if b['score'] < cfg.min_supported_score:
                flags.append('TEMPORAL_SUPPORT_SCORE_TOO_LOW')
                continue
            if not aligned or not req['pose_ok']:
                flags.append('TEMPORAL_SUPPORT_POSE_UNVERIFIED')
                continue
            ratio = np.maximum(cur[3:6]/local[3:6], local[3:6]/cur[3:6])
            if ratio.max() > cfg.max_size_ratio or abs(box_yaw_delta(cur[6], local[6])) > cfg.max_yaw_change:
                flags.append('TEMPORAL_SUPPORT_GEOMETRY_REJECTED')
                continue
            if abs(angle_delta(cur[6], local[6])) > cfg.max_yaw_change:
                flags.append('TEMPORAL_HEADING_FLIP')
            count = support_count(target, b['box'], bi)
            if count < cfg.min_points:
                flags.append('TEMPORAL_SUPPORT_LOW_POINT_SUPPORT')
                continue
            candidates.append((distance, ri, bi, count))
    ambiguous = set()
    for distance, ri, bi, _ in candidates:
        target = requests[ri]['target']
        if any((rj == ri or (requests[rj]['target'] == target and bj == bi))
               and (rj, bj) != (ri, bi) and abs(distance-other) < cfg.ambiguity_margin_m
               for other, rj, bj, _ in candidates):
            ambiguous.add((ri, bi))
            output[target]['boxes'][bi]['temporal_flags'].append('TEMPORAL_SUPPORT_AMBIGUOUS')
    ambiguous_requests = {ri for ri, _ in ambiguous}
    ambiguous_boxes = {(requests[ri]['target'], bi) for ri, bi in ambiguous}
    used_requests, used_boxes = set(), set()
    for distance, ri, bi, count in sorted(candidates):
        req = requests[ri]
        target = req['target']
        if ri in ambiguous_requests or (target, bi) in ambiguous_boxes or ri in used_requests or (target, bi) in used_boxes:
            continue
        track = tracks[req['track_index']]
        b = output[target]['boxes'][bi]
        b.update(track_id=track['track_id'], temporal_supported=True, qa_status='REVIEW',
                 training_policy='review_only', point_count=count,
                 source_frames=req['source_frames'], source_box_ids=req['source_box_ids'],
                 temporal_quality_hint=req['temporal_quality_hint'], support_distance_m=distance)
        b['temporal_flags'].append('TEMPORAL_SUPPORTED_DETECTION')
        track['observations'].append((target, bi))
        used_requests.add(ri)
        used_boxes.add((target, bi))

    def proposal(req):
        track = tracks[req['track_index']]
        target, local, kind = req['target'], req['local'], req['kind']
        f = output[target]
        # Existing low-confidence detections remain intact; don't add overlapping proposals.
        for existing in f['boxes'] + f['proposals']:
            if (existing['class_name'] == 'car'
                    and np.linalg.norm(np.array(existing['box'][:3]) - local[:3]) < cfg.base_gate_m):
                existing.setdefault('temporal_flags', []).append('TEMPORAL_SUPPORT_EXISTING_BOX')
                return
        count = support_count(target, local)
        flags = ['TEMPORAL_PROPOSAL_REQUIRES_REVIEW']
        estimated, pose_ok = req['estimated'], req['pose_ok']
        if estimated:
            flags.append('EGO_POSE_ESTIMATED')
        if not pose_ok:
            flags.append('REGISTRATION_QUALITY_FAILED')
        if not aligned:
            flags.append('EGO_MOTION_UNCOMPENSATED')
        if count < cfg.min_points:
            flags.append('LOW_POINT_SUPPORT')
        if kind != 'interpolated':
            flags.append('EXTRAPOLATION_UNCERTAIN')
        f['proposals'].append(dict(
            id=f'{track["track_id"]}_{kind}_{f["frame_id"]}', class_name='car',
            box=local.tolist(), score=None,  # no detector confidence exists for an inferred box
            temporal_quality_hint=req['temporal_quality_hint'],
            origin=kind, track_id=track['track_id'], point_count=count,
            source_frames=req['source_frames'], source_box_ids=req['source_box_ids'],
            qa_status='REVIEW', temporal_flags=flags, review_status='unreviewed',
            training_policy='ignore_region' if (aligned and count >= cfg.min_points and kind == 'interpolated'
                                               and pose_ok and (not estimated or cfg.allow_estimated_pose_for_ignore))
                            else 'review_only'))

    for req in requests:
        proposal(req)
    for track in tracks:
        track['observations'].sort()
    for f in output:
        for b in f['boxes']:
            b['temporal_flags'] = sorted(set(b['temporal_flags']))
    proposals = [p for f in output for p in f['proposals']]
    public_tracks = [dict(track_id=t['track_id'], sequence_id=t['sequence_id'],
                          observations=[dict(frame_id=frames[i]['frame_id'], box_id=frames[i]['boxes'][j]['id'])
                                        for i, j in t['observations']],
                          anchor_observations=[dict(frame_id=frames[i]['frame_id'], box_id=frames[i]['boxes'][j]['id'])
                                               for i, j in t['anchor_observations']]) for t in tracks]
    return dict(schema_version=1, mode=mode, ego_motion_compensated=mode == 'pose',
                settings=asdict(cfg), frames=output, tracks=public_tracks,
                summary=dict(frames=len(frames), tracks=len(tracks),
                             multi_frame_tracks=sum(len(t['observations']) > 1 for t in tracks),
                             proposals=len(proposals),
                             interpolated=sum(p['origin'] == 'interpolated' for p in proposals),
                             extrapolated=sum(p['origin'].startswith('extrapolated') for p in proposals),
                             low_point_support=sum('LOW_POINT_SUPPORT' in p['temporal_flags'] for p in proposals),
                             ignore_regions=sum(p['training_policy'] == 'ignore_region' for p in proposals),
                             supported_detections=sum(b['temporal_supported'] for f in output for b in f['boxes']),
                             training_performed=False, gold_validation=False))
