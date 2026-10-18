"""Explicit binary point contracts; never read target labels or guess by size."""
from functools import lru_cache
from pathlib import Path
import re
from .io import digest, file_signature, read


def validate_features(names):
    if (not isinstance(names, list) or not 3 <= len(names) <= 16
        or any(not isinstance(name, str) or not name or name != name.strip() for name in names)
        or len(set(names)) != len(names) or not {'x', 'y', 'z'} <= set(names)):
        raise ValueError('feature_names cần tên cột duy nhất, có x, y, z')
    return list(names)


@lru_cache(maxsize=16)
def _descriptor(path, signature):
    if signature[2] > 4 * 1024 * 1024:
        raise ValueError('pointcloud_format.json quá lớn')
    document = read(path)
    if document.get('schema_version') != '1.0' or document.get('dtype') != 'float32_le':
        raise ValueError('Point format cần schema_version 1.0 và dtype float32_le')
    names = validate_features(document.get('feature_names'))
    files = document.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Point format cần chỉ mục file có SHA256')
    identity=digest(path)
    if file_signature(path) != signature:
        raise ValueError('Point format thay đổi trong lúc đọc')
    return names, files, identity


def binary_contract(frame):
    """Return declared features/evidence or None for unknown BIN; no inference."""
    path = Path(frame['media']).resolve()
    declared = frame.get('feature_names')
    explicit=None
    if declared is not None:
        explicit = validate_features(declared)
        if path.suffix.lower() == '.bin' and path.stat().st_size % (4 * len(explicit)):
            raise ValueError('BIN không khớp số cột đã khai báo')
    if path.suffix.lower() != '.bin':
        return dict(feature_names=explicit) if explicit else None
    # Bounded search accommodates training/velodyne and samples/LIDAR_TOP.
    for parent in list(path.parents)[:8]:
        descriptor = parent / 'pointcloud_format.json'
        if not descriptor.is_file():
            continue
        names, files, identity = _descriptor(str(descriptor), file_signature(descriptor))
        if explicit is not None and names!=explicit:
            raise ValueError('feature_names không khớp point format của dataset')
        row = files.get(path.relative_to(parent).as_posix())
        if not isinstance(row, dict) or not re.fullmatch('[a-f0-9]{64}', str(row.get('sha256', ''))):
            raise ValueError('BIN chưa có bằng chứng SHA256 trong point format')
        if type(row.get('points')) is not int or row['points'] <= 0:
            raise ValueError('Point format thiếu số điểm hợp lệ')
        if path.stat().st_size != row['points'] * len(names) * 4:
            raise ValueError('BIN không khớp shape point format')
        if row['sha256'] != frame.get('sha256', None) and frame.get('sha256') is not None:
            raise ValueError('SHA256 BIN không khớp point format')
        if frame.get('sha256') is None and row['sha256'] != digest(path):
            raise ValueError('SHA256 BIN không khớp point format')
        return dict(feature_names=list(names), point_format_evidence=dict(path=str(descriptor), sha256=identity))
    return dict(feature_names=explicit) if explicit else None


def verification_contract(dataset_sha256, config, frames):
    """Bind GPU verification to column semantics as well as media identities."""
    result=dict(dataset_sha256=dataset_sha256,bin_columns=config.get('bin_columns'),
                coordinates_confirmed=config.get('coordinates_confirmed'))
    if config.get('mmdet_input_mode'):result['mmdet_input_mode']=config['mmdet_input_mode']
    formats={}
    for frame in frames:
        if Path(frame['media']).suffix.lower()!='.bin':continue
        contract=binary_contract(frame)
        if not contract:continue
        if result['bin_columns'] is not None and len(contract['feature_names'])!=result['bin_columns']:
            raise ValueError('Số cột BIN đã chọn không khớp khai báo dataset: '+str(len(contract['feature_names'])))
        evidence=frame.get('point_format_evidence',contract.get('point_format_evidence'))
        if evidence and evidence['path'] not in formats:
            if digest(evidence['path'])!=evidence['sha256']:
                raise ValueError('Khai báo point format đầu vào đã thay đổi')
            formats[evidence['path']]=evidence['sha256']
    if formats:result['input_formats']=formats
    return result
