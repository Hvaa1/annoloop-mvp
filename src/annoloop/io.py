"""Atomic storage and file identity shared by services and workers."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


_DIGESTS = {}


def display_digest(path):
    """SHA256 reused while the file identity is unchanged; for page views only.

    Dispatch and upload paths must keep calling digest() directly.
    """
    path = Path(path).resolve()
    signature = file_signature(path)
    key = (str(path), signature)
    if key in _DIGESTS:
        return _DIGESTS[key]
    value = digest(path)
    # Like git's racy-index rule: a file touched within the timestamp granularity
    # could change again without a new signature, so it is never cached.
    if time.time_ns() - max(signature[3], signature[4]) > 2_000_000_000:
        if len(_DIGESTS) >= 256:
            _DIGESTS.clear()
        _DIGESTS[key] = value
    return value


def file_signature(path):
    """Cheap invalidation identity; callers still hash before sensitive actions."""
    stat=Path(path).stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Đường dẫn nằm ngoài dataset/project")
    return path


@contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
