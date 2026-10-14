"""Reject user data, database/weight/binary artifacts and credentials before push."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {"src", "ui", "scripts", "configs", "tests", "docs", "vendor", "resources", "examples"}
FORBIDDEN = {".db", ".db3", ".sqlite", ".sqlite3", ".sql", ".dump", ".rdb", ".aof", ".mdb",
             ".lmdb", ".pth", ".pt", ".ckpt", ".pkl", ".pickle", ".pcd", ".npy", ".npz", ".bin",
             ".bag", ".mcap", ".tfrecord", ".h5", ".hdf5", ".parquet", ".arrow", ".csv",
             ".so", ".o", ".a", ".log", ".pid", ".zip", ".deb", ".pem", ".key"}
SECRET = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def audit():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    problems = []
    count = size = 0
    for name in filter(None, paths):
        path = ROOT / name
        reason = None
        if len(Path(name).parts) > 1 and Path(name).parts[0] not in ALLOWED_ROOTS:
            reason = "folder outside source/documentation allowlist"
        elif path.suffix.lower() in FORBIDDEN or Path(name).parts[0] in ("data", "datasets", "database", "databases", "outputs", "runtime") or any(part in (".venv", ".venv-gpu", "__pycache__") for part in Path(name).parts):
            reason = "database/dataset/model/runtime artifact"
        elif path.is_symlink():
            reason = "tracked symlink may expose external data"
        elif not path.is_file():
            reason = "missing tracked file"
        elif path.stat().st_size > 5 * 1024 * 1024:
            reason = "large file, review explicitly"
        elif path.name.startswith(".env") and path.name != ".env.example":
            reason = "local environment settings"
        elif SECRET.search(path.read_bytes()):
            reason = "credential pattern detected"
        if reason:
            problems.append(name + ": " + reason)
        else:
            count += 1;size += path.stat().st_size
    if problems:
        raise SystemExit("Do not push:\n" + "\n".join(problems))
    print(f"Git audit passed: {count} source/documentation files, {size / 1024 / 1024:.2f} MiB; no database/dataset/weights/credentials detected.")


if __name__ == "__main__":
    audit()
