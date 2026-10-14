"""Install only the CUDA 11.8 compiler/headers needed by the vendored runtime.

Official NVIDIA redistributables, checksum-verified. No sudo or driver changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = {
    "cuda_nvcc": ("11.8.89", "7ee8450dbcc16e9fe5d2a7b567d6dec220c5894a94ac6640459e06231e3b39a5"),
    "cuda_cudart": ("11.8.89", "56129e0c42df03ecb50a7bb23fc3285fa39af1a818f8826b183cf793529098bb"),
    "cuda_cccl": ("11.8.89", "99d77d9e4c75d5e4663e473577f1871e65bca4ea0b9023f544a3556f0c1776c7"),
}


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(stream)


def _hash(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def extract(archive, destination):
    """Allow vendor symlinks only when their resolved target stays in the archive."""
    destination = Path(destination).resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = destination / member.name
            if not path.resolve().is_relative_to(destination):
                raise ValueError("Unsafe archive path: " + member.name)
            if member.isdev() or member.isfifo():
                raise ValueError("Unsupported archive member")
            if member.issym() or member.islnk():
                link = (path.parent if member.issym() else destination) / member.linkname
                if not link.resolve().is_relative_to(destination):
                    raise ValueError("Unsafe archive link: " + member.name)
        # Validate again after preceding links are created, preventing link traversal.
        for member in tar.getmembers():
            if not (destination / member.name).resolve().is_relative_to(destination):
                raise ValueError("Archive link traversal")
            tar.extract(member, destination)


def merge_component(component, source, merged):
    """NVIDIA licenses may be read-only; preserve them without overwriting one."""
    for license_file in source.glob("LICENSE*"):
        licenses = merged / "licenses" / component
        licenses.mkdir(parents=True, exist_ok=True)
        shutil.copy2(license_file, licenses / license_file.name)
    shutil.copytree(source, merged, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns("LICENSE*"))


def install(output):
    output = Path(output).expanduser().absolute()
    if (output / "bin/nvcc").is_file() and (output / "include/cuda/std/cassert").is_file():
        print("CUDA toolkit đã có: " + str(output))
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Toolkit có dữ liệu nhưng chưa đầy đủ; chọn folder trống: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "downloads"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cuda-setup-", dir=output.parent) as temp:
        temp = Path(temp)
        merged = temp / "toolkit"
        merged.mkdir()
        receipt = {}
        for component, (version, expected) in COMPONENTS.items():
            filename = f"{component}-linux-x86_64-{version}-archive.tar.xz"
            relative = f"{component}/linux-x86_64/{filename}"
            archive = cache / filename
            if not archive.is_file() or checksum(archive) != expected:
                partial = archive.with_suffix(".part")
                print("Downloading " + component, flush=True)
                with urlopen(BASE + relative, timeout=60) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if checksum(partial) != expected:
                    raise ValueError("CUDA download checksum mismatch: " + component)
                partial.replace(archive)
            extracted = temp / component
            extracted.mkdir()
            extract(archive, extracted)
            roots = list(extracted.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError("Unexpected CUDA archive layout")
            merge_component(component, roots[0], merged)
            receipt[component] = dict(url=BASE + relative, sha256=expected, version=version)
        if not (merged / "include/cuda/std/cassert").is_file():
            raise ValueError("Missing libcudacxx headers after extraction")
        (merged / "annoloop_cuda_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if output.exists():
            output.rmdir()  # Only an empty destination is allowed above.
        os.replace(merged, output)
    print("CUDA compiler/headers ready: " + str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    install(parser.parse_args().output)
