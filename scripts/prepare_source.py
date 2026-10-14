"""Prepare labeled nuScenes SOURCE for the folder wizard; never inspect target gold.

Creates a separate loader layout using symlinks to the raw dataset. The input
dataset is unchanged. CPU metadata processing; no training or CUDA kernels.
"""
import argparse
import os
from pathlib import Path
import sys


def source_layout(raw, output, version):
    raw, output = Path(raw).expanduser().resolve(), Path(output).expanduser().resolve()
    if raw == output or output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError("Folder prepared source phải tách khỏi folder dữ liệu gốc")
    if not (raw / version / "sample.json").is_file() or not (raw / "samples").is_dir():
        raise ValueError("Cần nuScenes raw: samples/ và " + version + "/sample.json")
    root = output / version
    root.mkdir(parents=True, exist_ok=True)
    links = [(version, raw / version), ("samples", raw / "samples")]
    for name in ("sweeps", "maps"):
        if (raw / name).is_dir():
            links.append((name, raw / name))
    for name, source in links:
        destination = root / name
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.exists() or destination.is_symlink():
            raise ValueError("Prepared source có đường dẫn khác: " + str(destination))
        destination.symlink_to(source, target_is_directory=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", choices=["v1.0-mini", "v1.0-trainval"], default="v1.0-mini")
    args = parser.parse_args()
    root = source_layout(args.input, args.output, args.version)
    infos = [root / f"nuscenes_infos_1sweeps_{split}.pkl" for split in ("train", "val")]
    if any(p.exists() for p in infos):
        raise ValueError("Infos đã tồn tại; dùng folder output mới để giữ bản cũ")
    runtime = Path(os.getenv("ANNOLOOP_ST3D_RUNTIME", Path(__file__).resolve().parents[1] / "vendor/ST3D"))
    sys.path.insert(0, str(runtime))
    from pcdet.datasets.nuscenes.nuscenes_dataset import create_nuscenes_info
    create_nuscenes_info(args.version, Path(args.output).expanduser().resolve(),
                         Path(args.output).expanduser().resolve(), max_sweeps=1)
    from annoloop.setup import inspect_source
    from annoloop.io import write
    result = inspect_source(args.output)
    write(root / "annoloop_source.json", {key: result[key] for key in ("version", "counts", "evidence", "scene_metadata")})
    print("Chọn folder source trong UI: " + str(Path(args.output).resolve()))
    print("Class và số nhãn: " + str(result["counts"]))


if __name__ == "__main__":
    main()
