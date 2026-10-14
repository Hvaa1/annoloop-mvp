"""Standalone path/bootstrap/source-layout contracts; no model jobs or real data."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from annoloop.settings import ROOT, Settings


def script(name):
    spec = importlib.util.spec_from_file_location("portable_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortableTests(unittest.TestCase):
    def test_defaults_do_not_escape_checkout_or_require_demo(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.load()
        self.assertEqual(settings.python, ROOT / ".venv/bin/python")
        self.assertEqual(settings.trainer_python, ROOT / ".venv-gpu/bin/python")
        self.assertEqual(settings.training_launcher, ROOT / "scripts/training_launcher.py")
        self.assertEqual(settings.data, ROOT / "data")
        self.assertIsNone(settings.runtime_deps)

    def test_launcher_environment_follows_relocated_checkout(self):
        with tempfile.TemporaryDirectory(prefix="annoloop space ") as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            env_script = root / "scripts/env.sh"
            env_script.write_text((ROOT / "scripts/env.sh").read_text())
            env = {key: value for key, value in os.environ.items()
                   if key not in ("VINAI_PYTHON", "CUDA_HOME") and not key.startswith("ANNOLOOP_")}
            result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\\n" "$ANNOLOOP_ROOT" "$VINAI_PYTHON" "$ANNOLOOP_WORKER_PYTHON" "$ANNOLOOP_DATA_DIR"',
                                     "env-test", str(env_script)], env=env, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [str(root), str(root / ".venv/bin/python"),
                             str(root / ".venv-gpu/bin/python"), str(root / "data")])

    def test_source_layout_links_input_without_copying_or_overwriting_it(self):
        prepare = script("prepare_source")
        with tempfile.TemporaryDirectory() as temp:
            raw, output = Path(temp) / "raw", Path(temp) / "prepared"
            version = "v1.0-mini"
            (raw / version).mkdir(parents=True)
            (raw / version / "sample.json").write_text("[]")
            (raw / "samples").mkdir()
            (raw / "maps").mkdir()
            (raw / "maps/map.png").write_bytes(b"source map fixture")
            root = prepare.source_layout(raw, output, version)
            self.assertEqual((root / version).resolve(), raw / version)
            self.assertEqual((root / "samples").resolve(), raw / "samples")
            self.assertEqual((root / "maps").resolve(), raw / "maps")
            self.assertEqual((root / "maps/map.png").read_bytes(), b"source map fixture")
            self.assertEqual((raw / version / "sample.json").read_text(), "[]")
            self.assertEqual(prepare.source_layout(raw, output, version), root)
            with self.assertRaises(ValueError):
                prepare.source_layout(raw, raw / "prepared", version)
            with self.assertRaises(ValueError):
                prepare.source_layout(Path(temp) / "missing", output, version)

    def test_cuda_extraction_rejects_escape_and_preserves_internal_links(self):
        cuda = script("install_cuda")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, dest = root / "safe.tar", root / "extracted"
            with tarfile.open(archive, "w") as tar:
                member = tarfile.TarInfo("sdk/lib/libcudart.so.11");member.size = 2
                tar.addfile(member, io.BytesIO(b"ok"))
                link = tarfile.TarInfo("sdk/lib/libcudart.so")
                link.type = tarfile.SYMTYPE;link.linkname = "libcudart.so.11"
                tar.addfile(link)
            cuda.extract(archive, dest)
            self.assertEqual((dest / "sdk/lib/libcudart.so").read_bytes(), b"ok")
            for name, linkname in [("../outside", None), ("sdk/link", "../../outside")]:
                bad = root / "bad.tar"
                with tarfile.open(bad, "w") as tar:
                    member = tarfile.TarInfo(name)
                    if linkname:
                        member.type = tarfile.SYMTYPE;member.linkname = linkname
                    tar.addfile(member)
                with self.assertRaises(ValueError):
                    cuda.extract(bad, root / "bad-output")
            self.assertFalse((root / "outside").exists())

    def test_source_only_cuda_manifest_covers_all_five_extension_sources(self):
        manifest = json.loads((ROOT / "vendor/ST3D/ANNOLOOP_SOURCE_MANIFEST.json").read_text())
        paths = {row["path"] for row in manifest["files"]}
        for name in ("iou3d_nms", "roiaware_pool3d", "roipoint_pool3d", "pointnet2/pointnet2_stack", "pointnet2/pointnet2_batch"):
            self.assertTrue(any(p.startswith("pcdet/ops/" + name + "/src/") and p.endswith(".cu") for p in paths))
        self.assertFalse(any(p.endswith((".so", ".o", ".pth", ".pkl")) for p in paths))

    def test_cuda_component_readonly_licenses_are_preserved_separately(self):
        cuda = script("install_cuda")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);merged = root / "merged"
            for name in ("cuda_nvcc", "cuda_cudart"):
                source = root / name;source.mkdir()
                license_file = source / "LICENSE"
                license_file.write_text(name);license_file.chmod(0o444)
                (source / (name + ".txt")).write_text("component data")
                cuda.merge_component(name, source, merged)
                self.assertEqual((merged / "licenses" / name / "LICENSE").read_text(), name)
                self.assertEqual((merged / (name + ".txt")).read_text(), "component data")
            self.assertFalse((merged / "LICENSE").exists())
