# Third-party source

`vendor/ST3D` is derived from [CVMI-Lab/ST3D](https://github.com/CVMI-Lab/ST3D),
which builds on OpenPCDet. The original [Apache-2.0 license](vendor/ST3D/LICENSE)
and source notices are retained. Its source-only manifest is
[ANNOLOOP_SOURCE_MANIFEST.json](vendor/ST3D/ANNOLOOP_SOURCE_MANIFEST.json).

This snapshot includes local compatibility changes for Torch 2.0/CUDA/spconv,
NumPy and separate classification/IoU prediction output needed by AnnoLoop.
Native extensions are compiled locally; no compiled binaries are shipped.

Selected AnnoLoop helpers were adapted from the original local LiDAR demo. The
historical hashes/origins are recorded in [docs/REUSE.json](docs/REUSE.json);
those origins do not create a runtime dependency on that workspace.

Python dependencies are installed from their respective official distributions
and retain their own licenses. The optional setup downloads NVIDIA CUDA
redistributable components, preserving component license files under the local
ignored toolkit directory. No dataset, trained checkpoint or user database is
bundled in this repository.
