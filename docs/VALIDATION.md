# Validation of the standalone export

Date: 15/10/2026. No target gold, real training or CVAT upload was used in these packaging checks.

- All **171 backend/UI tests passed**, including folder onboarding, class gates,
  calibration contracts, sequential handoff, immutable label memory, rejected
  rounds, rollback, interrupted-run recovery and saved sample display.
- Python compilation and shell syntax checks passed.
- Added portability tests for checkout-relative paths (including spaces), source
  layout links without changing raw inputs, complete CUDA source manifests and
  safe CUDA archive extraction.
- Standalone launcher smoke ran a print-only Python command on the host, recorded
  real worker/trainer PID identities, log and completed/exit-code-0 receipt.
- Git's actual tracked files were audited: source/docs only; no database, datasets,
  point clouds, labels, checkpoints, environments, logs or credentials included.
- Original local app/training workers and user storage were kept separate.

## Installation verification

- A local Git clone in a separate `/tmp` checkout created its own UI venv via
  the shipped bootstrap and pinned UI requirements; `pip check` found no broken
  requirements. No parent UI/model environment was reused.
- The clean clone also passed all **171 tests** in its freshly installed UI/test
  environment after dependency installation completed.
- Streamlit in that clone started on a separate port (8513), HTTP health returned
  200/`ok`, and AppTest rendered the initial empty-project page without exceptions.
- The three CUDA redistributables downloaded from NVIDIA passed SHA-256 checks;
  compiler/libcudacxx headers extracted correctly. No system driver changes.
- Both freshly installed UI and GPU environments passed `pip check` after
  separating optional UI dependencies from the backend package. Full dependency
  resolutions are pinned in constraints-ui.txt and constraints-gpu.txt.
- Added a regression test and fixed read-only license collisions during CUDA
  component merging. Also check Python development headers before installation.
- All five native CUDA extensions built successfully from source with the clean
  clone's Torch 2.0.0+cu118 environment and freshly downloaded CUDA 11.8 toolkit.
  CPU import preflight returned `ready: true` for trainer/dataset/model imports,
  compiled ops and NVRTC header configuration; `gpu_kernels_checked: false`.
- Native build used the host's existing GCC 11 / Python 3.10 development headers
  as the system toolchain. No parent Python/model environment or precompiled
  extension was reused. Ninja used two CPU compiler workers; no GPU model job ran.


- Real labeled nuScenes mini SOURCE was prepared into a separate temporary output:
  323 train / 81 validation frames, eight / two disjoint scenes. Raw inputs were
  linked, not modified or copied into Git. The helper completed and its inventories
  were read with NumPy 1.26; source setup tests also passed under UI NumPy 2.2.
- This caught/fixed missing SDK map links and the NumPy 1.26 `_core` compatibility
  stub. After the final fixes, all 16 source setup tests and six portability tests
  passed again. Generated source infos/JSON/maps remain outside the repository.

## Limits

Tests exercise CPU fixtures/contracts and Streamlit actions; they do not measure
model accuracy or prove GPU memory capacity on another machine. Native adapter
forward/backward is checked per project through the UI after selecting data.
Source-assisted memory/co-training remains a partial ST3D++ implementation.
Final CVAT delivery UI, temporal tracking adaptation and 2D detectors remain
unimplemented as described in README.md.
