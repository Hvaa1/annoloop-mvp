# Project handoff

Standalone private repository prepared from AnnoLoop MVP on 15/10/2026.
Original local app, active workers and user data are kept separately. This repo
starts with empty storage and contains application/runtime source code only.

## Current implementation

- Folder source/target/class wizard, generated source schema/inventory/YAML,
  checkpoint class compatibility and conditional source training.
- SECOND-IoU / PV-RCNN sequential GPU adapters, real source/target prediction,
  GPU forward/backward diagnostics with zero optimizer steps.
- Calibration from source validation; target adaptation on common pseudo-labels
  with source loss. Configurable rounds, epochs, manual/automatic controls.
- Versioned labels/checkpoints, early stops, atomic state, rollback and recovery
  from completed artifacts or saved Adam epoch checkpoints.
- Saved fixed-frame A/B sample viewer before/after each round.
- Guarded pseudo-label memory, distinct checkpoint confirmations, confidence/
  quality/IoU/point support, conflict ignores and finite retention/expiry.
- Explicit final inference/export and CVAT shape upload UI; saved accepted/ignore viewer.
- Model hub, lazy adapter binding, native source options and inference-only local ensemble.
- Six-step navigation and faster saved-preview/package reads; version comparison retained.

## Portable installation

Scripts/env.sh defines checkout-relative defaults. setup.sh creates isolated UI
and GPU environments, pins model dependencies, downloads verified CUDA compiler/
headers, builds vendor/ST3D and performs CPU-only import checks. prepare_source.sh
converts raw labeled nuScenes SOURCE into 1-sweep loader infos using a separate
output and links; inputs remain unchanged. No copied dataset or checkpoint.

vendor/ST3D is the verified adapted local source with Apache license/provenance,
including native CUDA compatibility and separate classification/IoU outputs.
Do not replace it with unpatched upstream when changing installation.

## Remaining product work

- Connect final full-dataset prediction/media normalization/export to CVAT UI.
- Bind temporal tracking to verified sequence/timestamp/ego pose in adaptation.
- Add detector/training adapters for 2D and additional source formats.
- Full ST3D++ DSNorm/EMA/curriculum and repeated-trial safeguards are separate work.
- Validate GPU kernel/model behavior on other GPU generations and large datasets.
- Resume restores saved epochs/Adam, not unsaved iterations/RNG order.

The exported code passed 171 backend/UI tests in a clean clone. The clean-clone UI
started and returned HTTP health 200/ok; launcher receipts were checked using a
print-only command, without training. Full installation/build evidence is in
docs/VALIDATION.md. A clone may open the UI after
setup; useful real model results still require the user's data and model training
or compatible verified checkpoints. Never claim clone alone includes those assets.

## 16/10/2026 — Synchronize current MVP

Merged newer local application modules, UI, catalog, docs and fixture tests.
Kept checkout-relative settings/runtime/setup, NumPy compatibility, portable shell
entry points, standalone source preparation, constraints, vendor runtime sources
and source-only Git exclusions. No local database/project/dataset/checkpoint,
logs, CVAT credentials or compiled binaries copied.

See FINAL_DELIVERY.md, MODEL_HUB.md, LOCAL_ENSEMBLE.md and UI_PERFORMANCE.md.
Final delivery is shape-only and explicit. Download/binding does not certify GPU
or generic adaptation readiness; ensemble is not N-model training. Cache reads
never replace full hash checks on upload. Existing snapshots/rollback preserved.

## 16/10/2026 — Standalone update validation

269 unique test scenarios checked in batches: 258 passed before stopping the
long-lived UI test process for runtime cost; 10 of the remaining11 passed in a
fresh process. The new-project test caught an existing standalone setup bug:
the class anchor loop overwrote the project name. Renamed the loop variable;
that case and related creation/portable installation checks then passed (8
checks). No unresolved failures; this is batch evidence, not a successful
completion of the initial monolithic run.

Python wheel built using existing GPU-venv setuptools (no GPU execution);
catalog JSON/YAML/license inclusion verified. UI venv lacks setuptools for
no-build-isolation packaging; no environment was modified. Syntax/whitespace
checks and source-only Git audit passed. No GPU model run, CVAT request or
cold installation on another machine was performed for this publish.
