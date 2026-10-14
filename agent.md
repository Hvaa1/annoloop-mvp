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
- Pure shape/track CVAT packages and backend API; UI delivery still disabled.

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
