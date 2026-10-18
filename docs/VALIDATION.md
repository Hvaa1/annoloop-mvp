# Kiểm tra backend — 13/10/2026

Các mục đầu ghi baseline; kết quả mới nhất nằm ở cuối file.

Đã dùng venv VIN_Ai `.venv/bin/python`; không tạo/thay venv.

| Kiểm tra | Kết quả |
|---|---|
| Unit/controller/device contracts | 44 tests pass |
| Python compileall | Pass |
| Bash syntax của hai launcher | Pass |
| CLI help | Pass |
| CLI doctor | torch 2.5.1+cu124, CUDA 12.4, available=true, 1 device |

Tests chỉ dùng fixture tạm. Không ghi dataset giả vào project thật, không launch
trainer, không tạo prediction thật, không upload CVAT. Fixtures của controller
chỉ kiểm chứng orchestration/receipt; không kiểm chứng model quality.

Đã kiểm tra: class theo schema, no hardcoded car; broad mapping bị chặn;
class source thực trong infos; holdout theo sequence; immutable snapshots và
checkpoint copies; source-only calibration; không match khác class; ignore
khi một detector không đủ confidence (luật calibrated_agreement_v2: cả hai
confidence ≥ ngưỡng, hình học lấy box thật có quality cao hơn); tracking per class; shape/track riêng; final
checkpoint không tăng epochs; A rồi B dùng cùng nhãn; manual/automatic modes.

Chưa kiểm chứng: CUDA compiled ops ST3D tương thích venv; source training thật;
native adapter inference/train-target; CVAT media/upload/read-back trên server
thật; source dataset converters; resume worker; final prediction full dataset.
Đây là các bước integration còn lại, không được coi là tính năng đã chạy thật.

Lệnh chạy lại:

```bash
bash buildphase/annoloop_mvp/scripts/check.sh
bash buildphase/annoloop_mvp/scripts/annoloop.sh doctor
```

## UI Streamlit — 13/10/2026

- 44 backend tests + 6 UI/selection tests pass bằng VIN_Ai `.venv`.
- AppTest đi qua 8 trang, dataset sai hiển thị lỗi có thể sửa, tạo project subset
  class, xem snapshot hai vòng và rollback giữ nguyên artifacts.
- Kiểm tra submit source launcher đúng một lần; refresh/chuyển trang không
  gọi lại launcher. Test dùng mock launcher, không chạy trainer.
- Compile Python và Bash syntax pass; Streamlit localhost:8502 health trả `ok`.
- Test fixtures nằm trong temporary directories; không sinh model outputs thật.
- Chưa kiểm chứng native GPU worker/CVAT; adaptation/final upload bị khóa trong UI.

- Đã xem screenshot UI thật bằng Firefox headless; ảnh ở `screenshots/overview.png`.
  Font hệ thống hoạt động offline; ảnh chụp trạng thái chưa có dataset, không có số liệu giả.

## Folder workflow — 13/10/2026

63 tests pass: backend/controller/device, 11 folder-setup tests, 6 UI tests.
- Read source labels from prepared nuScenes; NumPy-only pickle allowlist rejects
  executable globals. Split checks include frame and scene when metadata exists.
- Raw target without schema works through user class selection; target gold not
  parsed and input folders unchanged. Project stores generated schema/inventory.
- Ready checkpoint and requested subset never trigger source training. If one
  detector lacks a class, only that detector gets a job; two jobs run sequentially.
- Missing source labels block training; valid source train geometry can provide
  anchor medians for a new class. Never estimates anchors from target gold.
- Completed checkpoints promoted only after receipt completes and PID stops;
  active receipt/checkpoint cannot be reported as completed.
- UI has folder picker, source/target/class wizard, ready-vs-train plans; no
  manual JSON/YAML fields. Reruns/navigation never create an extra trainer.
No real trainer/GPU model/CVAT request was started by these checks. Source adapter
currently supports prepared nuScenes; arbitrary/raw source imports remain work.

## Runtime/preflight/retry — 13/10/2026

- 68 tests pass, including model-Python command/receipt, preflight before run
  allocation, duplicate submission, and retry only after controller/child exit.
- easydict 1.13 and tensorboardX 2.6.5 present in VIN_Ai; pip check reports no
  broken dependencies. No Torch/NumPy version changed.
- Existing ST3D extensions cannot import under VIN_Ai Torch 2.5.1/cu124
  (libcudart.so.11.0 missing). Compatible model venv Torch 2.0/cu118 plus
  st3d_runtime dependency path passes all source trainer/compiled-op imports.
- Real preflight with Settings.load passed. This is an import check, not a
  GPU-kernel/optimizer training verification. No trainer restarted during repair.
- Failed job 639c178ec92f42c386dbfa74175b1832 and source pair preserved; project
  failed status refreshed only after actual registered processes stopped.

## CUDA headers and GPU diagnostic — 13/10/2026

72 tests pass. Tests cover selected toolkit path, missing libcudacxx header
failure before dispatch, and hashed NVRTC library alias outside the Torch wheel.
GPU diagnostic under the model venv: SubMConv3d(3,16,kernel=3) forward/backward
and dense cuDNN Conv2d forward/backward passed; gradients finite, optimizer steps
0, peak allocated tensor memory 68,096 bytes. NVRTC compiled fallback algorithms
for the GPU during this check. No dataset/model optimizer run was launched.
This verifies these kernels/runtime paths, not all model layers or training VRAM.

## Checkpoint inspection — 13/10/2026

80 tests pass, including TP/FP/FN one-to-one matching, empty predictions/GT,
explicit BIN layout, intermediate checkpoint discovery, coordinate gate, dead PID
status and UI eval action without trainer launch. Actual native source validation
ran 81 frames in ~38 seconds after model load; saved result.json and per-frame
predictions under project inspections/3238a1fa92334d7d85b67846d3a03898.
Metrics at score .3 / 3D IoU .5: car P .28336/R .07785; pedestrian P .13636/R
.02153. These are local fixed-threshold box metrics, not nuScenes mAP.
Target PCD loader confirmed finite XYZ; target sample requires axes/units confirmation.
No train/resume performed; stopped run and checkpoint epoch 10 preserved.

Native target inference adapter also passed a GPU smoke check on one raw source
LiDAR scan with known axes, loaded as unlabelled XYZ NPY: 6 predicted boxes,
point preview and JSON saved in data/diagnostics/target_adapter_smoke_7f1b554d.
This verifies the no-GT inference path; it is not a Day14 target result.
Actual Day14 prediction remains gated by the user's units/axes confirmation.


## Native adaptation binding và project UX — 13/10/2026, 23h

93 tests pass (VIN_Ai venv); no training or CVAT requests in the software suite.

Software checks cover native dispatch gates (both models/coordinates/hash/GPU
verification), source evidence and BIN layout invalidation, signed multiclass
labels, cross-class dense ignore while preserving positives, orphaned workers,
registered target worker command and required trained receipt, project settings,
explicit UI verify/run and no refresh dispatch. Source subset-B dispatch/refresh
is also covered: never misidentify a B-only job as A.

Real GTX 960M diagnostic using SECOND-IoU epoch 10:
`data/diagnostics/native_adapter_backward_ba549cbd/verified.json`.
Native cls/IoU scores, source inference, unlabelled raw source scan through target
inference (34 boxes), source backward and target DatasetTemplate backward passed;
finite gradients, optimizer steps **0**, no weights saved/updated, no target gold.
This uses a known-axis source scan for diagnostics; **not Day14 inference/training**.
No PV-RCNN car/pedestrian checkpoint exists in this project, so B verification and
real full two-model target training/rounds were not executed. The UI remains
blocked for these concrete missing inputs instead of an unimplemented-adapter
warning. Streamlit health checked on localhost:8502; GPU idle after diagnostics.

## Live Streamlit backend reload fix — 13/10/2026, 23h14

Live server reported AttributeError: old imported ProjectStore had no configure.
Current source contains the method. App now includes backend src in PYTHONPATH
for Streamlit watchers; launcher defaults to polling. Verified UI process alone
was restarted. 12 UI tests pass (save settings and direct-launch source watching
included); compile/Bash syntax and restarted localhost:8502 health pass. No
training run launched or interrupted, no project dataset/snapshot rewritten.

## Early-stop source handoff — 14/10/2026

101 tests pass. New cases verify: stopped A waits for PID exit before B, GPU busy
blocks B, failed/interrupted A never advances, whole-sequence cancellation skips B,
stopped B does not restart anything, duplicate stop sends one SIGTERM, completed
B metadata is promoted while stopped A remains intermediate. Artifacts remain.
Focused pair tests (6) and compileall passed after the final dispatch cancellation
guard. Launches/PIDs were mocked; no real trainer started or interrupted in checks.

## Model identity and log display — 14/10/2026

102 tests pass. UI regression test uses MODEL.NAME=PVRCNN in a file named
A_source.yaml and tqdm ESC[A in the log; monitor must identify B and hide ANSI.
Actual running run 416edf89e1874e8b917a9bae32ed49c8 checked as PVRCNN B with live
PIDs. Its raw log/receipt/config and trainer kept unchanged. Compileall pass.

## GPU verify refresh, finish request và retry vòng — 14/10/2026

110 tests pass; compileall pass. Tests cover explicit fresh-attempt retry,
preservation of failed artifacts, live/orphan worker guards in nested attempts,
immutable round work-directory lookup, full-page refresh after background checks,
clearing a finish request without dispatch, and explicit UI retry exactly once.
Refresh/navigation never starts training. Tests use temporary fixtures/mocks.

Read-only actual-host check for project Test `60fdb405fa904512`:
readiness=true, GPU verified for A epoch 10/B epoch 20, no active adaptation job,
GPU compute idle, localhost:8502 health ok. Verification job
`b94d237fb4b64063b14c086fe386c596` completed both native checks at
02:56:45 UTC with 0 optimizer steps. No certificate was generated by these checks.
The previous M1 was interrupted during target A inference, with 7 raw frame
outputs preserved; no completed two-model adaptation round is claimed.

Real-project AppTest renders without errors and shows enabled “Cho phép chạy
tiếp” because stop_after_round=true. The stop request remains unchanged. UI test
then covers clearing the request and dispatching “Chạy lại vòng M1” explicitly.
Hashes of real project.json, native_verified.json and legacy work/M1/state.json
were identical before/after read-only rendering. No actual training or CVAT job
was started, stopped or resumed for this repair.

## Thiếu mẫu calibration và chọn subset class — 14/10/2026

118 tests pass; compileall pass. After the last Streamlit widget-state adjustment,
all 14 UI tests pass again. Tests verify both source calibrations precede target
prediction, insufficient samples prevent any target predict/train, rare-class
diagnostics stay source-only, all-zero overlaps do not produce high calibrated
confidence, and reports are ignored for mismatched weights/config/runtime/source
infos or partial scans. Scope selection reuses the existing GPU verification;
empty/unknown/duplicate scopes and changing scope after a snapshot are rejected.
End-to-end fixture checks preserve motorcycle class ID 3 as signed ignore -3,
never a positive label, and leave schema/model class ordering unchanged.

Actual failed attempt `3774a49e91f64423b262a6838bcb8699` completed A source
validation on 81 frames: car 2030 predictions, pedestrian 1392, motorcycle 5.
All 5 motorcycle source IoUs are zero (eval GT motorcycle 232). No target gold
was involved; minimum calibration sample count remains 10 per active class.

After the user explicitly selected car + pedestrian, settings were saved with
that adaptation subset; source loss/config/checkpoint retain all original three
classes. The original setup was backed up under project/setup_history and the
native GPU certificate stayed byte-for-byte unchanged. Real requested M1 job
`fb09f62816c04f439802d7bd8af32e47` was dispatched at 03:37:20 UTC (10:37 local).
PID 13367 verified alive on the host; latest observed stage source calibration A,
not yet target optimizer training. This requested run is separate from software
checks; no completed adaptation round or quality improvement is claimed.
Read-only actual-project AppTest shows car/pedestrian selected, zero UI errors,
and zero additional submissions. Failed job artifacts remain intact.

## Sample sau mỗi M và preview vòng bị ngắt — 14/10/2026

127 tests pass; compileall pass. Added contracts cover partial post-train target
outputs, matching frame/checkpoint/config/dataset provenance, missing predictions
versus empty detections, rejection of gold-bearing/wrong-frame documents, unchanged
project snapshots, separate M1/M2 checkpoint links, and a full-page refresh when
a new round completes while the five-second preview fragment is open. UI test
renders four before/after A/B plots without a completed snapshot, then three
when a selected frame is absent in B after-predict. No GPU job is dispatched.

Read-only actual project preview: M1 attempt fb09f62816c04f439802d7bd8af32e47,
both trainers completed 3 epochs / 468 optimizer steps. A after-predict is 198/198;
B after-predict is 12/198 without result.json. Host has no live controller/trainer;
stop_after_round remains true, completed_rounds remains empty. A later retry
ded588065e014820a0ee63acc5fbb872 only has pre-train outputs, preserved separately.
Viewer defaults to the latest attempt with post-train predictions, so M1's
trained-model samples are visible rather than hidden by that incomplete retry.

Actual first fixed sample loads 300,820 PCD points; 12,000 displayed in each
BEV view. AppTest renders four plots with zero exceptions/errors/warnings.
Point reader pypcd4 1.5.1 installed in VIN_Ai venv using the declared [pcd]
dependency; Torch 2.5.1+cu124 and NumPy 2.2.6 unchanged. Model venv untouched.
No inference/training was started or resumed, and no snapshot was fabricated.

## Explicit interrupted-round recovery — 14/10/2026

139 tests pass; Python compileall and launcher bash syntax pass. New contracts
cover immutable old attempt files, dataset/source/model/config/class identity,
shared labels bound to before-prediction cache, rejecting a live old worker,
skipping completed A/B or completed A only, latest target checkpoint selection,
repeated interruption fallback, and restoring Adam moments/epoch/iteration using
a temporary CPU checkpoint. The remaining epoch budget is preserved.

Inference cache tests verify real saved frame documents, missing-frame handling,
checkpoint mismatch rejection, file/point hashes, and zero GPU subprocesses when
all target frames are already saved. UI tests dispatch explicit resume once and
do not resubmit on refresh; retry is a separate command. Worker finishes one
resumed round before returning, so automatic mode does not launch another round.

Actual project artifacts were read only: fb09f628… completed A/B 3 epochs and
468 steps each; A post-predict198/B12 on198 target frames. ebacbae9… completed A;
B's237 unsaved steps cannot be resumed because no target checkpoint exists.
The earlier ded58806… attempt remains preserved. No completed round snapshot yet.
Actual app health on localhost:8502 is OK, PID9927 alive; no real model/controller
worker running at this check. No new GPU train/predict job submitted.

This verifies recovery contracts and CPU optimizer restoration, not real GPU
end-to-end recovery. Unsaved iteration progress and RNG/augmentation state cannot
be recovered. Resuming the existing fb09f628… attempt requires the user's explicit
"Tiếp tục vòng M1" action and should reuse completed trainers, A198 and B12 outputs.

## Automatic resume and selection explanation — 14/10/2026

142 tests pass; compileall/bash syntax pass. Controller test resumes completed A/B
in M1, accepts it automatically, then runs M2 using M1's selected checkpoints and
fresh A/B training. The recovery plan is cleared before M2. An automatic first
round with agreement0.00525 below0.6 stops with reject and no M2 training. UI test
shows0.53% versus60%, rejects automatic selection, allows explicit manual override,
and dispatches zero jobs while viewing/selecting snapshots. New round receipts
include the policy used; legacy UI clearly labels current policy fallback.

The user's real resume completed M1 in7d4d26c1…: A198 reused/0 new forwards;
B12 reused/186 new forwards. Saved decision is reject, agreement_below_threshold,
34 labels, agreement0.005250965250965251 versus configured minimum0.6. Project then
has M1 selected manually and automatic M2 started by the user. Original thresholds,
snapshots, model checkpoints and selection are preserved by this repair.

Host check: M2 controller14399 and A native trainer15116 alive, launcher15115,
GPU1752MiB; app8502 health OK. No real job launched/stopped by implementation
checks. Real inference-only recovery is now evidenced by the user's saved run;
GPU optimizer-state recovery remains untested. This change supersedes the earlier
one-round-only resume rule; code already loaded by the active worker remains in use.

## Pseudo-label memory across model rounds

165 tests pass; compileall/bash syntax pass. New CPU contracts cover stable
per-frame IDs across native IDs, distinct checkpoint confirmations, score/IoU
history, finite retention, ignore/expiration/reactivation, actual point support,
class conflicts, current weak/partial evidence, score regression, immutable
updates and rejection of gold/invalid history. Signed ignores cannot suppress
retained positives of the same class.

Controller tests retain a stable missing label but keep agreement0 and reject
the round under unchanged thresholds. Rejected-round memory is persisted, while
the next trainer still receives the selected version's label SHA. Memory can be
disabled. Snapshot/store tests bind memory hash and positive geometry to labels,
reject mutation on selection/loading, and replay legacy machine labels without
opening a target gold fixture or rewriting earlier snapshots. Resume binds its
pre-round memory and rejects changed policy. UI test renders M0/M1 confidence,
quality and historical IoU with no model submission.

Actual existing-project read-only replay used198 point-cloud frames and saved
M0→M3 machine-label artifacts. Memory304 objects: observed19, retained0, ignored28,
expired257, stable11. All input snapshot hashes unchanged; no new project files,
GPU inference, training or labels generated/promoted by the check. This is a CPU
reconstruction test, not evidence of better detector accuracy.

The active controller14399 still ran M4 source calibrationB with GPU worker25061,
984MiB; appPID9927/8502 health OK. That process retains old code; new memory is
available on subsequent explicit controller launches. Native GPU training with
new memory-generated selected labels has not been integration-tested yet.
