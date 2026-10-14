# AnnoLoop standalone repository

Read README.md, docs/USER_GUIDE.md and agent.md before changing this application.
Keep code portable: default paths must live in this checkout or use explicit
ANNOLOOP_* overrides. Never depend on the parent VIN_Ai workspace/demo.

- UI/controller use .venv; native models use .venv-gpu, Torch 2.0.0/CUDA 11.8.
  Do not downgrade user environments or copy native binaries across venvs.
- Do not start training, GPU verification or CVAT uploads during code checks.
- Every real training run needs a fresh folder and scripts/training_launcher.py
  registration; keep process receipts/logs. Save every 10 epochs plus final.
- GPU workers run sequentially; check actual process exit before model handoff.
- Never restart active workers just to apply UI/code changes. Preserve interrupted
  artifacts and immutable completed snapshots; explicit recovery uses a new attempt.
- Target gold is only for final evaluation, never calibration/selection/training.
- Folder/class onboarding generates project JSON/YAML; users should not need to
  author these. Check class support in real source/checkpoint metadata.
- Missing checkpoint/class may require source training; verified checkpoints that
  cover all requested classes skip source pretrain. Broad classes are not renamed
  into multiple narrower classes. Never invent class evidence or predictions.
- Keep per-project GPU/coordinate checks and calibration gates; never fabricate
  certificates or bypass minimum class samples. Signed ignores remain ignores.
- Memory IDs are independent of temporal track IDs. Preserve checkpoint evidence,
  context/hash binding, expiry/conflict rules, rollback and resume validation.
- Native training is source-assisted co-training, not a full ST3D++ reproduction.
- Final CVAT UI, tracking adaptation and 2D detector actions remain disabled until
  concretely implemented/tested. Do not present design mockups as measured output.
- Do not commit user data: data/, datasets/, databases, weights, point clouds,
  labels, logs, credentials or compiled artifacts. Respect .gitignore and audit
  git's actual staged files before push, including files under unexpected names.

Implementation checks: bash scripts/check.sh. See docs/VALIDATION.md for evidence.
