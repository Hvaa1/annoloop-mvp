"""Environment-configurable paths; no Streamlit session state or Docker assumptions."""
from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(os.getenv("ANNOLOOP_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
# Folder picker starts at this standalone checkout, never at its parent's workspace.
WORKSPACE = ROOT


@dataclass(frozen=True)
class Settings:
    data: Path
    python: Path
    training_launcher: Path
    worker_python: Path | None = None
    runtime_deps: Path | None = None

    @property
    def trainer_python(self):
        return self.worker_python or self.python

    @classmethod
    def load(cls):
        # Do not resolve Python symlinks: that would bypass the venv.
        return cls(
            Path(os.getenv("ANNOLOOP_DATA_DIR", ROOT / "data")).expanduser().absolute(),
            Path(os.getenv("VINAI_PYTHON", ROOT / ".venv/bin/python")).expanduser().absolute(),
            Path(os.getenv("ANNOLOOP_TRAINING_LAUNCHER",
                 ROOT / "scripts/training_launcher.py")).expanduser().absolute(),
            Path(os.getenv("ANNOLOOP_WORKER_PYTHON", ROOT / ".venv-gpu/bin/python")).expanduser().absolute(),
            Path(os.environ["ANNOLOOP_RUNTIME_DEPS"]).expanduser().absolute()
                if os.environ.get("ANNOLOOP_RUNTIME_DEPS") else None,
        )
