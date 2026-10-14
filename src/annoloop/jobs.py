"""Persistent pretrain jobs registered with the workspace training monitor."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from .device import gpu_idle, probe
from .io import lock, now, read, write
from .reused.training_monitor import process_identity, read_log
from .settings import ROOT, Settings


def model_identity(spec):
    """Use architecture from the YAML, never a progress-bar escape or run ID."""
    import yaml
    path = Path(spec.get('config', ''))
    document = yaml.safe_load(path.read_text()) if path.is_file() else {}
    architecture = document.get('MODEL', {}).get('NAME') if isinstance(document, dict) else None
    detector = {'SECONDNetIoU': 'A', 'PVRCNN': 'B'}.get(architecture)
    return dict(architecture=architecture, detector=detector,
                label=f'{detector} · {architecture}' if detector else architecture or 'Chưa xác định model')


def start_pretrain(spec, settings=None):
    settings = settings or Settings.load()
    from .runtime import require_runtime, worker_environment
    required = ("config", "runtime", "source_train_annotations", "source_validation_annotations")
    for key in required:
        if not spec.get(key) or not Path(spec[key]).exists():
            raise ValueError("Thiếu tài nguyên pretrain: " + key)
    runtime_check = require_runtime(spec, settings)
    job_id = uuid.uuid4().hex
    root = settings.data / "jobs" / job_id
    root.mkdir(parents=True)
    identity = model_identity(spec)
    write(root / "job.json", dict(id=job_id, kind="source_pretrain", status="planned", spec=spec,
                                  model=identity, python=str(settings.trainer_python), runtime_check=runtime_check, created_at=now()))
    environment = worker_environment(settings)
    command = [str(settings.python), str(settings.training_launcher), "--run", str(root),
               "--cwd", str(ROOT), "--name", "AnnoLoop source · " + identity['label'], "--",
               str(settings.trainer_python), "-m", "annoloop.jobs", "--worker", str(root)]
    try:
        subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    except Exception as error:
        document = read(root / "job.json")
        document.update(status="launch_failed", error=str(error))
        write(root / "job.json", document)
        raise
    return str(root)


def status(root):
    root = Path(root)
    receipt = read(root / "training.json") if (root / "training.json").exists() else {}
    active = any(process_identity(receipt.get(key)) == receipt.get(identity) and receipt.get(identity) is not None
                 for key, identity in (("pid", "process_start"), ("trainer_pid", "trainer_process_start"))
                 if receipt.get(key))
    if receipt.get("status") == "running" and not active:
        receipt = {**receipt, "status": "interrupted"}
    job = read(root / 'job.json')
    return dict(job=job, model=job.get('model') or model_identity(job.get('spec', {})), training=receipt, process_alive=active,
                log=read_log(root / "train.log"))


def stop(root):
    root = Path(root)
    with lock(root / '.stop.lock'):
        receipt = read(root / "training.json")
        if not receipt.get('process_start') or process_identity(receipt.get('pid')) != receipt['process_start']:
            return False
        request = root / 'stop_request.json'
        previous = read(request) if request.exists() else {}
        if previous.get('process_start') == receipt['process_start']:
            return True
        try:
            os.kill(receipt['pid'], signal.SIGTERM)
        except ProcessLookupError:
            return False
        write(request, dict(pid=receipt['pid'], process_start=receipt['process_start'], requested_at=now()))
        return True


def start_source_pair(specs, settings=None):
    """One controller; two independently registered training runs, in A then B order."""
    settings = settings or Settings.load()
    if set(specs) != {"A", "B"}:
        raise ValueError("Cần source spec của A và B")
    root = settings.data / "source_pairs" / uuid.uuid4().hex
    root.mkdir(parents=True)
    write(root / "pair.json", dict(specs=specs, jobs={}, outcomes={}, status="starting", stop_requested=False))
    environment = os.environ.copy()
    environment.update(ANNOLOOP_DATA_DIR=str(settings.data), VINAI_PYTHON=str(settings.python),
                       ANNOLOOP_TRAINING_LAUNCHER=str(settings.training_launcher),
                       ANNOLOOP_WORKER_PYTHON=str(settings.trainer_python))
    if settings.runtime_deps:environment['ANNOLOOP_RUNTIME_DEPS']=str(settings.runtime_deps)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    with (root / "controller.log").open("w") as stream:
        process = subprocess.Popen([str(settings.python), "-m", "annoloop.jobs", "--pair", str(root)],
                                   cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    write(root / "controller.json", dict(pid=process.pid, process_start=process_identity(process.pid)))
    return str(root)


def pair_for_run(run, settings=None):
    settings = settings or Settings.load()
    run = Path(run).resolve()
    for path in settings.data.glob('source_pairs/*/pair.json'):
        document = read(path)
        for key, directory in document.get('jobs', {}).items():
            if Path(directory).resolve() == run:
                return dict(path=str(path.parent), detector=key, pair=document)
    return None


def stop_source_pair(root):
    """Cancel the whole sequence; unlike stop(A), this must not start B."""
    root = Path(root)
    with lock(root / '.pair_state.lock'):
        document = read(root / 'pair.json')
        if document.get('status') not in ('starting', 'running'):
            return False
        document.update(stop_requested=True)
        write(root / 'pair.json', document)
        active = document.get('jobs', {}).get(document.get('active_detector'))
    if active:
        stop(active)
    return True


def update_pair(root, document):
    # Preserve cancellation set by the UI while the controller was waiting.
    with lock(root / '.pair_state.lock'):
        current = read(root / 'pair.json')
        document['stop_requested'] = current.get('stop_requested', False)
        write(root / 'pair.json', document)


def pair_worker(root):
    root = Path(root)
    document = read(root / "pair.json")
    document.setdefault('outcomes', {})
    try:
        for key in ("A", "B"):
            if read(root / 'pair.json').get('stop_requested'):
                document.update(status='cancelled', active_detector=None)
                update_pair(root, document)
                return
            document.update(status="running", active_detector=key)
            update_pair(root, document)
            if read(root / 'pair.json').get('stop_requested'):
                document.update(status='cancelled', active_detector=None)
                update_pair(root, document)
                return
            directory = start_pretrain(document["specs"][key])
            document["jobs"][key] = directory
            update_pair(root, document)
            began = time.monotonic()
            stop_sent = False
            while True:
                snapshot = status(directory)
                state = snapshot["training"].get("status")
                cancelled = read(root / 'pair.json').get('stop_requested', False)
                if cancelled and snapshot['process_alive'] and not stop_sent:
                    stop_sent = stop(directory)
                if state in ('completed', 'stopped'):
                    if snapshot['process_alive']:
                        time.sleep(.5)
                        continue
                    if state == 'completed' and not (Path(directory) / 'checkpoint.metadata.json').is_file():
                        raise RuntimeError('Worker thiếu checkpoint metadata cuối')
                    # A stopped run remains intermediate, but no longer blocks B.
                    document['outcomes'][key] = dict(status=state, run=directory,
                        checkpoints=[str(p) for p in sorted((Path(directory) / 'checkpoints').glob('*.pth'))])
                    update_pair(root, document)
                    break
                if state in ("failed", "interrupted"):
                    raise RuntimeError(f"Detector {key}: {state}; giữ artifacts {directory}")
                if state == "starting" and time.monotonic() - began > 30:
                    raise RuntimeError("Training worker không đăng ký PID trong 30 giây")
                time.sleep(1)
            gpu_idle()
        terminal = 'completed_with_stops' if any(v['status'] == 'stopped' for v in document['outcomes'].values()) else 'completed'
        if read(root / 'pair.json').get('stop_requested'):
            terminal = 'cancelled'
        document.update(status=terminal, active_detector=None)
    except BaseException as error:
        document.update(status="failed", active_detector=None, error=str(error))
        update_pair(root, document)
        raise
    update_pair(root, document)


def worker(root):
    root = Path(root)
    document = read(root / "job.json")
    try:
        with lock(root.parent.parent / ".gpu.lock", blocking=False):
            gpu_idle()
            hardware = probe(document["python"])
            if not hardware["ready"]:
                raise RuntimeError("Venv worker chưa sẵn sàng cho CUDA: " + str(hardware))
            document.update(status="running", started_at=now(), hardware=hardware)
            write(root / "job.json", document)
            subprocess.run([document["python"], "-m", "annoloop.source_worker", "--job", str(root)], check=True)
            document.update(status="completed", finished_at=now())
    except BaseException as error:
        document.update(status="failed", finished_at=now(), error=str(error))
        write(root / "job.json", document)
        raise
    write(root / "job.json", document)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker")
    mode.add_argument("--pair")
    options = parser.parse_args()
    worker(options.worker) if options.worker else pair_worker(options.pair)
