"""Concrete blocking ST3D GPU adapter; used under RoundController's GPU lease."""
from pathlib import Path
import os
import signal
import subprocess
import time
import uuid
from .adapters import DetectorAdapter
from .io import read, write, digest, now
from .runtime import worker_environment
from .settings import ROOT
from .reused.training_monitor import process_identity


class NativeDetectorAdapter(DetectorAdapter):
    def __init__(self, candidate, classes, frames, output, settings, bin_columns=4):
        self.candidate = dict(candidate)
        for field, hash_key in (('path', 'checkpoint_sha256'), ('config', 'config_sha256')):
            if hash_key not in self.candidate:
                self.candidate[hash_key] = digest(self.candidate[field])
        self.classes = classes
        self.frames = frames
        self.output = Path(output)
        self.settings = settings
        self.bin_columns = bin_columns
        self.reuse_job = None

    def spec(self, checkpoint, frames=None):
        if digest(self.candidate['config']) != self.candidate['config_sha256']:
            raise ValueError('Config đã thay đổi sau khi lưu thiết lập')
        if str(checkpoint) == self.candidate['path'] and digest(checkpoint) != self.candidate['checkpoint_sha256']:
            raise ValueError('Checkpoint khởi tạo đã thay đổi')
        return dict(**{k: self.candidate[k] for k in ('config', 'runtime', 'classes')},
                    path=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                    config_sha256=digest(self.candidate['config']), frames=frames or self.frames,
                    bin_columns=self.bin_columns, coordinates_confirmed=True,
                    score_threshold=.05, iou_threshold=.5, require_components=True,
                    save_point_preview=False)

    def infer(self, checkpoint, frames, validation=False):
        root = self.output / ('source' if validation else 'target') / uuid.uuid4().hex
        spec = self.spec(checkpoint, frames)
        spec['mode'] = 'validation' if validation else 'target_sample'
        if self.reuse_job:
            from .inference_cache import matches, seed_target, finish_cached_target
            key = self.output.name
            if validation:
                for path in sorted((self.reuse_job / key / 'source').glob('*/result.json'),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
                    old = read(path.parent / 'spec.json'); result = read(path)
                    if matches(old, spec) and result.get('domain') == 'source_validation' \
                       and result.get('checkpoint_sha256') == spec['checkpoint_sha256']:
                        write(root / 'reuse.json', dict(source=str(path), sha256=digest(path), optimizer_steps=0))
                        return path.parent, result
            else:
                spec['reused_frames'] = seed_target(self.reuse_job, key, spec, root)
                if len(spec['reused_frames']) == len(spec['frames']):
                    write(root / 'spec.json', spec)
                    return root, finish_cached_target(root, spec)
        write(root / 'spec.json', spec)
        write(root / 'inspection.json', dict(status='running', started_at=now()))
        command = [str(self.settings.trainer_python), '-u', '-c',
                   'import sys; from annoloop.inspection_worker import run; run(sys.argv[1])', str(root)]
        self.call(command, root / 'inference.log')
        return root, read(root / 'result.json')

    def call(self, command, log):
        with Path(log).open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=worker_environment(self.settings),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
            write(Path(log).with_suffix('.process.json'), dict(pid=process.pid,
                  process_start=process_identity(process.pid), command=command, started_at=now()))
            try:
                code = process.wait()
                if code:
                    raise RuntimeError(f'GPU worker exit {code}; xem {log}')
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise

    def predict_target(self, checkpoint, frames):
        root, result = self.infer(checkpoint, frames)
        return {f['frame_id']: dict(boxes=[b for b in read(root / f['file'])['boxes']
                                         if b['class_name'] in self.classes]) for f in result['frames']}

    def predict_source_validation(self, checkpoint):
        _, result = self.infer(checkpoint, None, validation=True)
        records = result['calibration_records']
        return dict(domain='source_validation', boxes=[b for b in records['boxes'] if b['class_name'] in self.classes])

    def train_target(self, checkpoint, labels_path, output, policy):
        from .jobs import status, stop
        root = Path(output)
        if root.exists():
            raise ValueError('Giữ run target cũ; không train ghi đè')
        if policy.get('ckpt_interval') != 10 or policy.get('save_final') is not True:
            raise ValueError('Cần lưu mỗi 10 epoch + final')
        spec = self.spec(checkpoint)
        resume = policy.get('resume_checkpoint')
        if resume:
            if digest(resume['path']) != resume['sha256']:
                raise ValueError('Checkpoint optimizer cần tiếp tục đã đổi')
            spec = self.spec(resume['path'])
        spec.update(labels_path=str(labels_path), shared_labels_sha256=digest(labels_path), policy=policy)
        write(root / 'spec.json', spec)
        write(root / 'job.json', dict(kind='target_adaptation', id=root.name, status='planned',
              spec=spec, python=str(self.settings.trainer_python), created_at=now()))
        command = [str(self.settings.python), str(self.settings.training_launcher), '--run', str(root),
                   '--cwd', str(ROOT), '--name', f'AnnoLoop target · {self.candidate["architecture"]}', '--',
                   str(self.settings.trainer_python), '-u', '-m', 'annoloop.native_training', '--run', str(root)]
        subprocess.run(command, env=worker_environment(self.settings), text=True, capture_output=True, check=True)
        began = time.monotonic()
        try:
            while True:
                snapshot = status(root)
                state = snapshot['training'].get('status')
                if state in ('failed', 'stopped', 'interrupted') and not snapshot['process_alive']:
                    raise RuntimeError(f'Target trainer {state}; giữ artifacts {root}')
                if state == 'completed' and not snapshot['process_alive']:
                    return read(root / 'trained.json')
                if state == 'starting' and time.monotonic() - began > 30:
                    raise RuntimeError('Target launcher chưa đăng ký PID')
                time.sleep(1)
        except BaseException:
            # Launcher may have written its starting receipt before PID registration.
            # Wait for that short handoff before stopping, rather than orphaning it.
            deadline = time.monotonic() + 30
            while (root / 'training.json').exists() and status(root)['training'].get('status') == 'starting' and time.monotonic() < deadline:
                time.sleep(.2)
            if (root / 'training.json').exists():
                stop(root)
            # Do not release the controller's GPU lease while its trainer lives.
            while status(root)['process_alive']:
                time.sleep(.5)
            raise
