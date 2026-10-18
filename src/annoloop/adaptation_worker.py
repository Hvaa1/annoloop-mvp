"""Background native verification/round controller; no Streamlit imports."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
from .io import read, write, digest, now, lock
from .settings import ROOT, Settings


def run(root):
    root = Path(root)
    spec = read(root / 'spec.json')
    settings = Settings.load()
    from .projects import ProjectStore
    from .native_adapter import NativeDetectorAdapter
    from .controller import RoundController
    from .device import gpu_idle
    from .inspection import load_xyz
    from .runtime import worker_environment
    store = ProjectStore(settings.data)
    pid = spec['project_id']
    project_root = store.path(pid)
    # Wait for dispatch to finish writing PID and releasing the brief GPU lease.
    with lock(project_root / '.adaptation_dispatch.lock'):
        pass
    if digest(project_root / 'dataset.json') != spec['dataset_sha256']:
        raise ValueError('Dataset identity thay đổi')
    for evidence in spec['source_evidence'].values():
        if digest(evidence['path']) != evidence['sha256']:
            raise ValueError('Source infos đã thay đổi')
    inventory = store.verify_dataset(pid)
    frames = [f for f in inventory['frames'] if f['split'] == 'adaptation']
    config = spec['config']
    from .point_format import verification_contract
    point_contract=verification_contract(spec['dataset_sha256'],config,frames)
    for model in config['models'].values():
        if digest(model['path']) != model['checkpoint_sha256'] or digest(model['config']) != model['config_sha256']:
            raise ValueError('Checkpoint/config identity thay đổi sau dispatch')
    classes = store.get(pid)['schema']['classes']
    continuation = spec.get('resume_plan')
    if continuation:
        from .resume import plan
        checked = plan(store, pid, continuation['work'])
        if checked != continuation:
            raise ValueError('Artifacts cần tiếp tục đã đổi sau dispatch')
    from .detector_factory import create
    adapters = {key: create(model, classes, frames, root / key, settings, config['bin_columns'],read(project_root/'setup.json'))
                for key, model in config['models'].items()}
    if continuation:
        for adapter in adapters.values():
            adapter.reuse_job = Path(continuation['job'])
    if spec['mode'] == 'verify':
        preparation_checks={}
        with lock(settings.data / '.gpu.lock', blocking=False):
            for key in ('A', 'B'):
                gpu_idle()
                directory = root / key / 'verify'
                model_spec = adapters[key].spec(config['models'][key]['path'])
                model_spec.update(mode='validation', limit=1)
                write(directory / 'spec.json', model_spec)
                write(directory / 'inspection.json', dict(status='running'))
                state = read(root / 'state.json')
                state.update(active_detector=key, stage='verify_native_gpu')
                write(root / 'state.json', state)
                command = [str(getattr(adapters[key],'settings',settings).trainer_python), '-u', '-m',
                           getattr(adapters[key],'worker_module','annoloop.native_training'),
                           '--verify', '--run', str(directory)]
                adapters[key].call(command, directory / 'verification.log')
                if read(directory / 'verified.json')['optimizer_steps'] != 0:
                    raise ValueError('Verification không được optimizer step')
                gpu_idle()
                if config.get('preparation_protocol')==1:
                    from .fusion import calibrate_source
                    state=read(root/'state.json');state.update(stage='calibrate_source_validation',active_detector=key)
                    write(root/'state.json',state)
                    records=adapters[key].predict_source_validation(config['models'][key]['path'])
                    calibration=calibrate_source(records,config['models'][key]['checkpoint_sha256'],
                        config.get('adaptation_classes',classes))
                    calibration_path=root/key/'calibration.json';write(calibration_path,calibration)
                    preparation_checks[key]=dict(verification=str(directory/'verified.json'),
                        verification_sha256=digest(directory/'verified.json'),calibration=str(calibration_path),
                        calibration_sha256=digest(calibration_path))
                    gpu_idle()
            write(project_root / 'native_verified.json', dict(
                models={k: {f: m[f] for f in ('checkpoint_sha256', 'config_sha256', 'runtime')}
                        for k, m in config['models'].items()}, source_evidence=spec['source_evidence'],
                point_contract=point_contract,
                optimizer_steps=0, checked_at=now(), artifacts=str(root),
                preparation_protocol=config.get('preparation_protocol'),preparation_checks=preparation_checks,
                adaptation_classes=config.get('adaptation_classes',classes)))
    else:
        controller = RoundController(store, pid, adapters, {k: m['path'] for k, m in config['models'].items()},
            lambda frame: load_xyz(frame['media'], config['bin_columns']), epochs=config['epochs'],
            training_options=dict(learning_rate=config['learning_rate'], source_weight=config['source_weight']),
            adaptation_classes=config.get('adaptation_classes', classes),
            attempt_id=spec.get('attempt_id', root.name),
            resume_plan=continuation,
            memory_policy=config.get('memory_policy'),
            on_stage=lambda stage, key=None: update_stage(root, stage, key, controller.active_work))
        write(root / 'result.json', controller.run())


def update_stage(root, stage, key=None, work_directory=None):
    state = read(root / 'state.json')
    state.update(stage=stage, active_detector=key)
    if work_directory:
        state['work_directory'] = str(work_directory)
    write(root / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run)
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Controller interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        state = read(root / 'state.json')
        from .reused.training_monitor import process_identity
        state.update(status='running', started_at=now(), pid=os.getpid(), process_start=process_identity(os.getpid()))
        write(root / 'state.json', state)
        run(root)
        state = read(root / 'state.json')
        state.update(status='completed', finished_at=now(), active_detector=None)
        write(root / 'state.json', state)
    except BaseException as error:
        state = read(root / 'state.json')
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error), finished_at=now())
        write(root / 'state.json', state)
        raise
