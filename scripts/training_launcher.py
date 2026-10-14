"""Register and launch any local trainer in the background for watch_train.sh."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from training_monitor import DATA, HERE, process_identity, registered_process_alive


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--cwd',type=Path,default=HERE)
    p.add_argument('--name',default='local training')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:p.error('Provide trainer command after --')
    run=args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    state=run/'training.json'
    old=json.loads(state.read_text()) if state.exists() else {}
    if not args.worker:
        if registered_process_alive(run):
            p.error('A trainer/pipeline is already running in this folder; use watch_train.sh')
        identity=process_identity(old['pid']) if old.get('pid') else None
        if identity is not None and identity==old.get('process_start'):
            p.error('A registered trainer is still running in this folder')
        if old or (run/'train.log').exists():
            p.error('Use a fresh training folder; preserve the existing launcher receipt/log')
        initial=dict(status='starting',name=args.name,run=str(run),command=command,
                     cwd=str(args.cwd.resolve()),run_id=uuid.uuid4().hex,
                     started_at=datetime.now().astimezone().isoformat())
        write_json(state,initial)
        latest=DATA/'monitor/latest_training.json'
        latest.parent.mkdir(parents=True,exist_ok=True)
        write_json(latest,dict(run=str(run),name=args.name,run_id=initial['run_id']))
        with (run/'launcher.log').open('w') as log:
            worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--run',str(run),
                    '--cwd',str(args.cwd.resolve()),'--name',args.name,'--',*command],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'Launched {args.name}; worker PID {worker.pid}\nRun: {run}\nMonitor: bash {HERE}/scripts/watch_train.sh',flush=True)
        return
    metadata=old
    metadata.update(status='running',pid=os.getpid(),process_start=process_identity(os.getpid()))
    write_json(state,metadata)
    child=None

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        with (run/'train.log').open('w') as log:
            child=subprocess.Popen(command,cwd=args.cwd.resolve(),stdout=log,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
            metadata.update(trainer_pid=child.pid,trainer_process_start=process_identity(child.pid))
            write_json(state,metadata)
            code=child.wait()
        metadata.update(status='completed' if code==0 else 'failed',exit_code=code)
    except KeyboardInterrupt:
        if child is not None:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        metadata.update(status='stopped',exit_code=-signal.SIGTERM)
    except Exception as exc:
        metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}',exit_code=None)
        raise
    finally:
        metadata['finished_at']=datetime.now().astimezone().isoformat()
        write_json(state,metadata)


if __name__=='__main__':
    main()
