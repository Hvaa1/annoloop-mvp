"""Terminal monitor for any local training log; no model/GPU libraries needed."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("ANNOLOOP_DATA_DIR", HERE/"data")).expanduser().absolute()
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def process_identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rpartition(') ')[2].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def registered_process_alive(run,ignore_pid=None):
    run=Path(run)
    for path in [run/'training.json',run/'pipeline.json']:
        receipt=read_json(path)
        if receipt.get('pid'):
            if receipt['pid']==ignore_pid:
                continue
            identity=process_identity(receipt['pid'])
            if identity is not None and identity==receipt.get('process_start'):
                return True
    try:
        pid=int((run/'launcher.pid').read_text())
        if pid==ignore_pid:return False
        return process_identity(pid) is not None and 'co_training.run' in Path(f'/proc/{pid}/cmdline').read_text()
    except (OSError,ValueError):
        return False


def read_log(path):
    """Read only the last 128 KiB, including tqdm carriage-return updates."""
    path = Path(path)
    if not path.is_file():
        return dict(path=str(path),status='waiting')
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-131072))
        text = ANSI.sub('',stream.read().decode(errors='replace'))
    steps = re.findall(r'total_it\s*=\s*(\d+)',text)
    epochs = re.findall(r'epochs:\s*\d+%\|[^|]*\|\s*(\d+)\s*/\s*(\d+)',text)
    if not epochs:
        epochs = re.findall(r'\bEpoch\s*[\[: ]\s*(\d+)\s*/\s*(\d+)',text,re.I)
    losses = re.findall(r'loss=([0-9.eE+-]+)\(([0-9.eE+-]+)\), st_loss=([0-9.eE+-]+)\(([0-9.eE+-]+)\)',text)
    result = dict(path=str(path),status='log_available',age_seconds=max(0,time.time()-path.stat().st_mtime),
                  last_line=next((line.strip() for line in reversed(text.splitlines()) if line.strip()),''))
    if steps:
        result['optimizer_steps'] = int(steps[-1])
    if epochs:
        result['epoch_completed'],result['epochs'] = map(int,epochs[-1])
    if losses:
        source,source_mean,target,target_mean = map(float,losses[-1])
        result['loss'] = dict(source=source,source_mean=source_mean,target=target,target_mean=target_mean)
    else:
        generic = re.findall(r'\bloss\s*[:=]\s*([0-9.eE+-]+)',text,re.I)
        if generic:
            result['loss'] = dict(value=float(generic[-1]))
    return result


def snapshot(run=None, logs=None, total_steps=None):
    if run is None and not logs:
        latest = read_json(DATA/'monitor/latest_training.json')
        run = Path(latest['run']) if latest else DATA/'monitor/no_run'
    run = Path(run).resolve() if run else None
    training = read_json(run/'training.json') if run else {}
    pipeline = read_json(run/'pipeline.json') if run else {}
    status = training.get('status',pipeline.get('status','unregistered_log'))
    process = None
    if training.get('pid'):
        identity = process_identity(training['pid'])
        process = identity is not None and identity == training.get('process_start')
    elif pipeline.get('pid'):
        identity = process_identity(pipeline['pid'])
        process = identity is not None and identity == pipeline.get('process_start')
    elif run and (run/'launcher.pid').exists():
        try:
            pid = int((run/'launcher.pid').read_text())
            process = process_identity(pid) is not None and 'co_training.run' in Path(f'/proc/{pid}/cmdline').read_text()
        except (ValueError,OSError):
            process = False
    if status in ['running','starting'] and process is False:
        status = 'interrupted_process_not_alive'
    if logs:
        paths = list(map(Path,logs))
    elif pipeline and 'planned_rounds' in pipeline:
        paths = [Path(pipeline['active_log'])] if pipeline.get('active_log') else []
        if pipeline.get('active_training_run'):
            request = read_json(run/'request.json')
            total_steps = total_steps or request.get('epochs', 0)*pipeline.get('frames', 0) or None
    elif pipeline:
        paths = [run/f'{name}.log' for name in ['SECOND_M2_joint','PVRCNN_M2_joint']]
        consensus = read_json(run/'consensus/consensus.json')
        if consensus:
            total_steps = total_steps or pipeline['epochs']*len(consensus['frames'])
    else:
        paths = [run/'train.log']
    rows = []
    for path in paths:
        row = read_log(path)
        row['expected_steps'] = total_steps
        if run and pipeline:
            if 'planned_rounds' in pipeline:
                receipt = read_json(Path(pipeline['active_training_run'])/'trained.json') if pipeline.get('active_training_run') else {}
            else:
                receipt = read_json(run/path.stem/'trained.json')
            if receipt:
                row.update(status='trained',optimizer_steps=receipt['iterations'],epochs=receipt['epochs'],
                           epoch_completed=receipt['epochs'],checkpoint=receipt['m1'])
        rows.append(row)
    try:
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                              '--format=csv,noheader'],capture_output=True,text=True,timeout=5)
        gpu_text = (gpu.stdout or gpu.stderr).strip()
    except (OSError,subprocess.TimeoutExpired):
        gpu_text = 'nvidia-smi unavailable'
    return dict(time=datetime.now().astimezone().isoformat(timespec='seconds'),run=str(run) if run else None,
                status=status,process_alive=process,current_stage=pipeline.get('current_stage'),
                exit_code=training.get('exit_code'),gpu=gpu_text,logs=rows)


def render(value):
    print(f'[{value["time"]}] {value["status"]}')
    if value['run']:
        print('Run:',value['run'])
    if value['current_stage']:
        print('Stage:',value['current_stage'])
    print('GPU: name, VRAM used/total (MiB), utilization (%), temperature (C)')
    print('    ',value['gpu'])
    for row in value['logs']:
        steps = str(row.get('optimizer_steps','?'))
        if row.get('expected_steps'):
            steps += '/'+str(row['expected_steps'])
        epoch = f' | epoch {row["epoch_completed"]}/{row["epochs"]}' if 'epochs' in row else ''
        print(f'\n{Path(row["path"]).name}: {row["status"]} | optimizer steps {steps}{epoch}')
        if 'loss' in row:
            print('    loss:',', '.join(f'{k}={v:.4f}' for k,v in row['loss'].items()))
        if 'age_seconds' in row:
            print(f'    log updated {row["age_seconds"]:.0f}s ago')
        if 'checkpoint' in row:
            print('    checkpoint:',row['checkpoint'])
        elif row.get('last_line'):
            print('    '+row['last_line'][-240:])
    if value['process_alive'] is None:
        print('\nProcess chưa đăng ký: log chỉ cho biết lần cập nhật cuối, không chứng minh train còn chạy.')
    if value['exit_code'] is not None:
        print('\nExit code:',value['exit_code'])
    print('\nCtrl+C chỉ đóng monitor; không dừng trainer.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,help='Folder output; mặc định run đăng ký mới nhất')
    p.add_argument('--log',type=Path,action='append',help='Log bất kỳ; có thể truyền nhiều lần')
    p.add_argument('--total-steps',type=int)
    p.add_argument('--interval',type=float,default=5.)
    p.add_argument('--once',action='store_true',help='In một snapshot rồi thoát')
    p.add_argument('--json',action='store_true')
    p.add_argument('--no-clear',action='store_true')
    args=p.parse_args()
    if args.interval<=0 or (args.total_steps is not None and args.total_steps<1):
        p.error('interval/total-steps must be positive')
    try:
        while True:
            value=snapshot(args.run,args.log,args.total_steps)
            if args.json:
                print(json.dumps(value,indent=2),flush=True)
            else:
                if sys.stdout.isatty() and not args.no_clear and not args.once:
                    print('\033[2J\033[H',end='')
                render(value)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__=='__main__':
    main()
