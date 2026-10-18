"""Explicit background download -> bind -> project verification orchestration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from .io import read,write,lock,now
from .reused.training_monitor import process_identity
from .settings import Settings,ROOT


def context(store,pid):
    project=store.get(pid);root=store.path(pid);cfg=read(root/'adaptation.json')
    value=dict(project_id=pid,dataset=project['dataset_sha256'],classes=cfg.get('adaptation_classes',project['schema']['classes']),
        selection=cfg.get('model_selection'),models=cfg.get('models'),input_mode=cfg.get('mmdet_input_mode','strict'),source=read(root/'setup.json')['source_evidence'])
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def status(root):
    state=read(Path(root)/'state.json')
    alive=bool(state.get('process_start') and process_identity(state.get('pid'))==state['process_start'])
    if state['status'] in ('starting','running') and not alive:state=dict(state,status='interrupted')
    return dict(state,process_alive=alive)


def active(store,pid):
    for path in (store.path(pid)/'preparation_jobs').glob('*/state.json'):
        state=status(path.parent)
        # The preparation worker may hand off to native verification itself;
        # every other process must respect this project's preparation lease.
        if state['process_alive'] and state.get('pid')!=os.getpid():return str(path.parent)
    return None


def start(store,pid,settings,axes,columns,mmdet_input_mode=None):
    from .adaptation import execution_busy
    from .runtime import worker_environment
    if not axes:raise ValueError('Xác nhận đơn vị/trục target trước khi chuẩn bị và kiểm tra.')
    if columns not in (3,4,5):raise ValueError('BIN cần 3/4/5 cột.')
    with lock(store.path(pid)/'.adaptation_dispatch.lock'):
        if execution_busy(store,pid):raise ValueError('Một job dự án đang chạy; không gửi thêm lượt chuẩn bị.')
        if mmdet_input_mode is None:mmdet_input_mode=read(store.path(pid)/'adaptation.json').get('mmdet_input_mode','strict')
        if mmdet_input_mode not in ('strict','current_scan_only'):raise ValueError('MMDet input mode không hợp lệ')
        root=store.path(pid)/'preparation_jobs'/uuid.uuid4().hex
        write(root/'spec.json',dict(project_id=pid,context=context(store,pid),axes=axes,columns=columns,mmdet_input_mode=mmdet_input_mode,created_at=now()))
        write(root/'state.json',dict(status='starting',stage='starting',created_at=now()))
        env=worker_environment(settings);env['ANNOLOOP_DATA_DIR']=str(settings.data)
        env['VINAI_PYTHON']=str(settings.python);env['ANNOLOOP_WORKER_PYTHON']=str(settings.trainer_python)
        with (root/'worker.log').open('w') as log:
            process=subprocess.Popen([str(settings.python),'-u','-m','annoloop.preparation_jobs','--run',str(root)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        state=read(root/'state.json');state.update(pid=process.pid,process_start=process_identity(process.pid))
        write(root/'state.json',state)
    return str(root)


def stop(root):
    state=status(root)
    if state['process_alive']:
        os.kill(state['pid'],signal.SIGTERM);return True
    return False


def run(root,settings=None):
    from .projects import ProjectStore
    from .model_hub import load_catalog
    from .adapter_factory import select_and_prepare
    from .preparation import check_prepared,overview
    from . import adaptation
    settings=settings or Settings.load();root=Path(root);spec=read(root/'spec.json')
    store=ProjectStore(settings.data);pid=spec['project_id'];child=None
    with lock(store.path(pid)/'.adaptation_dispatch.lock'):pass
    def stage(name,**values):
        state=read(root/'state.json');state.update(status='running',stage=name,**values);write(root/'state.json',state)
    def unchanged():
        if context(store,pid)!=spec['context']:raise ValueError('Lựa chọn model/class/dataset/source đã đổi; không gắn kết quả vào dự án.')
    try:
        unchanged();stage('download_and_prepare')
        cfg=read(store.path(pid)/'adaptation.json');catalog={m['id']:m for m in load_catalog()['models']}
        ids=list(dict.fromkeys(v['id'] for v in cfg.get('model_selection',{}).values() if v['kind']=='hub'))
        for identifier in ids:
            unchanged();stage('download_and_prepare',model_id=identifier)
            select_and_prepare(settings.data,catalog[identifier])
        unchanged();stage('check_recipe')
        report=check_prepared(store,pid,settings,spec['axes'],spec['columns'],spec.get('mmdet_input_mode'))
        if report['status']=='checking':
            child=report['job'];stage('verify_models',verification_job=child)
            while adaptation.job_status(child)['process_alive']:time.sleep(2)
            state=adaptation.job_status(child)
            if state['status']!='completed':raise ValueError('Kiểm chứng chưa đạt: '+str(state.get('error',state['status'])))
            report=overview(store,pid)
        write(root/'result.json',report)
        state=read(root/'state.json');state.update(status='completed' if report['ready'] else 'blocked',stage='finished',finished_at=now())
        write(root/'state.json',state)
    except BaseException as error:
        if child and adaptation.job_status(child)['process_alive']:
            adaptation.stop_job(child)
            while adaptation.job_status(child)['process_alive']:time.sleep(.5)
        state=read(root/'state.json');state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            error=type(error).__name__+': '+str(error),finished_at=now());write(root/'state.json',state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    def interrupt(*_):raise KeyboardInterrupt('User stopped model preparation')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    run(args.run)
