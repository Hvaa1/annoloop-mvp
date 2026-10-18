"""CPU-only import preflight for the model's Python and compiled ST3D runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
from .settings import ROOT, WORKSPACE


def configure_sparse_headers():
    """cumm 0.4 ignores CUDA_HOME when it builds NVRTC include paths."""
    toolkit=Path(os.getenv('ANNOLOOP_CUDA_TOOLKIT',
                 os.getenv('CUDA_HOME',ROOT/'runtime/cuda-11.8'))).expanduser().resolve()
    if not (toolkit/'include/cuda/std/cassert').is_file():
        raise ValueError('CUDA toolkit thiếu include/cuda/std/cassert: '+str(toolkit))
    library=toolkit/'lib64' if (toolkit/'lib64').is_dir() else toolkit/'lib'
    if not library.is_dir():raise ValueError('CUDA toolkit thiếu folder thư viện: '+str(toolkit))
    import cumm.common
    cumm.common._CACHED_CUDA_INCLUDE_LIB=(toolkit/'include',library)
    return str(toolkit)


def worker_environment(settings):
    environment=os.environ.copy()
    paths=[str(ROOT/'src')]
    if settings.runtime_deps and settings.runtime_deps.is_dir():paths.append(str(settings.runtime_deps))
    paths.append(environment.get('PYTHONPATH',''))
    environment['PYTHONPATH']=os.pathsep.join(paths)
    toolkit=Path(os.getenv('ANNOLOOP_CUDA_TOOLKIT',os.getenv('CUDA_HOME',ROOT/'runtime/cuda-11.8')))
    environment.setdefault('ANNOLOOP_CUDA_TOOLKIT',str(toolkit))
    # cuDNN resolves libnvrtc.so dynamically; Torch wheels may ship only a hashed name.
    native_libs=sorted(settings.trainer_python.parent.parent.glob('lib/python*/site-packages/torch/lib'))
    search=[]
    for library in native_libs:
        candidates=sorted(library.glob('libnvrtc-*.so.*'))
        if candidates and not (library/'libnvrtc.so').is_file():
            alias=settings.data/'runtime_libraries'/hashlib.sha256(str(library).encode()).hexdigest()[:12]
            alias.mkdir(parents=True,exist_ok=True)
            target=alias/'libnvrtc.so'
            try:target.symlink_to(candidates[0])
            except FileExistsError:pass
            search.append(str(alias))
        search.append(str(library))
    search.extend(filter(None,environment.get('LD_LIBRARY_PATH','').split(os.pathsep)))
    if search:environment['LD_LIBRARY_PATH']=os.pathsep.join(dict.fromkeys(search))
    return environment


def check(runtime):
    try:
        runtime=Path(runtime).resolve()
        if not (runtime/'pcdet/config.py').is_file():raise ValueError('Runtime ST3D không hợp lệ')
        sys.path[:0]=[str(runtime),str(runtime/'tools')]
        toolkit=configure_sparse_headers()
        import torch
        from pcdet.config import cfg,cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network,model_fn_decorator
        from train_utils.optimization import build_optimizer,build_scheduler
        from train_utils.train_utils import train_model,checkpoint_state,save_checkpoint
        return dict(ready=True,python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                    runtime=str(runtime),cuda_toolkit=toolkit,checks='trainer imports, compiled ops and NVRTC include configuration',gpu_kernels_checked=False)
    except Exception as error:
        return dict(ready=False,python=sys.executable,error=type(error).__name__+': '+str(error))


def probe_runtime(spec,settings):
    try:
        result=subprocess.run([str(settings.trainer_python),'-m','annoloop.runtime','--check',str(spec['runtime'])],
            env=worker_environment(settings),text=True,capture_output=True,timeout=90)
        lines=result.stdout.strip().splitlines()
        if not lines:raise ValueError(result.stderr[-1800:] or 'Worker không trả kết quả preflight')
        value=json.loads(lines[-1])
        if result.returncode:value.update(ready=False)
        return value
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:
        return dict(ready=False,python=str(settings.trainer_python),error=str(error))


def require_runtime(spec,settings):
    if spec.get('framework')=='mmdetection3d' and spec.get('mmdet_recipe',{}).get('implementation_supported'):
        from .mmdet_adapter import preflight
        return preflight(settings)
    value=probe_runtime(spec,settings)
    if not value['ready']:
        raise ValueError('Môi trường model chưa sẵn sàng; chưa gửi trainer. '+value.get('error','')+
                         ' · Python: '+value['python'])
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',required=True)
    result=check(parser.parse_args().check)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    raise SystemExit(0 if result['ready'] else 1)
