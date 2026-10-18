"""MMDetection3D recipe adapter; runtime is imported only by leaf workers."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import uuid
from .native_adapter import NativeDetectorAdapter
from .io import read,write,now
from .runtime import worker_environment


def worker_python(settings):
    from .settings import WORKSPACE
    dedicated=WORKSPACE/'.venv-mmdet3d/bin/python'
    fallback=dedicated if dedicated.is_file() else settings.trainer_python
    return Path(os.getenv('ANNOLOOP_MMDET3D_PYTHON',str(fallback))).expanduser().absolute()


def preflight(settings):
    actual=replace(settings,worker_python=worker_python(settings))
    result=subprocess.run([str(actual.trainer_python),'-m','annoloop.mmdet_adapter','--probe'],
        env=worker_environment(actual),capture_output=True,text=True,timeout=90)
    try:report=json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError,ValueError):raise ValueError('MMDetection3D runtime preflight lỗi: '+result.stderr[-1200:])
    if result.returncode or not report['ready']:raise ValueError('MMDetection3D runtime chưa sẵn sàng: '+report.get('error','')+'. Xem scripts/setup_mmdet3d.sh')
    return report


class MMDetDetectorAdapter(NativeDetectorAdapter):
    worker_module='annoloop.mmdet_worker'

    def __init__(self,candidate,classes,frames,output,settings,bin_columns=4,source_setup=None):
        super().__init__(candidate,classes,frames,output,replace(settings,worker_python=worker_python(settings)),bin_columns)
        self.source_setup=source_setup

    def spec(self,checkpoint,frames=None):
        spec=super().spec(checkpoint,frames)
        if self.source_setup is None:raise ValueError('Thiếu source setup để calibration/train MMDetection3D')
        spec.update(mmdet_recipe=self.candidate['mmdet_recipe'],source_setup=self.source_setup,architecture=self.candidate['architecture'])
        return spec

    def infer(self,checkpoint,frames,validation=False):
        # Fresh immutable receipts. Native caches are not reused across backend
        # score semantics; completed optimizer checkpoints still support resume.
        root=self.output/('source' if validation else 'target')/uuid.uuid4().hex
        spec=self.spec(checkpoint,frames);spec['mode']='validation' if validation else 'target_sample'
        write(root/'spec.json',spec);write(root/'inspection.json',dict(status='running',started_at=now()))
        self.call([str(self.settings.trainer_python),'-u','-m',self.worker_module,'--infer','--run',str(root)],root/'inference.log')
        return root,read(root/'result.json')


def probe():
    import sys
    try:
        import torch,mmcv,mmengine,mmdet,mmdet3d
        from mmcv.ops import nms3d
        from mmdet3d.apis import init_model
        from mmdet3d.structures import Det3DDataSample
        if not mmdet3d.__version__.startswith('1.4.'):raise ValueError('Recipe yêu cầu MMDetection3D 1.4.x')
        return dict(ready=True,python=sys.executable,torch=torch.__version__,mmdet3d=mmdet3d.__version__,gpu_kernels_checked=False)
    except Exception as error:return dict(ready=False,python=sys.executable,error=type(error).__name__+': '+str(error))


if __name__=='__main__':
    print(json.dumps(probe()))
