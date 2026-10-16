"""Lazy adapter construction after an explicit user model selection.

Use maintained framework adapters; never generate/import arbitrary Python from
model names. Preparing a binding neither downloads nor verifies GPU capability.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import uuid
from .io import read, write, digest, now

RECIPES={
    'OpenPCDet':dict(backend='openpcdet',runtime_variable='ANNOLOOP_OPENPCDET_RUNTIME',
                    supports_raw_prediction=True,supports_generic_target_training=False),
    'mmdetection3d':dict(backend='mmdetection3d',runtime_variable=None,
                       supports_raw_prediction=True,supports_generic_target_training=False),
}


def recipe_for(model):
    recipe=RECIPES.get(model['framework'])
    if recipe is None:raise ValueError('Chưa có code adapter cho framework '+model['framework'])
    if model.get('modality','point_cloud')!='point_cloud':
        raise ValueError('Adapter point cloud không nhận model cần ảnh camera')
    return dict(recipe)


def prepare_adapter(data,model):
    """Create a per-checkpoint binding from installed files; no model imports/GPU."""
    from .model_hub import model_root,installed
    recipe=recipe_for(model)
    if not installed(data,model):raise ValueError('Cần tải đủ weights/config hợp lệ trước khi tạo adapter')
    root=model_root(data,model['id']);receipt=read(root/'installed.json')
    profile=dict(schema_version=1,model_id=model['id'],framework=model['framework'],
        architecture=model['architecture'],classes=model['classes'],recipe=recipe,
        checkpoint=receipt['weights'],checkpoint_sha256=receipt['sha256'],
        config=receipt['config_path'],config_sha256=digest(receipt['config_path']),
        upstream_root=str(root/'upstream'),revision=model['revision'],modality=model['modality'],
        status='prepared_unverified',runtime_verified=False,prepared_at=now(),
        input_contract='Preserve real configured point features; no fabricated intensity/time/sweeps.',
        adaptation_blockers=['Cần kiểm tra runtime, strict weights, feature/trục/đơn vị trên dữ liệu người dùng.',
                             'Generic predict không cung cấp chất lượng IoU head và signed-ignore training đã kiểm chứng.'])
    write(root/'adapter.json',profile)
    return profile


def load_adapter(data,model_id,settings):
    from .model_hub import model_root,load_catalog,installed
    model=next((m for m in load_catalog()['models'] if m['id']==model_id),None)
    if model is None:raise ValueError('Model không nằm trong registry')
    if not installed(data,model):raise ValueError('Weights/config cache đã thiếu hoặc thay đổi')
    root=model_root(data,model_id);profile=read(root/'adapter.json');receipt=read(root/'installed.json')
    expected=dict(model_id=model_id,framework=model['framework'],architecture=model['architecture'],
        classes=model['classes'],recipe=recipe_for(model),checkpoint=receipt['weights'],checkpoint_sha256=receipt['sha256'],
        config=receipt['config_path'],config_sha256=model['config_files'][model['config_path']]['sha256'],
        upstream_root=str(root/'upstream'),revision=model['revision'])
    if any(profile.get(k)!=v for k,v in expected.items()) or profile['checkpoint_sha256']!=digest(profile['checkpoint']) or profile['config_sha256']!=digest(profile['config']):
        raise ValueError('Adapter binding không khớp checkpoint/config/recipe')
    return FrameworkAdapter(profile,settings)


@dataclass
class FrameworkAdapter:
    profile: dict
    settings: object

    def predict_raw(self,frames,output, *, device='cuda:0',coordinates_confirmed=False,bin_columns=None):
        """Explicit real inference, no calibration/training or silent feature synthesis.

        Raw predictions are separate from DetectorAdapter's calibrated adaptation
        contract. Do not plug classification-only scores into an IoU-head field.
        """
        if not coordinates_confirmed:raise ValueError('Cần xác nhận trục và đơn vị point cloud')
        if not device.startswith('cuda'):raise ValueError('Adapter MVP dùng GPU CUDA')
        if not frames:raise ValueError('Cần frame point cloud để predict')
        root=Path(output)/uuid.uuid4().hex
        write(root/'request.json',dict(adapter=self.profile,frames=frames,device=device,
                                      coordinates_confirmed=True,bin_columns=bin_columns))
        env=os.environ.copy()
        env['PYTHONPATH']=str(Path(__file__).parents[1])+os.pathsep+env.get('PYTHONPATH','')
        command=[str(self.settings.trainer_python),'-u','-m','annoloop.framework_worker',str(root)]
        # Same cross-process lease as native workers; never overlap GPU models.
        from .io import lock
        from .device import gpu_idle
        with lock(Path(self.settings.data)/'.gpu.lock',blocking=False):
            gpu_idle()
            with (root/'worker.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return read(root/'result.json')


def select_and_prepare(data,model, *, downloader=None):
    """Future user action: resolve code first, then download and bind only this model."""
    from .model_hub import download,model_root
    recipe_for(model)  # Unsupported backend/modality fails before spending bandwidth.
    root=model_root(data,model['id'])
    write(root/'selection.json',dict(status='downloading',model_id=model['id'],started_at=now()))
    try:
        (downloader or download)(data,model,resume=True)
        write(root/'selection.json',dict(status='preparing_adapter',model_id=model['id']))
        from .connect_model import connect
        report=connect(data,model)
        profile=read(report['adapter'])
        write(root/'selection.json',dict(status='prepared_unverified',model_id=model['id'],
                                       adapter=str(root/'adapter.json'),finished_at=now(),runtime_verified=False))
        return profile
    except Exception as error:
        write(root/'selection.json',dict(status='failed',model_id=model['id'],error=str(error),finished_at=now()))
        raise
