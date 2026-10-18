"""Construct an adaptation model with its training components intact."""
from copy import deepcopy


def build_model(cfg,device='cuda:0'):
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import MODELS
    from mmdet3d.apis.inference import convert_SyncBN

    model_cfg=deepcopy(cfg.model)
    if model_cfg.get('train_cfg') is None:
        raise ValueError('MMDet config thiếu model.train_cfg cho adaptation / signed-ignore')
    # As in the framework's inference helper, use regular BN locally; unlike
    # that helper, never replace train_cfg=None before building assigners/head.
    convert_SyncBN(model_cfg)
    init_default_scope(cfg.get('default_scope','mmdet3d'))
    model=MODELS.build(model_cfg)
    model.cfg=cfg
    model.to(device)
    model.eval()
    return model
