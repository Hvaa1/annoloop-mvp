"""Strict checkpoint coverage with the framework's versioned loading hooks.

Sparse kernel layouts are owned by MMDetection3D, not inferred from arbitrary
tensor permutations here. Keep PyTorch state metadata for its spconv1/2 hook.
"""
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy


def load_weights(model, raw_state):
    import torch
    if not isinstance(raw_state, Mapping):
        raise ValueError('MMDet checkpoint state_dict không hợp lệ')
    state=OrderedDict()
    for key,value in raw_state.items():
        name=key.removeprefix('module.')
        if name in state:
            raise ValueError('MMDet checkpoint key trùng sau bỏ module.: '+name)
        if not isinstance(value,torch.Tensor):
            raise ValueError('MMDet checkpoint không phải tensor: '+key)
        state[name]=value
    metadata=getattr(raw_state,'_metadata',None)
    if metadata is not None:
        state._metadata=OrderedDict()
        for key,value in metadata.items():
            # The outer DDP root is superseded by the wrapped model's root.
            if key=='' and 'module' in metadata:
                continue
            name='' if key=='module' else key.removeprefix('module.')
            state._metadata[name]=deepcopy(value)
    expected=model.state_dict()
    optional=lambda key:key.endswith('num_batches_tracked')
    missing=[key for key in expected if key not in state and not optional(key)]
    unexpected=sorted(set(state)-set(expected))
    if missing or unexpected:
        raise ValueError(f'Strict MMDet weights mismatch: missing={missing[:5]}; unexpected={unexpected[:5]}')
    if any(not torch.isfinite(value).all() for value in state.values() if torch.is_floating_point(value)):
        raise ValueError('Weights NaN/Inf')
    # The official version-aware module hook validates/converts sparse kernels.
    # PyTorch still raises for incompatible tensor shapes with strict=False.
    # False allows only the legacy BN counters checked explicitly above/below.
    try:
        result=model.load_state_dict(state,strict=False)
    except RuntimeError as error:
        raise ValueError('Strict MMDet weights mismatch sau framework load: '+str(error)) from error
    missing=[key for key in result.missing_keys if not optional(key)]
    if missing or result.unexpected_keys:
        raise ValueError(f'Strict MMDet weights mismatch: missing={missing[:5]}; unexpected={result.unexpected_keys[:5]}')
