"""Route by implemented binding, never dispatch a generic model into ST3D."""
def create(model,classes,frames,output,settings,columns,source_setup):
    if model.get('backend','native_st3d')=='native_st3d':
        from .native_adapter import NativeDetectorAdapter
        return NativeDetectorAdapter(model,classes,frames,output,settings,columns)
    if model.get('framework')=='mmdetection3d' and model.get('mmdet_recipe',{}).get('implementation_supported'):
        from .mmdet_adapter import MMDetDetectorAdapter
        return MMDetDetectorAdapter(model,classes,frames,output,settings,columns,source_setup)
    raise ValueError('Chưa có DetectorAdapter train/calibration cho model này')
