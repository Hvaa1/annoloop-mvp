"""Explicit negative-loss masks, limited to maintained dense-head interfaces."""


def inside_xy(torch,xy,boxes):
    result=torch.zeros(xy.shape[0],dtype=torch.bool,device=xy.device)
    for box in boxes:
        delta=xy-box[:2];c,s=torch.cos(box[6]),torch.sin(box[6])
        x=delta[:,0]*c+delta[:,1]*s;y=-delta[:,0]*s+delta[:,1]*c
        result|=(x.abs()<=box[3]/2)&(y.abs()<=box[4]/2)
    return result


class ChunkedOverlaps:
    """Same pairwise IoU, computed over anchor blocks to bound GPU peak memory.

    GT x anchors with ~110 source boxes peaked at 3.3GiB on a 4GB GPU.
    """
    def __init__(self,inner,chunk=65536):
        self.inner=inner;self.chunk=chunk

    def __call__(self,first,second,mode='iou',is_aligned=False):
        import torch
        if is_aligned or max(len(first),len(second))<=self.chunk:
            return self.inner(first,second,mode,is_aligned)
        if len(second)>=len(first):
            return torch.cat([self.inner(first,second[i:i+self.chunk],mode,False)
                              for i in range(0,len(second),self.chunk)],dim=1)
        return torch.cat([self.inner(first[i:i+self.chunk],second,mode,False)
                          for i in range(0,len(first),self.chunk)],dim=0)


def chunk_assigners(head):
    assigners=getattr(head,'bbox_assigner',None)
    assigners=assigners if isinstance(assigners,list) else [assigners] if assigners is not None else []
    for assigner in assigners:
        if hasattr(assigner,'iou_calculator') and not isinstance(assigner.iou_calculator,ChunkedOverlaps):
            assigner.iou_calculator=ChunkedOverlaps(assigner.iou_calculator)


def install(head,kind):
    import torch
    if kind in ('Anchor3DHead','CenterHead') and getattr(head,'train_cfg',None) is None:
        raise ValueError('MMDet head thiếu train_cfg; cần dựng model với cấu hình train trước khi cài signed-ignore')
    context=dict(boxes=None,masked=0)
    if kind=='Anchor3DHead':
        chunk_assigners(head)
        original=head.anchor_target_single_assigner
        def assign(assigner,anchors,gt,ignored,meta,*args,**kwargs):
            result=list(original(assigner,anchors,gt,ignored,meta,*args,**kwargs))
            boxes=context['boxes']
            if boxes is not None and len(boxes):
                flat=anchors.reshape(-1,anchors.shape[-1])
                mask=inside_xy(torch,flat[:,:2],boxes)&(result[3].abs().sum(-1)==0)&(result[1]>0)
                context['masked']+=int(mask.sum().item())
                result[1]=result[1].clone();result[1][mask]=0
            # No invented velocity supervision: geometry-only target/source.
            if result[3].shape[-1]>7:
                result[3]=result[3].clone();result[3][:,7:]=0
            return tuple(result)
        head.anchor_target_single_assigner=assign
    elif kind=='CenterHead':
        original=head.loss_cls.forward
        def loss(pred,target,*args,**kwargs):
            boxes=context['boxes']
            if boxes is not None and len(boxes):
                if args:raise ValueError('Gaussian loss positional weight chưa hỗ trợ')
                if pred.shape[0]!=1:raise ValueError('Ignore recipe hiện dùng batch size 1')
                bounds=head.train_cfg['point_cloud_range'];voxel=head.train_cfg['voxel_size'];stride=head.train_cfg['out_size_factor']
                yy,xx=torch.meshgrid(torch.arange(pred.shape[2],device=pred.device),torch.arange(pred.shape[3],device=pred.device),indexing='ij')
                xy=torch.stack((xx.reshape(-1)*voxel[0]*stride+bounds[0],yy.reshape(-1)*voxel[1]*stride+bounds[1]),-1)
                mask=inside_xy(torch,xy,boxes).reshape(1,1,pred.shape[2],pred.shape[3]).expand_as(pred)&(target==0)
                weight=kwargs.get('weight',torch.ones_like(pred));weight=weight.clone();weight[mask]=0
                context['masked']+=int(mask.sum().item());kwargs['weight']=weight
            return original(pred,target,**kwargs)
        head.loss_cls.forward=loss
        codes=list(head.train_cfg.get('code_weights',[]))
        if len(codes)>8:head.train_cfg['code_weights']=codes[:8]+[0.]*(len(codes)-8)
    else:raise ValueError('Không có signed-ignore handler cho '+kind)
    return context
