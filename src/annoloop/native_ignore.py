"""Runtime hooks for signed ignore boxes; upstream ST3D files stay intact."""


def install(model):
    import torch
    from pcdet.utils.box_utils import boxes3d_nearest_bev_iou
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

    head = model.dense_head
    model._annoloop_ignore_counts={'dense':0,'roi':0,'point':0}
    original = head.assign_targets

    def assign_dense(gt_boxes):
        positive = [gt[gt[:, -1] > 0] for gt in gt_boxes]
        packed = gt_boxes.new_zeros((len(positive), max(1, max(len(g) for g in positive)), gt_boxes.shape[-1]))
        for index, gt in enumerate(positive):
            packed[index, :len(gt)] = gt
        targets = original(packed)
        anchors = torch.cat(head.anchors, dim=-3).reshape(-1, head.anchors[0].shape[-1])
        for index, gt in enumerate(gt_boxes):
            ignored = gt[gt[:, -1] < 0, :7]
            if not len(ignored):
                continue
            # Ignore across every anchor class, preserving positive assignments.
            for start in range(0, len(anchors), 4096):
                chunk = slice(start, start + 4096)
                overlap = boxes3d_nearest_bev_iou(anchors[chunk, :7], ignored).max(dim=1)[0]
                labels = targets['box_cls_labels'][index, chunk]
                mask=(overlap >= .1) & (labels <= 0)
                model._annoloop_ignore_counts['dense']+=int((mask & (labels==0)).sum().item())
                labels[mask] = -1
        return targets

    if head.use_multihead:
        raise ValueError('Signed ignore adapter hiện chỉ hỗ trợ AnchorHeadSingle')
    head.assign_targets = assign_dense

    def ignore_roi(module, inputs, output):
        for index, gt in enumerate(inputs[0]['gt_boxes']):
            ignored = gt[gt[:, -1] < 0, :7]
            if len(ignored):
                overlap = boxes_iou3d_gpu(output['rois'][index, :, :7], ignored).max(dim=1)[0]
                mask = (overlap >= .1) & (output['reg_valid_mask'][index] == 0)
                model._annoloop_ignore_counts['roi']+=int((mask & (output['rcnn_cls_labels'][index]!=-1)).sum().item())
                output['rcnn_cls_labels'][index, mask] = -1
                output['reg_valid_mask'][index, mask] = 0
        return output

    model.roi_head.proposal_target_layer.register_forward_hook(ignore_roi)
    if model.model_cfg.NAME == 'PVRCNN':
        original_points = model.point_head.assign_targets

        def assign_points(batch):
            targets = original_points(batch)
            coords = batch['point_coords']
            for index, gt in enumerate(batch['gt_boxes']):
                ignored, positive = gt[gt[:, -1] < 0, :7], gt[gt[:, -1] > 0, :7]
                if not len(ignored):
                    continue
                indices = torch.nonzero(coords[:, 0] == index, as_tuple=False).flatten()
                points = coords[indices, 1:4].unsqueeze(0).contiguous()
                inside = points_in_boxes_gpu(points, ignored.unsqueeze(0).contiguous())[0] >= 0
                if len(positive):
                    inside &= points_in_boxes_gpu(points, positive.unsqueeze(0).contiguous())[0] < 0
                masked=indices[inside]
                model._annoloop_ignore_counts['point']+=int((targets['point_cls_labels'][masked]!=-1).sum().item())
                targets['point_cls_labels'][masked] = -1
            return targets

        model.point_head.assign_targets = assign_points
