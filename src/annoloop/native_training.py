"""Source-assisted target fine-tuning on one frozen shared pseudo-label file."""
import argparse
import logging
from pathlib import Path
import sys
import numpy as np
from .io import read, write, digest, now
from .native_data import make_target_dataset
from .predictions import assert_no_gold
from .runtime import configure_sparse_headers


def prepare(spec):
    if digest(spec['path']) != spec['checkpoint_sha256'] or digest(spec['config']) != spec['config_sha256']:
        raise ValueError('Checkpoint/config đã thay đổi')
    configure_sparse_headers()
    runtime = Path(spec['runtime'])
    sys.path[:0] = [str(runtime), str(runtime / 'tools')]
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.models import build_network
    cfg_from_yaml_file(spec['config'], cfg)
    if cfg.get('SELF_TRAIN') or cfg.get('DATA_CONFIG_TAR'):
        raise ValueError('Adapter nhận profile nguồn XYZ; DSNorm legacy chưa được hỗ trợ')
    if list(cfg.CLASS_NAMES) != spec['classes']:
        raise ValueError('Class order không khớp checkpoint/config')
    dataset = NuScenesDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=True, logger=logging.getLogger('adaptation'))
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    state = torch.load(spec['path'], map_location='cpu')
    if state.get('epoch', 0) <= 0 or state.get('it', 0) <= 0:
        raise ValueError('Checkpoint chưa có optimizer steps')
    if any(not torch.isfinite(v).all() for v in state['model_state'].values() if torch.is_floating_point(v)):
        raise ValueError('Weights NaN/Inf')
    _, loaded = model._load_state_dict(state['model_state'], strict=False)
    missing = [k for k in model.state_dict() if k not in loaded and not k.endswith('num_batches_tracked')]
    if missing:
        raise ValueError('Checkpoint không khớp model: ' + str(missing[:10]))
    from .native_ignore import install
    install(model)
    return torch, cfg, dataset, model.cuda()


def verify(root):
    """Real source/target inference and backward; zero optimizer steps."""
    root = Path(root)
    spec = read(root / 'spec.json')
    # The normal inference worker verifies native separate scores too.
    from .inspection_worker import run as infer
    infer(root)
    target_root = root / 'target_predict'
    target_spec = dict(spec, mode='target_sample', frames=spec['frames'][:1], limit=1)
    write(target_root / 'spec.json', target_spec)
    write(target_root / 'inspection.json', dict(status='running'))
    infer(target_root)
    target_result = read(target_root / 'result.json')
    prediction = read(target_root / target_result['frames'][0]['file'])
    if not prediction['boxes']:
        raise ValueError('Sample target không có box để kiểm tra backward adapter; xem predict sample trước')
    # These real predictions exercise the data/loss path only. No optimizer or
    # pseudo-label selection is performed and no labels are promoted to a round.
    diagnostic_labels = dict(target_gold_read=False, diagnostic_only=True, frames=[dict(
        frame_id=prediction['frame_id'], boxes=[dict(b, training_policy='positive') for b in prediction['boxes']],
        ignored=[])])
    torch, cfg, dataset, model = prepare(spec)
    from pcdet.models import load_data_to_gpu
    row = dataset[0]
    truth = row['gt_boxes']
    if not len(truth):
        raise ValueError('Source diagnostic frame không có nhãn')
    # Add a class-specific signed ignore away from the positives. Only source
    # labels are involved in this diagnostic, never target gold.
    ignored = truth[:1].copy()
    ignored[:, 0] += 15
    ignored[:, -1] *= -1
    row['gt_boxes'] = np.concatenate([truth, ignored])
    batch = dataset.collate_batch([row])
    load_data_to_gpu(batch)
    model.train()
    loss = model(batch)[0]['loss'].mean()
    if not torch.isfinite(loss):
        raise ValueError('Diagnostic loss không hữu hạn')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Diagnostic gradients NaN/Inf')
    source_loss = float(loss.detach().cpu())
    del loss, batch
    model.zero_grad(set_to_none=True)
    target_dataset = make_target_dataset(cfg.DATA_CONFIG, list(cfg.CLASS_NAMES), spec['frames'][:1],
                                        spec['bin_columns'], diagnostic_labels)
    target_batch = target_dataset.collate_batch([target_dataset[0]])
    load_data_to_gpu(target_batch)
    target_loss = model(target_batch)[0]['loss'].mean()
    if not torch.isfinite(target_loss):
        raise ValueError('Target diagnostic loss NaN/Inf')
    target_loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Target diagnostic gradients NaN/Inf')
    write(root / 'verified.json', dict(verified_at=now(), checkpoint_sha256=spec['checkpoint_sha256'],
          config_sha256=spec['config_sha256'], architecture=cfg.MODEL.NAME,
          separate_scores_checked=True, forward_backward_checked=True,
          signed_ignore_hooks_installed=True, optimizer_steps=0, target_gold_read=False,
          diagnostic_domain='source_train_validation_and_unlabelled_target',
          source_loss=source_loss, target_loss=float(target_loss.detach().cpu()),
          target_adapter_backward_checked=True, diagnostic_predictions_not_promoted=True))
    print('Native inference/backward verified; 0 optimizer steps.', flush=True)


def restore_optimizer(torch, spec, optimizer, epochs):
    resume = spec['policy'].get('resume_checkpoint')
    if not resume:
        return 0, 0
    if digest(spec['path']) != resume['sha256']:
        raise ValueError('Checkpoint resume không khớp')
    state = torch.load(spec['path'], map_location='cpu')
    epoch, steps = int(state['epoch']), int(state['it'])
    if not 0 < epoch <= epochs or epoch != resume['epoch'] or steps <= 0 or not state.get('optimizer_state'):
        raise ValueError('Checkpoint thiếu epoch/optimizer state hợp lệ để tiếp tục')
    optimizer.load_state_dict(state['optimizer_state'])
    return epoch, steps


def train(root):
    root = Path(root)
    spec = read(root / 'spec.json')
    labels_path = Path(spec['labels_path'])
    if digest(labels_path) != spec['shared_labels_sha256']:
        raise ValueError('Shared labels đã thay đổi')
    labels = read(labels_path)
    assert_no_gold(labels)
    if labels.get('target_gold_read') is not False:
        raise ValueError('Shared labels chưa xác nhận không dùng target gold')
    torch, cfg, source, model = prepare(spec)
    from pcdet.models import load_data_to_gpu
    from train_utils.optimization import build_optimizer, build_scheduler
    from train_utils.train_utils import checkpoint_state, save_checkpoint
    from pcdet.utils.common_utils import set_random_seed
    set_random_seed(spec.get('seed', 42))
    target = make_target_dataset(cfg.DATA_CONFIG, list(cfg.CLASS_NAMES), spec['frames'],
                                 spec['bin_columns'], labels, logging.getLogger('target'))
    epochs = spec['policy']['epochs']
    if type(epochs) is not int or epochs < 1:
        raise ValueError('Epoch target phải > 0')
    cfg.OPTIMIZATION.NUM_EPOCHS = epochs
    cfg.OPTIMIZATION.LR = spec['policy'].get('learning_rate', .0001)
    # Fixed Adam LR for short rounds; avoid OneCycle degenerate 1-step phases.
    cfg.OPTIMIZATION.OPTIMIZER = 'adam'
    cfg.OPTIMIZATION.DECAY_STEP_LIST = []
    cfg.OPTIMIZATION.LR_WARMUP = False
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    start_epoch, start_steps = restore_optimizer(torch, spec, optimizer, epochs)
    scheduler, _ = build_scheduler(optimizer, len(target), epochs, -1, cfg.OPTIMIZATION)
    checkpoint_dir = root / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    progress = dict(status='running', epochs=epochs, epoch_completed=start_epoch, optimizer_steps=start_steps,
                    resumed_epoch=start_epoch, resumed_optimizer_steps=start_steps,
                    target_frames=len(target), skipped_target_frames=len(spec['frames']) - len(target),
                    source_loss_weight=spec['policy'].get('source_weight', .5), target_gold_used=False)
    write(root / 'progress.json', progress)
    steps = start_steps
    source_index = start_steps
    source_order = np.random.permutation(len(source))
    for epoch in range(start_epoch + 1, epochs + 1):
        for index in np.random.permutation(len(target)):
            scheduler.step(steps)
            optimizer.zero_grad()
            model.train()
            if source_index and source_index % len(source) == 0:
                source_order = np.random.permutation(len(source))
            source_batch = source.collate_batch([source[int(source_order[source_index % len(source)])]])
            source_index += 1
            load_data_to_gpu(source_batch)
            source_loss = model(source_batch)[0]['loss'].mean()
            if not torch.isfinite(source_loss):
                raise ValueError('Source loss NaN/Inf')
            (progress['source_loss_weight'] * source_loss).backward()
            source_value = float(source_loss.detach().cpu())
            del source_loss, source_batch
            batch = target.collate_batch([target[int(index)]])
            load_data_to_gpu(batch)
            target_loss = model(batch)[0]['loss'].mean()
            if not torch.isfinite(target_loss):
                raise ValueError('Target loss NaN/Inf')
            target_loss.backward()
            target_value = float(target_loss.detach().cpu())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            if not torch.isfinite(grad_norm):
                raise ValueError('Gradient NaN/Inf; chưa optimizer step')
            optimizer.step()
            model.update_global_step()
            steps += 1
            del target_loss, batch
            loss = progress['source_loss_weight'] * source_value + target_value
            progress.update(optimizer_steps=steps, epoch=epoch, loss=loss,
                            source_loss=source_value, target_loss=target_value)
            write(root / 'progress.json', progress)
            print(f'Epoch [{epoch-1}/{epochs}] total_it={steps} loss={loss:.5f} source={source_value:.5f} target={target_value:.5f}', flush=True)
        if epoch % 10 == 0 or epoch == epochs:
            save_checkpoint(checkpoint_state(model, optimizer, epoch, steps),
                            filename=str(checkpoint_dir / f'checkpoint_epoch_{epoch}'))
        progress.update(epoch_completed=epoch)
        write(root / 'progress.json', progress)
        print(f'Epoch [{epoch}/{epochs}] completed', flush=True)
    final = checkpoint_dir / f'checkpoint_epoch_{epochs}.pth'
    if not final.exists() and start_epoch == epochs:
        save_checkpoint(checkpoint_state(model, optimizer, epochs, steps),filename=str(final.with_suffix('')))
    if digest(labels_path) != spec['shared_labels_sha256']:
        raise ValueError('Shared labels thay đổi trong lúc train')
    progress.update(status='completed')
    write(root / 'progress.json', progress)
    write(root / 'trained.json', dict(checkpoint=str(final), checkpoint_sha256=digest(final),
          shared_labels_sha256=spec['shared_labels_sha256'], optimizer_steps=steps, epochs=epochs,
          target_gold_used=False, source_assisted=True, source_loss_weight=progress['source_loss_weight'],
          classes=list(cfg.CLASS_NAMES), architecture=cfg.MODEL.NAME, finished_at=now(),
          resumed_epoch=start_epoch, resumed_optimizer_steps=start_steps, new_optimizer_steps=steps-start_steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify(args.run) if args.verify else train(args.run)
