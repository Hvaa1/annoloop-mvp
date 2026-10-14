"""Source-only ST3D worker. Requires prepared source and a compiled ST3D runtime."""
import argparse
from pathlib import Path
import sys
import yaml
from collections import Counter
from .compatibility import checkpoint_metadata
from .inventory import source_counts
from .io import digest, read, write
from .reused.checkpoint_policy import with_final_checkpoint
from .schema import ClassSchema, discover


def selected_source_inventory(path, selected_schema, expected_split):
    """Validate a full source taxonomy, then require the classes being trained."""
    path = Path(path)
    if any((path.parent / name).is_file() for name in ("class_schema.json", "manifest.json", "dataset.json")):
        source_schema, _ = discover(path.parent)
    else:
        source_schema = selected_schema
    if source_schema.data_type != selected_schema.data_type:
        raise ValueError("Source schema khác loại dữ liệu của model")
    if not set(selected_schema.classes).issubset(source_schema.classes):
        raise ValueError("Source schema thiếu class cần train")
    inventory = source_counts(path, source_schema)
    if inventory["split"] != expected_split or any(inventory["counts"][c] == 0 for c in selected_schema.classes):
        raise ValueError("Source train/validation thiếu nhãn class: " + expected_split)
    return {**inventory, "selected_counts": {c: inventory["counts"][c] for c in selected_schema.classes}}


def actual_source_classes(dataset, classes):
    infos = getattr(dataset, "infos", getattr(dataset, "kitti_infos", None))
    if infos is None:
        raise ValueError("Dataset adapter chưa hỗ trợ kiểm tra nhãn source thực trong infos")
    counts = Counter()
    for info in infos:
        names = info.get("gt_names", info.get("annos", {}).get("name", []))
        counts.update(str(name) for name in names)
    if any(counts[name] == 0 for name in classes):
        raise ValueError("Infos source thực thiếu nhãn class cần train/validation")
    return {name: counts[name] for name in classes}


def run(root):
    root = Path(root).absolute()
    job = read(root / "job.json")
    spec = job["spec"]
    config = yaml.safe_load(Path(spec["config"]).read_text())
    if config.get("SELF_TRAIN") or config.get("DATA_CONFIG_TAR"):
        raise ValueError("Worker nguồn không nhận config target/self-training")
    schema = ClassSchema.parse(dict(data_type="3d", annotation_type="cuboid", classes=config["CLASS_NAMES"]))
    for key, expected in (("source_train_annotations", "train"), ("source_validation_annotations", "validation")):
        selected_source_inventory(spec[key], schema, expected)
    runtime = Path(spec["runtime"]).resolve()
    sys.path[:0] = [str(runtime), str(runtime / "tools")]
    from .runtime import configure_sparse_headers
    configure_sparse_headers()
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, model_fn_decorator
    from pcdet.utils import common_utils
    from train_utils.optimization import build_optimizer, build_scheduler
    from train_utils.train_utils import train_model, checkpoint_state, save_checkpoint
    cfg_from_yaml_file(spec["config"], cfg)
    cfg.LOCAL_RANK = 0
    epochs = cfg.OPTIMIZATION.NUM_EPOCHS
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Epoch source không hợp lệ")
    common_utils.set_random_seed(spec.get("seed", 42))
    logger = common_utils.create_logger(root / "source_worker.log", rank=0)
    _, loader, sampler = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                           batch_size=1, dist=False, workers=0, logger=logger, training=True)
    _, validation_loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                           batch_size=1, dist=False, workers=0, logger=logger, training=False)
    actual_counts = dict(train=actual_source_classes(loader.dataset, schema.classes),
                         validation=actual_source_classes(validation_loader.dataset, schema.classes))
    write(root / "actual_source_inventory.json", actual_counts)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(schema.classes), dataset=loader.dataset)
    transfer = {}
    if spec.get("checkpoint_metadata"):
        metadata = checkpoint_metadata(spec["checkpoint_metadata"])
        checkpoint = torch.load(metadata["checkpoint"], map_location="cpu")
        weights = checkpoint["model_state"]
        same_head = metadata["classes"] == list(schema.classes) and metadata.get("architecture") == cfg.MODEL.NAME
        if not same_head:
            weights = {key: value for key, value in weights.items()
                       if not key.startswith(("dense_head.", "roi_head.", "point_head."))}
        _, loaded = model._load_state_dict(weights, strict=False)
        if not loaded:
            raise ValueError("Không nạp được weights tương thích")
        transfer = dict(initial_checkpoint_sha256=metadata["checkpoint_sha256"],
                        head_reinitialized=not same_head, loaded_keys=len(loaded))
    model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    scheduler, warmup = build_scheduler(optimizer, total_iters_each_epoch=len(loader), total_epochs=epochs,
                                       last_epoch=-1, optim_cfg=cfg.OPTIMIZATION)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir()
    trainer = with_final_checkpoint(train_model, checkpoint_state, save_checkpoint)
    trainer(model=model, optimizer=optimizer, train_loader=loader, target_loader=None,
            model_func=model_fn_decorator(), lr_scheduler=scheduler, optim_cfg=cfg.OPTIMIZATION,
            start_epoch=0, total_epochs=epochs, start_iter=0, rank=0, tb_log=None,
            ckpt_save_dir=checkpoint_dir, ps_label_dir=root / "unused_pseudo",
            source_sampler=sampler, lr_warmup_scheduler=warmup, ckpt_save_interval=10,
            max_ckpt_save_num=max(2, epochs // 10 + 1), logger=logger)
    final = checkpoint_dir / f"checkpoint_epoch_{epochs}.pth"
    state = torch.load(final, map_location="cpu")
    if state["epoch"] != epochs or state["it"] <= 0:
        raise ValueError("Checkpoint chưa có optimizer steps/epoch cuối")
    write(root / "checkpoint.metadata.json", dict(checkpoint=str(final), checkpoint_sha256=digest(final),
          classes=list(schema.classes), classes_verified=True, namespace="annoloop:" + job["id"],
          architecture=cfg.MODEL.NAME, config_sha256=digest(spec["config"]),
          target_gold_read=False, source_only=True, epochs=epochs, optimizer_steps=state["it"], transfer=transfer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    run(parser.parse_args().job)
