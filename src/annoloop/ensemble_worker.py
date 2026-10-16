"""Explicit N-checkpoint inference worker; no optimizer, no target gold."""
import argparse
from pathlib import Path
import os
import signal
from .io import read, write, digest, now, lock
from .settings import Settings
from .reused.training_monitor import process_identity


def stage(root, value, **extra):
    state = read(root / 'state.json')
    write(root / 'state.json', dict(state, stage=value, **extra))
    print(value, extra, flush=True)


def run(root, settings):
    from .projects import ProjectStore
    from .native_adapter import NativeDetectorAdapter
    from .device import gpu_idle
    from .inspection import load_xyz
    from .fusion import calibrate_source, normalize, compare
    from .ensemble_fusion import fuse_frame, EnsemblePolicy
    from .cvat import package
    from .schema import ClassSchema
    root = Path(root)
    spec = read(root / 'spec.json')
    store = ProjectStore(settings.data); pid = spec['project_id']
    with lock(store.path(pid) / '.adaptation_dispatch.lock'):
        pass
    if digest(store.path(pid) / 'dataset.json') != spec['dataset_sha256']:
        raise ValueError('Dataset thay đổi sau dispatch')
    for item in spec['source_evidence'].values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Source infos thay đổi sau dispatch')
    inventory = store.verify_dataset(pid)
    delivery = spec.get('purpose') == 'final_delivery'
    if delivery:
        from .final_delivery import SCOPES
        if set(spec['frame_splits']) != SCOPES.get(spec['scope']):
            raise ValueError('Phạm vi final delivery không hợp lệ')
        frames = [f for f in inventory['frames'] if f['split'] in spec['frame_splits']]
        if [f['frame_id'] for f in frames] != spec['frame_ids']:
            raise ValueError('Frame bàn giao thay đổi')
        reference=spec['source_snapshot']
        if digest(reference['path'])!=reference['sha256']:
            raise ValueError('Phiên bản model đã chọn thay đổi')
    else:
        # Held-out frames never enter pseudo-label generation.
        frames = [f for f in inventory['frames'] if f['split'] == 'adaptation']
    config = spec['config']; classes = config['classes']
    policy = EnsemblePolicy(**config['policy'])
    providers, documents = {}, {}
    with lock(settings.data / '.gpu.lock', blocking=False):
        for index, model in enumerate(config['models']):
            key = f'model_{index + 1:03d}'
            if digest(model['path']) != model['checkpoint_sha256'] or digest(model['config']) != model['config_sha256']:
                raise ValueError('Checkpoint/config thay đổi sau dispatch')
            supported = [c for c in classes if c in model['classes']]
            if not supported:
                raise ValueError('Model đã chọn không hỗ trợ class đang chạy')
            gpu_idle()
            adapter = NativeDetectorAdapter(model, supported, frames, root / key, settings, config['bin_columns'])
            stage(root, 'source_validation', model=key, architecture=model['architecture'],
                  model_index=index + 1, model_total=len(config['models']))
            records = adapter.predict_source_validation(model['path'])
            calibration = calibrate_source(records, model['checkpoint_sha256'], supported)
            write(root / key / 'calibration.json', calibration)
            gpu_idle()
            stage(root, 'target_predict', model=key)
            predictions = adapter.predict_target(model['path'], frames)
            if set(predictions) != {f['frame_id'] for f in frames}:
                raise ValueError('Chưa đủ frame target; không hợp nhất kết quả từng phần')
            documents[key] = {fid: dict(boxes=normalize(doc, calibration,
                model['checkpoint_sha256'], supported)) for fid, doc in predictions.items()}
            for doc in documents[key].values():
                for box in doc['boxes']:
                    box['native_box_id']=box['id']
                    box['id']=key+':'+str(box['id'])
            write(root / key / 'calibrated_predictions.json', documents[key])
            providers[key] = dict(model, classes=supported)
            gpu_idle()
        # Recheck identities before publishing a complete result.
        if digest(store.path(pid) / 'dataset.json') != spec['dataset_sha256']:
            raise ValueError('Dataset thay đổi khi chạy')
        stage(root, 'fusing', model=None)
        outputs = []
        for index, frame in enumerate(frames):
            points = load_xyz(frame['media'], config['bin_columns'])
            if digest(frame['media']) != frame['sha256']:
                raise ValueError('Media thay đổi khi chạy ensemble')
            if config.get('fusion_mode')=='ab':
                from .reused.core import points_in_box
                if len(documents)!=2:
                    raise ValueError('Final A/B cần đúng hai detector')
                first,second=documents.values()
                positive,ignored=compare(first[frame['frame_id']]['boxes'],second[frame['frame_id']]['boxes'],
                                         policy.match_iou,policy.confidence,policy.quality)
                accepted=[]
                for box in positive:
                    box['point_count']=int(points_in_box(points,box['box']).sum())
                    box['origin']='final_ab'
                    if box['point_count']>=policy.min_points:accepted.append(box)
                    else:ignored.append(dict(box,training_policy='ignore_region',reason='insufficient_points'))
                doc=dict(boxes=accepted,ignored=ignored,target_gold_read=False)
            else:
                doc = fuse_frame({key: values[frame['frame_id']] for key, values in documents.items()},
                                 providers, classes, points, policy)
            doc.update(frame_id=frame['frame_id'], sequence_id=frame.get('sequence_id'))
            relative = f'frames/{index:06d}.json'
            write(root / relative, doc)
            outputs.append(dict(frame_id=frame['frame_id'], file=relative,
                                boxes=len(doc['boxes']), ignored=len(doc['ignored'])))
            stage(root, 'fusing', frame_index=index + 1, frame_total=len(frames))
        schema = ClassSchema.parse(dict(data_type='3d', annotation_type='cuboid', classes=classes))
        for item in spec['source_evidence'].values():
            if digest(item['path']) != item['sha256']:
                raise ValueError('Source infos thay đổi trước khi xuất kết quả')
        package_path = root / 'cvat_shapes.json'
        write(package_path, package([read(root / row['file']) for row in outputs], schema))
        if delivery:
            from .final_delivery import prepare_export
            stage(root,'packing_media')
            prepare_export(root,inventory,[read(root/row['file']) for row in outputs],schema,config['bin_columns'])
        result = dict(frames=outputs, providers=providers, class_names=classes,
                      dataset_sha256=spec['dataset_sha256'], target_gold_read=False, optimizer_steps=0,
                      scope=spec['scope'] if delivery else 'adaptation_only', policy=config['policy'],
                      cvat_package=str(package_path), cvat_sha256=digest(package_path),
                      positive_count=sum(f['boxes'] for f in outputs),
                      ignored_count=sum(f['ignored'] for f in outputs))
        if delivery:
            result.update(purpose='final_delivery',version=spec['version'],frame_ids=spec['frame_ids'],
                          frame_splits=spec['frame_splits'],not_for_training=True,
                          export_manifest_sha256=digest(root/'exports/shape/export.json'))
        write(root / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True)
    root = Path(parser.parse_args().run)
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Ensemble interrupted')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    try:
        # Dispatch writes its receipt before releasing this lock.
        spec = read(root / 'spec.json'); settings = Settings.load()
        from .projects import ProjectStore
        with lock(ProjectStore(settings.data).path(spec['project_id']) / '.adaptation_dispatch.lock'):
            stage(root, 'starting', status='running', pid=os.getpid(),
                  process_start=process_identity(os.getpid()), started_at=now())
        run(root, settings)
        stage(root, 'finished', status='completed', finished_at=now(),result_sha256=digest(root/'result.json'))
    except BaseException as error:
        stage(root, 'stopped' if isinstance(error, KeyboardInterrupt) else 'failed',
              status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
              error=f'{type(error).__name__}: {error}', finished_at=now())
        raise


if __name__ == '__main__':
    main()
