"""N-model contracts; fixtures only, never real GPU inference/training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from annoloop.ensemble_fusion import EnsemblePolicy, fuse_frame
from annoloop import local_ensemble as service
from annoloop.io import read, write, digest
from annoloop.projects import ProjectStore
from annoloop.settings import Settings
from annoloop.schema import ClassSchema


def box(name='car', identity='a', x=0., confidence=.95):
    return dict(id=identity, class_name=name, box=[x,0.,0.,4.,2.,2.,0.],
                score=.9, cls_score=.9, iou_score=.9, confidence=confidence, quality=.8)


def provider(identity, group, classes=('car',)):
    return dict(checkpoint_sha256=identity, vote_group=group, classes=list(classes))


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.points=np.zeros((6,3),dtype=np.float32)

    def test_class_specialists_do_not_vote_against_another_class(self):
        providers={'a':provider('1','SECOND',('car',)), 'b':provider('2','PV',('pedestrian',))}
        docs={'a':dict(boxes=[box()]),'b':dict(boxes=[box('pedestrian',x=10.)])}
        points=np.concatenate([self.points,self.points+[10,0,0]])
        result=fuse_frame(docs,providers,['car','pedestrian'],points)
        self.assertEqual(len(result['boxes']),2)
        self.assertTrue(all(b['available_groups']==1 and b['required_votes']==1 for b in result['boxes']))

    def test_same_architecture_snapshots_do_not_overwhelm_second_detector(self):
        providers={str(i):provider(str(i),'SECOND') for i in range(8)}
        providers['b']=provider('b','PV')
        docs={key:dict(boxes=[box(identity=key,confidence=.8)]) for key in providers}
        docs['b']=dict(boxes=[])
        result=fuse_frame(docs,providers,['car'],self.points)
        self.assertFalse(result['boxes'])
        self.assertEqual(result['ignored'][0]['independent_votes'],1)
        self.assertEqual(result['ignored'][0]['available_groups'],2)

    def test_independent_agreement_keeps_real_winner_geometry_and_provenance(self):
        providers={'a':provider('1','SECOND'),'b':provider('2','PV')}
        b=box(identity='b',x=.1);b['quality']=.9
        result=fuse_frame({'a':dict(boxes=[box()]),'b':dict(boxes=[b])},providers,['car'],self.points)
        self.assertEqual(result['boxes'][0]['box'],b['box'])
        self.assertEqual(len(result['boxes'][0]['evidence']),2)

    def test_duplicate_hash_and_missing_frame_are_rejected(self):
        docs={'a':dict(boxes=[]),'b':dict(boxes=[])}
        with self.assertRaisesRegex(ValueError,'trùng hash'):
            fuse_frame(docs,{'a':provider('same','x'),'b':provider('same','y')},['car'],self.points)
        with self.assertRaisesRegex(ValueError,'Thiếu kết quả'):
            fuse_frame({'a':docs['a']},{'a':provider('1','x'),'b':provider('2','y')},['car'],self.points)

    def test_single_group_stricter_threshold_and_real_points(self):
        providers={'a':provider('1','x')}
        result=fuse_frame({'a':dict(boxes=[box(confidence=.8)])},providers,['car'],self.points)
        self.assertFalse(result['boxes'])
        result=fuse_frame({'a':dict(boxes=[box()])},providers,['car'],np.empty((0,3)))
        self.assertFalse(result['boxes'])

    def test_cross_class_conflict_is_ignore_and_uncalibrated_scores_rejected(self):
        providers={'a':provider('1','x',('car','pedestrian'))}
        result=fuse_frame({'a':dict(boxes=[box(),box('pedestrian',identity='p')])},providers,['car','pedestrian'],self.points)
        self.assertFalse(result['boxes'])
        self.assertTrue(all(b['reason']=='cross_class_conflict' for b in result['ignored']))
        raw=box();del raw['quality']
        with self.assertRaisesRegex(ValueError,'calibration'):
            fuse_frame({'a':dict(boxes=[raw])},providers,['car'],self.points)

    def test_invalid_policy_rejected(self):
        for kwargs in ({'match_iou':float('nan')},{'min_points':0},{'singleton_confidence':.2}):
            with self.assertRaises(ValueError):EnsemblePolicy(**kwargs)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.data=self.root/'state'
        target=self.root/'target';target.mkdir()
        np.save(target/'frame.npy',np.zeros((6,3),dtype=np.float32))
        write(target/'class_schema.json',ClassSchema('3d','cuboid',('car',)).as_dict())
        self.store=ProjectStore(self.data);self.pid=self.store.create('ensemble test',target)
        self.settings=Settings(self.data,Path('/fake/python'),Path('/fake/launcher'))
        source=self.root/'source'/'v1.0-mini';source.mkdir(parents=True)
        self.evidence={}
        for split in ('train','validation'):
            path=source/(split+'.pkl');path.write_bytes(b'source info fixture never unpickled')
            self.evidence[split]=dict(path=str(path),sha256=digest(path))
        write(self.store.path(self.pid)/'setup.json',dict(source_evidence=self.evidence))
        self.cfg=dict(CLASS_NAMES=['car'],MODEL=dict(NAME='SECONDNetIoU'),DATA_CONFIG=dict(
            DATASET='NuScenesDataset',DATA_PATH=str(source.parent),VERSION='v1.0-mini',
            INFO_PATH={'train':['train.pkl'],'test':['validation.pkl']},
            POINT_FEATURE_ENCODING=dict(used_feature_list=['x','y','z'])))
        config=self.store.path(self.pid)/'generated/A.yaml';config.parent.mkdir(parents=True)
        config.write_text(yaml.safe_dump(self.cfg));self.config=config
        runtime=self.root/'runtime';runtime.mkdir()
        self.model=dict(config=str(config),runtime=str(runtime))
        self.weight=self.data/'jobs/a/checkpoints/checkpoint_epoch_10.pth'
        self.weight.parent.mkdir(parents=True);self.weight.write_bytes(b'fixture checkpoint; never loaded')
        write(self.data/'jobs/a/job.json',dict(kind='source_pretrain',spec=self.model))

    def test_discovery_shows_imported_but_does_not_grant_native_capability(self):
        root=self.data/'model_imports/st3d/secondiou_st3d';root.mkdir(parents=True)
        (root/'weights.pth').write_bytes(b'not a real checkpoint')
        write(root/'provenance.json',dict(name='official fixture',class_names_from_config=['Vehicle']))
        with patch('subprocess.Popen') as dispatch:
            rows=service.discover(self.store,self.pid,self.settings)
            self.assertEqual(len(rows),2)
            self.assertTrue(next(m for m in rows if m['path']==str(self.weight))['eligible'])
            imported=next(m for m in rows if m['origin'].startswith('ST3D'))
            self.assertFalse(imported['eligible'])
            dispatch.assert_not_called()
        with self.assertRaisesRegex(ValueError,'chưa đủ điều kiện'):
            service.save(self.store,self.pid,self.settings,[imported['path']],['car'],True)

    def test_save_deduplicates_hash_and_preserves_ab_history(self):
        copy=self.weight.with_name('checkpoint_epoch_20.pth');copy.write_bytes(self.weight.read_bytes())
        ab=self.store.path(self.pid)/'adaptation.json';write(ab,dict(models={},sentinel='keep'))
        before=ab.read_bytes()
        with patch('annoloop.adaptation.active_job',return_value=None):
            config=service.save(self.store,self.pid,self.settings,[str(self.weight),str(copy)],['car'],True)
            self.assertEqual(len(config['models']),1)
            service.disable(self.store,self.pid)
        self.assertEqual(before,ab.read_bytes())
        self.assertFalse(service.configuration(self.store,self.pid)['enabled'])

    def test_source_mismatch_and_live_job_block_configuration(self):
        self.cfg['DATA_CONFIG']['DATA_PATH']='/different/source'
        self.config.write_text(yaml.safe_dump(self.cfg))
        self.assertFalse(service.discover(self.store,self.pid,self.settings)[0]['eligible'])
        self.cfg['DATA_CONFIG']['DATA_PATH']=str(self.root/'source')
        self.config.write_text(yaml.safe_dump(self.cfg))
        with patch('annoloop.adaptation.active_job',return_value='live'):
            with self.assertRaisesRegex(ValueError,'Chờ job'):
                service.save(self.store,self.pid,self.settings,[str(self.weight)],['car'],True)

    def test_worker_sequential_calibration_and_publish_without_training(self):
        from annoloop.ensemble_worker import run
        with patch('annoloop.adaptation.active_job',return_value=None):
            config=service.save(self.store,self.pid,self.settings,[str(self.weight)],['car'],True)
        root=self.store.path(self.pid)/'ensemble_jobs/fixture'
        write(root/'spec.json',dict(project_id=self.pid,config=config,source_evidence=self.evidence,
              dataset_sha256=digest(self.store.path(self.pid)/'dataset.json')))
        write(root/'state.json',dict(status='running'))
        events=[]
        class FakeAdapter:
            def __init__(self,*args):pass
            def predict_source_validation(self,checkpoint):
                events.append('source')
                return dict(domain='source_validation',boxes=[dict(box(),source_gt_iou=.9) for _ in range(10)])
            def predict_target(adapter,checkpoint,frames):
                events.append('target')
                return {f['frame_id']:dict(boxes=[box()]) for f in frames}
            def train_target(self,*args):raise AssertionError('Ensemble must not train')
        with patch('annoloop.native_adapter.NativeDetectorAdapter',FakeAdapter),patch('annoloop.device.gpu_idle'):
            run(root,self.settings)
        self.assertEqual(events,['source','target'])
        result=read(root/'result.json')
        self.assertEqual(result['positive_count'],1)
        self.assertEqual(result['optimizer_steps'],0)
        self.assertFalse(result['target_gold_read'])
        self.assertTrue(Path(result['cvat_package']).exists())
        self.assertFalse(self.store.get(self.pid)['completed_rounds'])
