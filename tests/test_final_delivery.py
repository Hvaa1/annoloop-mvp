"""Final checkpoint/split/media/upload contracts; no real GPU or CVAT request."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import yaml
from annoloop import final_delivery as delivery
from annoloop.io import read,write,digest
from annoloop.projects import ProjectStore
from annoloop.settings import Settings
from annoloop.schema import ClassSchema
from annoloop.ensemble_worker import run


def box():
    return dict(id='0',class_name='car',box=[0.,0.,0.,4.,2.,2.,0.],score=.9,
                cls_score=.9,iou_score=.9,confidence=.95,quality=.9)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);target=self.root/'target';target.mkdir()
        write(target/'class_schema.json',ClassSchema('3d','cuboid',('car',)).as_dict())
        rows=[]
        for name,split in [('z','adaptation'),('a','holdout'),('b','delivery')]:
            np.save(target/(name+'.npy'),np.zeros((8,3),dtype=np.float32))
            rows.append(dict(frame_id=name,points=name+'.npy',split=split))
        write(target/'manifest.json',dict(frames=rows))
        (target/'gold.json').write_text('must not read target gold')
        self.data=self.root/'state';self.store=ProjectStore(self.data)
        self.pid=self.store.create('delivery fixture',target)
        project=self.store.path(self.pid)
        self.settings=Settings(self.data,Path('/fake/python'),Path('/fake/launcher'))
        evidence={}
        for split in ('train','validation'):
            path=self.root/(split+'.pkl');path.write_bytes(b'source fixture not deserialized')
            evidence[split]=dict(path=str(path),sha256=digest(path))
        write(project/'setup.json',dict(source_evidence=evidence))
        models={};final={}
        for key,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]:
            cfg=project/(key+'.yaml');cfg.write_text(yaml.safe_dump(dict(CLASS_NAMES=['car'],MODEL=dict(NAME=arch))))
            initial=self.root/(key+'_initial.pth');initial.write_bytes(('initializer '+key).encode())
            final[key]=self.root/(key+'_final.pth');final[key].write_bytes(('trained final '+key).encode())
            models[key]=dict(path=str(initial),config=str(cfg),config_sha256=digest(cfg),
                checkpoint_sha256=digest(initial),architecture=arch,classes=['car'],runtime=str(self.root))
        self.models=models
        write(project/'adaptation.json',dict(models=models,coordinates_confirmed=True,bin_columns=4,adaptation_classes=['car']))
        pred=dict(target_gold_read=False,frames=[dict(frame_id='z',boxes=[box()])])
        self.store.record_round(self.pid,1,pred,
            dict(domain='target_unlabelled_proxy',target_gold_read=False,agreement=1.,supported_fraction=1.,box_count=1),
            final,{'z':dict(boxes=[box()])})
        self.store.select(self.pid,'M1')

    def job(self,scope='all'):
        spec=delivery.plan(self.store,self.pid,scope)
        root=self.store.path(self.pid)/'delivery_jobs'/('fixture_'+scope)
        write(root/'spec.json',spec);write(root/'state.json',dict(status='running'))
        return root

    def execute(self,root,missing=False):
        events=[]
        class Adapter:
            def __init__(self,model,*args):self.model=model
            def predict_source_validation(adapter,path):
                events.append(('source',path))
                return dict(domain='source_validation',boxes=[dict(box(),source_gt_iou=.9) for _ in range(10)])
            def predict_target(adapter,path,frames):
                events.append(('target',[f['frame_id'] for f in frames]))
                return {f['frame_id']:dict(boxes=[box()]) for f in (frames[:-1] if missing else frames)}
            def train_target(*args):raise AssertionError('Final delivery must never train')
        with patch('annoloop.native_adapter.NativeDetectorAdapter',Adapter),patch('annoloop.device.gpu_idle'):
            run(root,self.settings)
        write(root/'state.json',dict(status='completed',result_sha256=digest(root/'result.json')))
        return events

    def test_plan_uses_selected_round_not_initializer_and_explicit_scope(self):
        with patch('subprocess.Popen') as popen:
            spec=delivery.plan(self.store,self.pid)
            self.assertEqual(spec['version'],'M1')
            self.assertEqual(spec['frame_ids'],['z','b'])
            self.assertEqual(delivery.plan(self.store,self.pid,'delivery')['frame_ids'],['b'])
            self.assertEqual(delivery.plan(self.store,self.pid,'all')['frame_ids'],['z','a','b'])
            for model in spec['config']['models']:
                self.assertIn('/rounds/M1/',model['path'])
                self.assertNotIn(model['checkpoint_sha256'],[m['checkpoint_sha256'] for m in self.models.values()])
            popen.assert_not_called()

    def test_complete_final_predict_preserves_order_pcd_and_annotation_index(self):
        root=self.job();before=read(self.store.path(self.pid)/'project.json')
        events=self.execute(root)
        self.assertEqual([e[0] for e in events],['source','target','source','target'])
        self.assertEqual(events[1][1],['z','a','b'])
        directory,document=delivery.verified_export(root)
        self.assertEqual(document['frame_ids'],['z','a','b'])
        self.assertEqual([s['frame'] for s in document['shapes']],[0,1,2])
        self.assertFalse(document['tracks'])
        with zipfile.ZipFile(directory/'media.zip') as archive:
            self.assertEqual(archive.namelist(),['00000000.pcd','00000001.pcd','00000002.pcd'])
            self.assertIn(b'FIELDS x y z',archive.read('00000000.pcd'))
        result=read(root/'result.json')
        self.assertTrue(result['not_for_training']);self.assertFalse(result['target_gold_read'])
        self.assertEqual(before,read(self.store.path(self.pid)/'project.json'))

    def test_no_partial_export_on_missing_prediction(self):
        root=self.job()
        with self.assertRaisesRegex(ValueError,'Chưa đủ frame'):
            self.execute(root,missing=True)
        self.assertFalse((root/'result.json').exists())
        self.assertFalse((root/'exports').exists())

    def test_view_cache_is_read_only_invalidates_edits_and_upload_rechecks(self):
        import os
        root=self.job();self.execute(root)
        with patch.object(delivery,'verified_export',wraps=delivery.verified_export) as verify:
            _,document=delivery.preview_export(root)
            document['frame_ids'].clear()
            self.assertEqual(delivery.preview_export(root)[1]['frame_ids'],['z','a','b'])
            self.assertEqual(verify.call_count,1)
            with patch('annoloop.cvat.upload',return_value={}) as send:
                delivery.upload(root,'http://localhost:8080','fixture','fixture','task')
                send.assert_called_once()
            self.assertEqual(verify.call_count,2)
            path=root/'exports/shape/media.zip';stamp=path.stat()
            content=path.read_bytes();path.write_bytes(bytes([content[0]^1])+content[1:])
            os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'thay đổi'):
                delivery.preview_export(root)
            self.assertEqual(verify.call_count,3)

    def test_upload_only_uses_verified_package_and_credentials_not_persisted(self):
        root=self.job('delivery');self.execute(root)
        with patch('annoloop.cvat.upload',return_value=dict(status='verified',url='http://localhost:8080/tasks/7')) as send:
            result=delivery.upload(root,'http://localhost:8080','user','private-fixture-password','new task')
            self.assertEqual(result['status'],'verified');send.assert_called_once()
        for path in root.rglob('*.json'):
            self.assertNotIn('private-fixture-password',path.read_text())
        annotations=root/'exports/shape/annotations.json'
        annotations.write_text('{}')
        with patch('annoloop.cvat.upload') as send:
            with self.assertRaisesRegex(ValueError,'thay đổi'):
                delivery.upload(root,'http://localhost:8080','user','password','new task')
            send.assert_not_called()

    def test_snapshot_change_fails_before_gpu_dispatch(self):
        model=delivery.plan(self.store,self.pid)['config']['models'][0]
        Path(model['path']).write_bytes(b'changed checkpoint')
        with patch('annoloop.runtime.require_runtime'),patch('subprocess.Popen') as popen:
            with self.assertRaisesRegex(ValueError,'thay đổi'):
                delivery.start(self.store,self.pid,self.settings)
            popen.assert_not_called()

    def test_no_version_and_forged_track_export_remain_blocked(self):
        project=read(self.store.path(self.pid)/'project.json');project['selected_version']=None
        write(self.store.path(self.pid)/'project.json',project)
        with self.assertRaisesRegex(ValueError,'Chọn một vòng'):
            delivery.plan(self.store,self.pid)
        with self.assertRaisesRegex(ValueError,'track giả'):
            delivery.verified_export(self.root,'track')
