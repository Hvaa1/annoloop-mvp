"""A downloaded package is selectable, but never silently granted native training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from annoloop import adaptation
from annoloop.io import read,write,digest
from annoloop.model_hub import download,model_root
from annoloop.projects import ProjectStore
from annoloop.settings import Settings
from test_model_hub import model,Response,CONFIG


class HubAdaptationTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        target=self.root/'target';target.mkdir();np.save(target/'frame.npy',np.ones((2,3)))
        write(target/'class_schema.json',dict(data_type='3d',annotation_type='cuboid',classes=['car','pedestrian']))
        self.data=self.root/'data';self.store=ProjectStore(self.data);self.pid=self.store.create('test',target)
        self.settings=Settings(self.data,Path('/python'),Path('/launcher'))
        self.m=dict(model(['Car','Pedestrian']),framework='mmdetection3d',architecture='CenterPoint',modality='point_cloud')
        def opener(request,timeout):return Response(CONFIG,url=request.full_url) if 'raw.githubusercontent' in request.full_url else Response()
        download(self.data,self.m,opener=opener)
        p=patch('annoloop.hub_selection.load_catalog',return_value={'models':[self.m]});p.start();self.addCleanup(p.stop)
        p=patch('annoloop.adaptation.checkpoint_choices',return_value=[]);p.start();self.addCleanup(p.stop)
        self.weight=str(model_root(self.data,self.m['id'])/'weights.pth')
        evidence=self.root/'infos.pkl';evidence.write_bytes(b'source fixture')
        self.evidence={k:dict(path=str(evidence),sha256=digest(evidence)) for k in ['train','validation']}
        write(self.store.path(self.pid)/'setup.json',dict(source_evidence=self.evidence))
    def configure(self):
        return adaptation.save_setup(self.store,self.pid,self.settings,dict(A=self.weight,B=self.weight),True)
    def test_downloaded_python_config_visible_in_both_model_slots(self):
        choices=adaptation.model_choices(self.store,self.pid,self.settings)
        for key in ['A','B']:
            self.assertEqual(choices[key][0]['path'],self.weight)
            self.assertEqual(choices[key][0]['architecture'],'CenterPoint')
            self.assertEqual(choices[key][0]['missing_classes'],[])
            self.assertEqual(choices[key][0]['adapter_status'],'not_prepared')
    def test_can_save_registry_selection_without_running_anything(self):
        with patch('subprocess.Popen',side_effect=AssertionError('No worker on save')):
            saved=self.configure()
        self.assertEqual(saved['models']['A']['backend'],'registry_framework')
        self.assertEqual(saved['models']['A']['path'],self.weight)
        ready=adaptation.readiness(self.store,self.pid)
        self.assertFalse(ready['ready']);self.assertFalse(ready['can_check'])
        self.assertTrue(any('chưa nối train target' in b for b in ready['blockers']))
    def test_forged_native_receipt_cannot_unlock_framework_training(self):
        config=self.configure()
        write(self.store.path(self.pid)/'native_verified.json',dict(
            models={k:{f:m[f] for f in ['checkpoint_sha256','config_sha256','runtime']} for k,m in config['models'].items()},
            source_evidence=self.evidence,optimizer_steps=0,
            point_contract=dict(dataset_sha256=self.store.get(self.pid)['dataset_sha256'],bin_columns=4,coordinates_confirmed=True)))
        ready=adaptation.readiness(self.store,self.pid)
        self.assertFalse(ready['gpu_verified']);self.assertFalse(ready['ready']);self.assertFalse(ready['can_check'])
        with patch('annoloop.adaptation.require_runtime') as runtime,self.assertRaises(ValueError):
            adaptation.start_job(self.store,self.pid,self.settings,mode='verify')
        runtime.assert_not_called()
    def test_corrupted_cache_rejected_on_save(self):
        Path(self.weight).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.configure()
        self.assertFalse((self.store.path(self.pid)/'adaptation.json').exists())
    def test_unknown_selection_not_silently_dropped(self):
        with self.assertRaises(ValueError):adaptation.save_setup(self.store,self.pid,self.settings,dict(A='/unknown',B=None),True)
    def test_existing_round_identity_preserved(self):
        self.configure();p=self.store.path(self.pid)/'project.json';old=read(p);old['completed_rounds']=['M1'];write(p,old)
        saved=read(self.store.path(self.pid)/'adaptation.json')
        with self.assertRaises(ValueError):adaptation.save_setup(self.store,self.pid,self.settings,dict(A=None,B=None),True)
        self.assertEqual(saved,read(self.store.path(self.pid)/'adaptation.json'))
    def test_single_model_plan_persists_through_round_parameter_edit(self):
        selection={k:dict(kind='hub',id=self.m['id']) for k in ['A','B']}
        with patch('annoloop.model_hub.load_catalog',return_value={'models':[self.m]}):
            adaptation.save_model_selection(self.store,self.pid,self.settings,selection)
        saved=self.configure()
        self.assertEqual(saved['model_selection'],selection)
    def test_pending_download_does_not_leave_old_models_executable(self):
        self.configure()
        (model_root(self.data,self.m['id'])/'installed.json').unlink()
        selection={k:dict(kind='hub',id=self.m['id']) for k in ['A','B']}
        with patch('annoloop.model_hub.load_catalog',return_value={'models':[self.m]}),patch('subprocess.Popen',side_effect=AssertionError('No download on select')):
            saved=adaptation.save_model_selection(self.store,self.pid,self.settings,selection)
        self.assertEqual(saved['models'],{});self.assertEqual(saved['model_selection'],selection)
        self.assertFalse(adaptation.readiness(self.store,self.pid)['ready'])


if __name__=='__main__':unittest.main()
