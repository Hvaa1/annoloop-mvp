"""Folder workflow, pretrain gating and generated artifacts without GPU work."""
from pathlib import Path
import json
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from annoloop.io import digest,read,write
from annoloop.projects import ProjectStore
from annoloop.selection import SelectionPolicy
from annoloop.settings import Settings
from annoloop.setup import inspect_source,inspect_target,make_plan,prepare_project,launch_required,source_infos,refresh_training_result
import yaml


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'source'/'v1.0-mini';self.source.mkdir(parents=True)
        for split in ('train','val'):
            info=[dict(token=split,gt_names=np.array(['car','pedestrian','ignore']))]
            (self.source/f'nuscenes_infos_1sweeps_{split}.pkl').write_bytes(pickle.dumps(info))
        write(self.source/'v1.0-mini/sample.json',[dict(token='train',scene_token='one'),dict(token='val',scene_token='two')])
        self.target=self.root/'target';self.target.mkdir()
        (self.target/'one.bin').write_bytes(b'fixture only')
        self.settings=Settings(self.root/'state',Path('/python'),Path('/launcher'))
        self.store=ProjectStore(self.settings.data)
        preflight=patch('annoloop.runtime.require_runtime',return_value=dict(ready=True))
        preflight.start();self.addCleanup(preflight.stop)

    def models(self,classes=('car','pedestrian')):
        references={}
        for key,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]:
            weight=self.root/(key+'.pth');weight.write_bytes(b'test only / no real model')
            path=self.root/(key+'.metadata.json')
            write(path,dict(checkpoint=str(weight),checkpoint_sha256=digest(weight),classes=list(classes),classes_verified=True,architecture=arch))
            references[key]=str(path)
        return references

    def project(self,models=None):
        return prepare_project(self.store,self.settings,'project',inspect_source(self.source.parent),inspect_target(self.target),['car','pedestrian'],models or self.models())

    def test_auto_artifacts_without_editing_input_folders_and_reuse_never_trains(self):
        models=self.models();pid=self.project(models)
        root=self.store.path(pid)
        setup=read(root/'setup.json')
        self.assertEqual(setup['specs'],{})
        self.assertTrue(all(not m['pretrain_required'] for m in setup['plan']['models'].values()))
        self.assertTrue((root/'generated/source/source_train.json').exists())
        self.assertTrue((root/'generated/source/source_validation.json').exists())
        self.assertEqual(read(root/'generated/target/class_schema.json')['classes'],['car','pedestrian'])
        self.assertFalse((self.target/'class_schema.json').exists())
        self.assertEqual(self.store.verify_dataset(pid)['schema']['classes'],['car','pedestrian'])
        with patch('annoloop.jobs.start_pretrain') as one,patch('annoloop.jobs.start_source_pair') as pair:
            self.assertEqual(launch_required(self.store,pid,self.settings)['action'],'reuse')
            one.assert_not_called();pair.assert_not_called()

    def test_only_missing_model_is_trained_and_double_submit_is_blocked(self):
        models=self.models()
        b=read(models['B']);b['classes']=['Vehicle'];write(models['B'],b)
        pid=self.project(models);setup=read(self.store.path(pid)/'setup.json')
        self.assertEqual(set(setup['specs']),{'B'})
        with patch('annoloop.jobs.start_pretrain',return_value='mock / no process') as one,patch('annoloop.jobs.start_source_pair') as pair:
            self.assertEqual(launch_required(self.store,pid,self.settings)['jobs_started'],1)
            one.assert_called_once();pair.assert_not_called()
            with self.assertRaises(ValueError):launch_required(self.store,pid,self.settings)

    def test_manual_b_only_is_recorded_as_b_even_when_a_is_also_missing(self):
        models=self.models(['Vehicle']);pid=self.project(models)
        directory=self.root/'run_B';directory.mkdir()
        setup=read(self.store.path(pid)/'setup.json')
        write(directory/'job.json',dict(spec=setup['specs']['B']))
        write(directory/'checkpoint.metadata.json',dict(read(models['B']),classes=['car','pedestrian'],
                                                        target_gold_read=False,optimizer_steps=10))
        with patch('annoloop.jobs.start_pretrain',return_value=str(directory)) as one,patch('annoloop.jobs.start_source_pair') as pair:
            result=launch_required(self.store,pid,self.settings,detectors=['B'])
            self.assertEqual(result['detectors'],['B']);self.assertEqual(one.call_args.args[0],setup['specs']['B'])
            pair.assert_not_called()
        with patch('annoloop.jobs.status',return_value=dict(process_alive=False,training=dict(status='completed'))):
            promoted=refresh_training_result(self.store,pid)
        self.assertTrue(promoted['plan']['models']['A']['pretrain_required'])
        self.assertFalse(promoted['plan']['models']['B']['pretrain_required'])
        with patch('annoloop.jobs.start_pretrain',return_value='fixture next A') as one:
            self.assertEqual(launch_required(self.store,pid,self.settings,detectors=['A'])['detectors'],['A'])

    def test_both_missing_models_launch_pair_once(self):
        pid=self.project(self.models(['Vehicle']))
        with patch('annoloop.jobs.start_source_pair',return_value='mock / no process') as pair:
            self.assertEqual(launch_required(self.store,pid,self.settings)['jobs_started'],2)
            pair.assert_called_once()

    def test_unknown_class_is_visible_and_blocks_training(self):
        src=inspect_source(self.source)
        models=self.models()
        pid=prepare_project(self.store,self.settings,'new',src,inspect_target(self.target),['forklift'],models)
        setup=read(self.store.path(pid)/'setup.json')
        self.assertTrue(setup['plan']['source_blockers'])
        with patch('annoloop.jobs.start_source_pair') as pair:
            with self.assertRaises(ValueError):launch_required(self.store,pid,self.settings)
            pair.assert_not_called()

    def test_source_same_scene_across_splits_is_rejected(self):
        write(self.source/'v1.0-mini/sample.json',[dict(token='train',scene_token='same'),dict(token='val',scene_token='same')])
        with self.assertRaisesRegex(ValueError,'scene'):inspect_source(self.source)

    def test_target_gold_never_read_even_without_target_schema(self):
        (self.target/'annotations.json').write_text('do not parse target gold')
        self.assertEqual(inspect_target(self.target)['classes'],[])
        pid=self.project()
        self.assertFalse(self.store.get(pid)['target_gold_read'])

    def test_source_info_rejects_executable_pickle(self):
        destination=self.root/'bad'
        class Malicious:
            def __reduce__(self):return os.system,('touch '+str(destination),)
        path=self.root/'bad.pkl';path.write_bytes(pickle.dumps(Malicious()))
        with self.assertRaises(ValueError):source_infos(path)
        self.assertFalse((self.root/'bad').exists())

    def test_ready_model_subset_does_not_reinitialize_or_retrain(self):
        src=inspect_source(self.source)
        models={key:dict(architecture=arch,classes=['car','pedestrian','truck'],classes_verified=True) for key,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]}
        self.assertTrue(all(item['action']=='reuse' for item in make_plan(['car'],src,models)['models'].values()))

    def test_raw_weight_class_names_are_not_guessed(self):
        src=inspect_source(self.source)
        unknown=dict(architecture='SECONDNetIoU',classes=['car'],classes_verified=False)
        with self.assertRaises(ValueError):make_plan(['car'],src,{'A':unknown,'B':None})

    def test_new_source_class_anchor_generated_from_source_train_geometry(self):
        for suffix in ('train','val'):
            infos=[dict(token=suffix,gt_names=np.array(['forklift']),gt_boxes=np.array([[0.,0.,1.5,3.,1.2,2.,0.]]))]
            (self.source/f'nuscenes_infos_1sweeps_{suffix}.pkl').write_bytes(pickle.dumps(infos))
        pid=prepare_project(self.store,self.settings,'new class',inspect_source(self.source),inspect_target(self.target),['forklift'],self.models())
        setup=read(self.store.path(pid)/'setup.json')
        self.assertFalse(setup['plan']['config_blockers'])
        for spec in setup['specs'].values():
            cfg=yaml.safe_load(Path(spec['config']).read_text())
            anchors=cfg['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG']
            self.assertEqual(len(anchors),1)
            self.assertEqual(anchors[0]['class_name'],'forklift')
            self.assertEqual(anchors[0]['anchor_sizes'],[[3.,1.2,2.]])

    def test_completed_checkpoint_becomes_reusable_without_new_launch(self):
        models=self.models();b=read(models['B']);b['classes']=['Vehicle'];write(models['B'],b)
        pid=self.project(models);directory=self.root/'run';directory.mkdir()
        write(directory/'job.json',{})
        learned=dict(read(models['B']),classes=['car','pedestrian'],target_gold_read=False,optimizer_steps=10)
        write(directory/'checkpoint.metadata.json',learned)
        with patch('annoloop.jobs.start_pretrain',return_value=str(directory)):
            launch_required(self.store,pid,self.settings)
        with patch('annoloop.jobs.status',return_value=dict(process_alive=True,training=dict(status='completed'))):
            self.assertTrue(refresh_training_result(self.store,pid)['plan']['models']['B']['pretrain_required'])
        with patch('annoloop.jobs.status',return_value=dict(process_alive=False,training=dict(status='completed'))):
            setup=refresh_training_result(self.store,pid)
        self.assertEqual(setup['launches'][-1]['status'],'completed')
        self.assertFalse(setup['plan']['models']['B']['pretrain_required'])
        with patch('annoloop.jobs.start_pretrain') as start:
            self.assertEqual(launch_required(self.store,pid,self.settings)['action'],'reuse')
            start.assert_not_called()

    def test_stopped_a_does_not_prevent_completed_b_metadata_promotion(self):
        models=self.models(['Vehicle']);pid=self.project(models)
        pair=self.root/'partial_pair';pair.mkdir()
        with patch('annoloop.jobs.start_source_pair',return_value=str(pair)):
            launch_required(self.store,pid,self.settings)
        a=self.root/'stopped_A';b=self.root/'completed_B'
        write(a/'job.json',{});write(b/'job.json',{})
        write(b/'checkpoint.metadata.json',dict(read(models['B']),classes=['car','pedestrian'],
                                                target_gold_read=False,optimizer_steps=10))
        write(pair/'pair.json',dict(status='completed_with_stops',jobs={'A':str(a),'B':str(b)}))
        def status(path):
            return dict(process_alive=False,training=dict(status='stopped' if Path(path)==a else 'completed'))
        with patch('annoloop.jobs.status',side_effect=status):
            setup=refresh_training_result(self.store,pid)
        self.assertEqual(setup['launches'][-1]['status'],'completed_with_stops')
        self.assertEqual(setup['launches'][-1]['stopped_detectors'],['A'])
        self.assertTrue(setup['plan']['models']['A']['pretrain_required'])
        self.assertFalse(setup['plan']['models']['B']['pretrain_required'])
        self.assertNotEqual(setup['model_metadata']['A'],str(a/'checkpoint.metadata.json'))
        self.assertEqual(setup['model_metadata']['B'],str(b/'checkpoint.metadata.json'))

    def test_failed_pair_can_retry_only_after_controller_and_children_stop(self):
        pid=self.project(self.models(['Vehicle']))
        pair=self.root/'pair';pair.mkdir()
        with patch('annoloop.jobs.start_source_pair',return_value=str(pair)):
            launch_required(self.store,pid,self.settings)
        write(pair/'pair.json',dict(status='failed',jobs={'A':str(self.root/'oldrun')},error='import failed'))
        write(pair/'controller.json',dict(pid=123,process_start='identity'))
        with patch('annoloop.reused.training_monitor.process_identity',return_value='identity'),patch('annoloop.jobs.status',return_value=dict(process_alive=False)):
            self.assertEqual(refresh_training_result(self.store,pid)['launches'][-1]['status'],'submitted')
        with patch('annoloop.reused.training_monitor.process_identity',return_value=None),patch('annoloop.jobs.status',return_value=dict(process_alive=False)):
            self.assertEqual(refresh_training_result(self.store,pid)['launches'][-1]['status'],'failed')
        with patch('annoloop.jobs.start_source_pair',return_value=str(self.root/'newpair')) as start:
            launch_required(self.store,pid,self.settings)
            start.assert_called_once()
        self.assertEqual(len(read(self.store.path(pid)/'setup.json')['launches']),2)
        self.assertTrue((pair/'pair.json').exists())

    def test_preflight_failure_keeps_project_retryable_without_dispatch(self):
        pid=self.project(self.models(['Vehicle']))
        with patch('annoloop.runtime.require_runtime',side_effect=ValueError('runtime not ready')),patch('annoloop.jobs.start_source_pair') as start:
            with self.assertRaisesRegex(ValueError,'runtime not ready'):launch_required(self.store,pid,self.settings)
            start.assert_not_called()
        self.assertEqual(read(self.store.path(pid)/'setup.json')['launches'],[])

    def test_stale_running_pair_after_controller_exit_is_retryable_only_when_children_stop(self):
        pid=self.project(self.models(['Vehicle']))
        pair=self.root/'pair';pair.mkdir()
        with patch('annoloop.jobs.start_source_pair',return_value=str(pair)):
            launch_required(self.store,pid,self.settings)
        write(pair/'pair.json',dict(status='running',jobs={'A':str(self.root/'oldrun')}))
        write(pair/'controller.json',dict(pid=123,process_start='old'))
        with patch('annoloop.reused.training_monitor.process_identity',return_value=None),patch('annoloop.jobs.status',return_value=dict(process_alive=True)):
            self.assertEqual(refresh_training_result(self.store,pid)['launches'][-1]['status'],'submitted')
        with patch('annoloop.reused.training_monitor.process_identity',return_value=None),patch('annoloop.jobs.status',return_value=dict(process_alive=False)):
            result=refresh_training_result(self.store,pid)
        self.assertEqual(result['launches'][-1]['status'],'failed')
        self.assertEqual(read(pair/'pair.json')['status'],'running')  # Preserve old receipt.


if __name__=='__main__':unittest.main()
