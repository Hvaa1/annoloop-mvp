"""Post-download gates consume actual receipts; fixture workers never use GPU."""
from pathlib import Path
import copy
import unittest
from unittest.mock import patch
import torch
from annoloop import adaptation,preparation
from annoloop.io import read,write,digest
from annoloop.native_training import checked_gradients
import test_adaptation as fixtures


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.NativeAdaptationTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.store=self.fixture.store;self.pid=self.fixture.pid;self.settings=self.fixture.settings
        self.root=self.store.path(self.pid)
        setup=read(self.root/'setup.json');setup['source_counts']={s:{'car':20,'pedestrian':20} for s in ('train','validation')}
        write(self.root/'setup.json',setup)
        for role,rows in self.fixture.candidates.items():
            Path(rows[0]['config']).write_text('DATA_CONFIG:\n  POINT_FEATURE_ENCODING:\n    used_feature_list: [x, y, z]\n')
        self.config=self.fixture.configure()

    def certificate(self):
        self.fixture.mark_verified(self.config)
        cert=read(self.root/'native_verified.json')
        cert.update(preparation_protocol=1,adaptation_classes=self.config['adaptation_classes'],preparation_checks={})
        for role,model in self.config['models'].items():
            directory=self.root/'adaptation_jobs/verified'/role
            verification=dict(checkpoint_sha256=model['checkpoint_sha256'],config_sha256=model['config_sha256'],
                optimizer_steps=0,target_gold_read=False,separate_scores_checked=True,forward_backward_checked=True,
                target_adapter_backward_checked=True,signed_ignore_effect_checked=True,
                source_gradient_tensors=2,target_gradient_tensors=2,source_loss=1.,target_loss=1.)
            calibration=dict(domain='source_validation',checkpoint_sha256=model['checkpoint_sha256'],
                curves={name:dict(confidence=[[0,0],[1,1]],quality=[[0,0],[1,1]]) for name in self.config['adaptation_classes']})
            receipt={}
            for key,doc in [('verification',verification),('calibration',calibration)]:
                path=directory/(key+'.json');write(path,doc);receipt[key]=str(path);receipt[key+'_sha256']=digest(path)
            cert['preparation_checks'][role]=receipt
        write(self.root/'native_verified.json',cert);return cert

    def test_old_forward_backward_certificate_cannot_unlock_new_workflow(self):
        config=read(self.root/'adaptation.json');config['preparation_protocol']=1;write(self.root/'adaptation.json',config)
        self.fixture.mark_verified(config)
        state=adaptation.readiness(self.store,self.pid)
        self.assertTrue(state['gpu_verified']);self.assertFalse(state['ready']);self.assertTrue(state['can_check'])
        cert=self.certificate()
        self.assertTrue(preparation.certificate_valid(self.root,cert,config))
        self.assertTrue(adaptation.readiness(self.store,self.pid)['ready'])

    def test_tampered_wrong_class_and_wrong_checkpoint_evidence_never_unlock(self):
        cert=self.certificate()
        for field in ('classes','weights','path'):
            row=copy.deepcopy(cert)
            if field=='classes':row['adaptation_classes']=['car']
            elif field=='weights':row['preparation_checks']['A']['verification_sha256']='0'*64
            else:row['preparation_checks']['A']['verification']=str(self.fixture.root/'outside.json')
            self.assertFalse(preparation.certificate_valid(self.root,row,self.config))
        path=Path(cert['preparation_checks']['A']['calibration']);write(path,{})
        self.assertFalse(preparation.certificate_valid(self.root,cert,self.config))

    def test_overview_does_not_dispatch_and_explicit_check_dispatches_once(self):
        before=digest(self.root/'adaptation.json')
        with patch('annoloop.adaptation.start_job',return_value='fixture verification job') as start:
            report=preparation.overview(self.store,self.pid);self.assertFalse(report['ready']);start.assert_not_called()
            self.assertEqual(before,digest(self.root/'adaptation.json'))
            with patch('annoloop.adaptation.model_choices',return_value=self.fixture.candidates):
                report=preparation.check_prepared(self.store,self.pid,self.settings,True,4)
            start.assert_called_once();self.assertEqual(start.call_args.kwargs['mode'],'verify')
            self.assertEqual(report['status'],'checking')
        self.assertEqual(read(self.root/'adaptation.json')['preparation_protocol'],1)
        self.assertFalse((self.root/'native_verified.json').exists())

    def test_backward_requires_finite_nonzero_trainable_gradients(self):
        model=torch.nn.Linear(2,1)
        with self.assertRaisesRegex(ValueError,'gradient'):checked_gradients(torch,model)
        model(torch.ones(1,2)).sum().backward();self.assertGreater(checked_gradients(torch,model),0)
        model.weight.grad[:]=float('nan')
        with self.assertRaisesRegex(ValueError,'NaN'):checked_gradients(torch,model)

    def test_native_verify_fits_full_source_calibration_before_certification(self):
        from annoloop.adaptation_worker import run
        config=read(self.root/'adaptation.json');config['preparation_protocol']=1;write(self.root/'adaptation.json',config)
        class Adapter:
            def __init__(self,model,classes,frames,output,settings,columns):self.model=model;self.output=output
            def spec(self,checkpoint):return dict(self.model,path=checkpoint)
            def predict_source_validation(self,checkpoint):
                return dict(domain='source_validation',boxes=[dict(class_name=name,cls_score=.8,iou_score=.8,source_gt_iou=.9)
                    for name in config['adaptation_classes'] for _ in range(12)])
        job=self.root/'adaptation_jobs/full-check';write(job/'state.json',dict(status='running'))
        write(job/'spec.json',dict(project_id=self.pid,mode='verify',config=config,dataset_sha256=self.store.get(self.pid)['dataset_sha256'],source_evidence=self.fixture.evidence))
        # Avoid writing a project certificate in the transport fixture itself.
        template=self.certificate();verification=read(template['preparation_checks']['A']['verification']);(self.root/'native_verified.json').unlink()
        def call(adapter,command,log):
            write(Path(log).parent/'verified.json',dict(verification,checkpoint_sha256=adapter.model['checkpoint_sha256'],config_sha256=adapter.model['config_sha256']))
        Adapter.call=call
        with patch('annoloop.native_adapter.NativeDetectorAdapter',Adapter),patch('annoloop.device.gpu_idle'), \
             patch('annoloop.settings.Settings.load',return_value=self.settings):
            run(job)
        certificate=read(self.root/'native_verified.json')
        self.assertTrue(preparation.certificate_valid(self.root,certificate,config))
        self.assertEqual(certificate['preparation_protocol'],1)
        for role in ('A','B'):self.assertTrue((job/role/'calibration.json').exists())
        (self.root/'native_verified.json').unlink()
        with patch.object(Adapter,'predict_source_validation',return_value=dict(domain='source_validation',boxes=[])), \
             patch('annoloop.native_adapter.NativeDetectorAdapter',Adapter),patch('annoloop.device.gpu_idle'), \
             patch('annoloop.settings.Settings.load',return_value=self.settings):
            with self.assertRaisesRegex(ValueError,'Chưa đủ dự đoán'):run(job)
        self.assertFalse((self.root/'native_verified.json').exists())

    def test_background_flow_downloads_selected_ids_once_and_preserves_context(self):
        from annoloop import preparation_jobs
        config=read(self.root/'adaptation.json');config['model_selection']={r:dict(kind='hub',id='one') for r in ('A','B')};write(self.root/'adaptation.json',config)
        job=self.root/'preparation_jobs/one';write(job/'state.json',dict(status='starting'))
        write(job/'spec.json',dict(project_id=self.pid,context=preparation_jobs.context(self.store,self.pid),axes=True,columns=4))
        with patch('annoloop.model_hub.load_catalog',return_value=dict(models=[dict(id='one')])), \
             patch('annoloop.adapter_factory.select_and_prepare') as download, \
             patch('annoloop.preparation.check_prepared',return_value=dict(status='blocked',ready=False)) as check, \
             patch('subprocess.Popen',side_effect=AssertionError('No GPU/process fixture')):
            preparation_jobs.run(job,self.settings)
            download.assert_called_once();check.assert_called_once()
        self.assertEqual(read(job/'state.json')['status'],'blocked')
        before=read(job/'spec.json');before['context']='stale';write(job/'spec.json',before)
        with patch('annoloop.adapter_factory.select_and_prepare') as download:
            with self.assertRaisesRegex(ValueError,'đã đổi'):preparation_jobs.run(job,self.settings)
            download.assert_not_called()

    def test_save_input_without_dispatch_and_background_preserves_saved_mode(self):
        from annoloop import preparation_jobs
        from types import SimpleNamespace
        with patch('annoloop.adaptation.model_choices',return_value=self.fixture.candidates), \
             patch('subprocess.Popen',side_effect=AssertionError('Save must not launch')):
            saved=preparation.save_input_setup(self.store,self.pid,self.settings,True,4,'current_scan_only')
        self.assertEqual(saved['mmdet_input_mode'],'current_scan_only')
        with patch('subprocess.Popen',return_value=SimpleNamespace(pid=987654)), \
             patch('annoloop.preparation_jobs.process_identity',return_value='fixture-process'):
            job=preparation_jobs.start(self.store,self.pid,self.settings,True,4)
        self.assertEqual(read(Path(job)/'spec.json')['mmdet_input_mode'],'current_scan_only')

    def test_background_dispatch_checks_confirmation_and_blocks_duplicate_jobs(self):
        from annoloop import preparation_jobs
        from types import SimpleNamespace
        with patch('subprocess.Popen',return_value=SimpleNamespace(pid=987654)) as launch, \
             patch('annoloop.preparation_jobs.process_identity',return_value='fixture-process'):
            with self.assertRaisesRegex(ValueError,'Xác nhận'):preparation_jobs.start(self.store,self.pid,self.settings,False,4)
            launch.assert_not_called()
            job=preparation_jobs.start(self.store,self.pid,self.settings,True,4)
            launch.assert_called_once()
            self.assertEqual(preparation_jobs.active(self.store,self.pid),job)
            with self.assertRaisesRegex(ValueError,'đang chạy'):preparation_jobs.start(self.store,self.pid,self.settings,True,4)
            launch.assert_called_once()
            with patch('annoloop.preparation_jobs.os.kill') as kill:
                self.assertTrue(preparation_jobs.stop(job));kill.assert_called_once()
        # A recycled/missing process identity never authorizes a signal.
        with patch('annoloop.preparation_jobs.process_identity',return_value='different'),patch('annoloop.preparation_jobs.os.kill') as kill:
            self.assertEqual(preparation_jobs.status(job)['status'],'interrupted')
            self.assertFalse(preparation_jobs.stop(job));kill.assert_not_called()


if __name__=='__main__':unittest.main()
