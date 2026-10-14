"""Native dispatch/ignore/label safeguards; these tests never run a trainer."""
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from annoloop import adaptation
from annoloop.io import read, write, digest
from annoloop.projects import ProjectStore
from annoloop.native_data import signed_labels
from annoloop.native_ignore import install
from annoloop.native_adapter import NativeDetectorAdapter
from annoloop.settings import Settings


class NativeAdaptationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset = self.root / 'target'
        dataset.mkdir()
        write(dataset / 'class_schema.json', dict(data_type='3d', annotation_type='cuboid', classes=['car', 'pedestrian']))
        np.save(dataset / 'frame.npy', np.asarray([[1., 2., 3.]], dtype=np.float32))
        self.store = ProjectStore(self.root / 'data')
        self.pid = self.store.create('fixture', dataset)
        self.settings = Settings(self.root / 'data', Path(sys.executable), self.root / 'launcher.py')
        self.candidates = {}
        for key, architecture in adaptation.ARCHITECTURES.items():
            weight = self.root / (key + '.pth')
            weight.write_bytes(b'fixture only')
            config = self.root / (key + '.yaml')
            config.write_text('fixture')
            self.candidates[key] = [dict(path=str(weight), config=str(config), runtime='fixture-runtime',
                 classes=['car', 'pedestrian'], architecture=architecture, name=key, intermediate=True)]
        evidence = self.root / 'infos.pkl'
        evidence.write_bytes(b'fixture')
        self.evidence = dict(train=dict(path=str(evidence), sha256=digest(evidence)),
                             validation=dict(path=str(evidence), sha256=digest(evidence)))
        write(self.store.path(self.pid) / 'setup.json', dict(source_evidence=self.evidence))

    def configure(self, confirmed=True):
        with patch('annoloop.adaptation.model_choices', return_value=self.candidates):
            return adaptation.save_setup(self.store, self.pid, self.settings,
                        {key: values[0]['path'] if values else None for key, values in self.candidates.items()}, confirmed)

    def mark_verified(self, config):
        write(self.store.path(self.pid) / 'native_verified.json', dict(
            models={k: {f: m[f] for f in ('checkpoint_sha256', 'config_sha256', 'runtime')}
                    for k, m in config['models'].items()}, optimizer_steps=0, source_evidence=self.evidence,
                    point_contract=dict(dataset_sha256=self.store.get(self.pid)['dataset_sha256'],
                                        bin_columns=config['bin_columns'], coordinates_confirmed=config['coordinates_confirmed'])))

    def test_coordinate_confirmation_and_real_verification_gate_dispatch(self):
        config = self.configure(False)
        self.assertFalse(adaptation.readiness(self.store, self.pid)['can_check'])
        with patch('subprocess.Popen') as process:
            with self.assertRaisesRegex(ValueError, 'Xác nhận'):
                adaptation.start_job(self.store, self.pid, self.settings)
            process.assert_not_called()
        config = self.configure(True)
        self.assertTrue(adaptation.readiness(self.store, self.pid)['can_check'])
        self.assertFalse(adaptation.readiness(self.store, self.pid)['ready'])
        self.mark_verified(config)
        self.assertTrue(adaptation.readiness(self.store, self.pid)['ready'])
        Path(config['models']['A']['path']).write_bytes(b'changed')
        self.assertFalse(adaptation.readiness(self.store, self.pid)['ready'])

    def test_missing_second_detector_never_dispatches(self):
        self.candidates['B'] = []
        self.configure()
        with patch('subprocess.Popen') as process:
            with self.assertRaisesRegex(ValueError, 'Detector B'):
                adaptation.start_job(self.store, self.pid, self.settings)
            process.assert_not_called()

    def test_source_change_invalidates_gpu_check(self):
        config = self.configure()
        self.mark_verified(config)
        Path(self.evidence['train']['path']).write_bytes(b'changed')
        self.assertFalse(adaptation.readiness(self.store, self.pid)['ready'])

    def test_binary_layout_change_requires_another_gpu_verification(self):
        config = self.configure()
        self.mark_verified(config)
        with patch('annoloop.adaptation.model_choices', return_value=self.candidates):
            adaptation.save_setup(self.store, self.pid, self.settings,
                {key: values[0]['path'] for key, values in self.candidates.items()}, True, bin_columns=3)
        self.assertFalse(adaptation.readiness(self.store, self.pid)['gpu_verified'])

    def calibration_result(self, config):
        job=self.store.path(self.pid)/'adaptation_jobs/report_fixture'
        write(job/'spec.json',dict(source_evidence=self.evidence))
        root=job/'A/source/full_validation'
        write(root/'spec.json',dict(config['models']['A'],mode='validation',score_threshold=.05))
        records=dict(domain='source_validation',boxes=[dict(class_name=name,cls_score=.2,
            iou_score=.3,source_gt_iou=0.) for name,n in [('car',20),('pedestrian',5)] for _ in range(n)])
        write(root/'result.json',dict(domain='source_validation',checkpoint_sha256=config['models']['A']['checkpoint_sha256'],
              calibration_records=records,frames=[{}]*81))
        return root

    def test_known_insufficient_calibration_blocks_repeat_run_but_keeps_gpu_verified(self):
        config=self.configure();self.mark_verified(config)
        self.calibration_result(config)
        check=adaptation.readiness(self.store,self.pid)
        self.assertTrue(check['gpu_verified']);self.assertTrue(check['can_check'])
        self.assertFalse(check['ready'])
        self.assertIn('5/10','; '.join(check['blockers']))
        self.assertEqual(check['calibration_reports']['A']['source_frames'],81)
        with patch('annoloop.adaptation.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError,'5/10'):
                adaptation.start_job(self.store,self.pid,self.settings)
            launch.assert_not_called()

    def test_calibration_diagnostics_ignore_other_weights_configs_and_partial_scans(self):
        config=self.configure();self.mark_verified(config)
        root=self.calibration_result(config)
        original=read(root/'spec.json')
        for field,value in [('checkpoint_sha256','other'),('config_sha256','other'),
                            ('runtime','other'),('limit',1),('mode','target_sample'),('score_threshold',.1)]:
            with self.subTest(field=field):
                write(root/'spec.json',dict(original,**{field:value}))
                self.assertEqual(adaptation.readiness(self.store,self.pid)['calibration_reports'],{})
        write(root/'spec.json',original)
        write(root.parents[2]/'spec.json',dict(source_evidence={'validation':'different'}))
        self.assertEqual(adaptation.readiness(self.store,self.pid)['calibration_reports'],{})

    def test_explicit_class_subset_reuses_verified_weights_and_excludes_scarce_class_from_calibration(self):
        config=self.configure();self.mark_verified(config);self.calibration_result(config)
        certificate=self.store.path(self.pid)/'native_verified.json'
        identity=digest(certificate)
        self.assertFalse(adaptation.readiness(self.store,self.pid)['ready'])
        with patch('annoloop.adaptation.model_choices',return_value=self.candidates):
            saved=adaptation.save_setup(self.store,self.pid,self.settings,
                {k:v[0]['path'] for k,v in self.candidates.items()},True,adaptation_classes=['car'])
        check=adaptation.readiness(self.store,self.pid)
        self.assertTrue(check['ready']);self.assertTrue(check['gpu_verified'])
        self.assertEqual(saved['adaptation_classes'],['car'])
        self.assertEqual(saved['models']['A']['classes'],['car','pedestrian'])
        self.assertEqual([r['class_name'] for r in check['calibration_reports']['A']['classes']],['car'])
        self.assertEqual(digest(certificate),identity)
        for invalid in ([],['unknown'],['car','car']):
            with self.subTest(classes=invalid),self.assertRaisesRegex(ValueError,'Class adaptation'):
                adaptation.save_setup(self.store,self.pid,self.settings,{},True,adaptation_classes=invalid)
        p=self.store.get(self.pid);p['completed_rounds']=['M1'];write(self.store.path(self.pid)/'project.json',p)
        with patch('annoloop.adaptation.model_choices',return_value=self.candidates), \
             self.assertRaisesRegex(ValueError,'giữ class adaptation'):
            adaptation.save_setup(self.store,self.pid,self.settings,
                {k:v[0]['path'] for k,v in self.candidates.items()},True,adaptation_classes=['car','pedestrian'])

    def test_project_settings_preserve_dataset_and_snapshots(self):
        identity = self.store.get(self.pid)['dataset_sha256']
        updated = self.store.configure(self.pid, 'Renamed', 5, 'automatic')
        self.assertEqual(updated['name'], 'Renamed')
        self.assertEqual(updated['max_rounds'], 5)
        self.assertEqual(updated['dataset_sha256'], identity)
        self.assertEqual(updated['completed_rounds'], [])

    def test_orphaned_inference_worker_blocks_reconfiguration(self):
        job = self.store.path(self.pid) / 'adaptation_jobs' / 'old'
        write(job / 'state.json', dict(status='running', pid=100, process_start='old-controller'))
        write(job / 'A' / 'worker.process.json', dict(pid=101, process_start='live-child'))
        with patch('annoloop.adaptation.process_identity', side_effect=lambda pid: 'live-child' if pid == 101 else None):
            state = adaptation.job_status(job)
            self.assertTrue(state['process_alive']);self.assertFalse(state['controller_alive'])
            self.assertEqual(state['status'], 'orphaned_worker')
            with self.assertRaisesRegex(ValueError, 'Chờ job'):
                self.store.configure(self.pid, 'rename', 5, 'manual')

    def test_retry_requires_explicit_action_and_preserves_legacy_work(self):
        config = self.configure();self.mark_verified(config)
        root = self.store.path(self.pid)
        write(root / 'work/M1/state.json', dict(status='running', old_receipt=True))
        old = (root / 'work/M1/state.json').read_bytes()
        with patch('annoloop.adaptation.require_runtime'), patch('annoloop.device.gpu_idle'), \
             patch('annoloop.adaptation.subprocess.Popen', return_value=SimpleNamespace(pid=123)) as launch, \
             patch('annoloop.adaptation.process_identity', return_value=None):
            with self.assertRaisesRegex(ValueError, 'Vòng M1 bị ngắt'):
                adaptation.start_job(self.store, self.pid, self.settings)
            launch.assert_not_called()
            job = Path(adaptation.start_job(self.store, self.pid, self.settings, retry=True))
            self.assertTrue(read(job / 'spec.json')['retry_interrupted_round'])
            self.assertEqual(read(job / 'spec.json')['attempt_id'], job.name)
            self.assertEqual(launch.call_count, 1)
        self.assertEqual((root / 'work/M1/state.json').read_bytes(), old)
        self.assertEqual(self.store.get(self.pid)['completed_rounds'], [])

    def test_retry_never_dispatches_when_controller_is_still_alive(self):
        config = self.configure();self.mark_verified(config)
        root = self.store.path(self.pid)
        write(root / 'work/M1/state.json', dict(status='running'))
        with patch('annoloop.adaptation.active_job', return_value='live-job'), \
             patch('annoloop.adaptation.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'còn sống'):
                adaptation.start_job(self.store, self.pid, self.settings, retry=True)
            launch.assert_not_called()

    def test_resume_dispatch_is_explicit_and_records_validated_plan(self):
        config=self.configure();self.mark_verified(config)
        work=self.store.path(self.pid)/'work/M1/attempts/old'
        write(work/'state.json',dict(status='interrupted'))
        recovery=dict(work=str(work),index=1,completed={'A':{'checkpoint':'fixture'}},checkpoints={})
        with patch('annoloop.resume.plan',return_value=recovery) as planner, \
             patch('annoloop.adaptation.require_runtime'),patch('annoloop.device.gpu_idle'), \
             patch('annoloop.adaptation.subprocess.Popen',return_value=SimpleNamespace(pid=123)) as launch, \
             patch('annoloop.adaptation.process_identity',return_value=None):
            job=Path(adaptation.start_job(self.store,self.pid,self.settings,resume_from=str(work)))
            self.assertEqual(read(job/'spec.json')['resume_plan'],recovery)
            planner.assert_called_once_with(self.store,self.pid,str(work))
            self.assertEqual(launch.call_count,1)
        with self.assertRaisesRegex(ValueError,'hai thao tác'):
            adaptation.start_job(self.store,self.pid,self.settings,retry=True,resume_from=str(work))

    def test_nested_attempt_trainer_is_detected_as_live_orphan(self):
        job = self.store.path(self.pid) / 'adaptation_jobs/old'
        write(job / 'state.json', dict(status='running', pid=100, process_start='old-controller'))
        run = self.store.path(self.pid) / 'work/M1/attempts/try1/A'
        write(run / 'training.json', dict(status='running'))
        with patch('annoloop.adaptation.process_identity', return_value=None), \
             patch('annoloop.jobs.status', return_value=dict(process_alive=True)):
            self.assertEqual(adaptation.job_status(job)['status'], 'orphaned_worker')

    def test_multiclass_signed_ids_are_preserved_and_gold_is_rejected(self):
        boxes = [dict(id='a', class_name='pedestrian', box=[1., 0., 0., 1., 1., 2., 0.], score=.8, training_policy='positive')]
        ignored = [dict(id='b', class_name='car', box=[4., 0., 0., 4., 2., 2., 0.], score=.2, training_policy='ignore_region')]
        geometry, names, _ = signed_labels(dict(boxes=boxes, ignored=ignored), ['car', 'pedestrian'])
        self.assertEqual(geometry[:, 7].tolist(), [2., -1.])
        self.assertEqual(names.tolist(), ['pedestrian', 'car'])
        with self.assertRaisesRegex(ValueError, 'gold'):
            signed_labels(dict(boxes=boxes, ignored=ignored, target_gold_read=True), ['car', 'pedestrian'])

    def test_dense_ignore_masks_all_classes_and_preserves_positive_assignment(self):
        modules = {}
        for name in ('pcdet.utils.box_utils', 'pcdet.ops.iou3d_nms.iou3d_nms_utils',
                     'pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils'):
            modules[name] = ModuleType(name)
        modules['pcdet.utils.box_utils'].boxes3d_nearest_bev_iou = lambda a, b: torch.ones(len(a), len(b))
        modules['pcdet.ops.iou3d_nms.iou3d_nms_utils'].boxes_iou3d_gpu = lambda a, b: torch.ones(len(a), len(b))
        modules['pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils'].points_in_boxes_gpu = lambda *a: None
        seen = []
        def original(gt):
            seen.append(gt.clone())
            return dict(box_cls_labels=torch.tensor([[2, 0]]))
        head = SimpleNamespace(assign_targets=original, use_multihead=False, anchors=[torch.zeros(1, 1, 1, 1, 2, 7)])
        layer = SimpleNamespace(register_forward_hook=lambda f: None)
        model = SimpleNamespace(dense_head=head, roi_head=SimpleNamespace(proposal_target_layer=layer),
                                model_cfg=SimpleNamespace(NAME='SECONDNetIoU'))
        with patch.dict(sys.modules, modules):
            install(model)
            labels = head.assign_targets(torch.tensor([[[4., 0., 0., 4., 2., 2., 0., 2.],
                                                        [8., 0., 0., 4., 2., 2., 0., -1.]]]))
        self.assertTrue((seen[0][:, :, -1] >= 0).all())
        self.assertEqual(labels['box_cls_labels'].tolist(), [[2, -1]])

    def test_target_adapter_uses_registered_native_python_and_requires_final_receipt(self):
        config = self.configure()
        frames = read(self.store.path(self.pid) / 'dataset.json')['frames']
        adapter = NativeDetectorAdapter(config['models']['A'], ['car', 'pedestrian'], frames,
                                        self.root / 'adapter', self.settings)
        labels = self.root / 'labels.json'
        write(labels, dict(target_gold_read=False, frames=[]))
        output = self.root / 'target_run'
        with patch('annoloop.native_adapter.subprocess.run') as launch, \
             patch('annoloop.jobs.status', return_value=dict(training=dict(status='completed'), process_alive=False)):
            # A launcher completed receipt is insufficient without trained.json.
            with self.assertRaises(FileNotFoundError):
                adapter.train_target(config['models']['A']['path'], labels, output,
                                     dict(epochs=1, ckpt_interval=10, save_final=True))
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--') + 1], str(self.settings.trainer_python))
            self.assertIn('annoloop.native_training', command)


if __name__ == '__main__':
    unittest.main()
