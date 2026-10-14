"""Controller tests with explicitly synthetic adapter fixtures, never real model runs."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from annoloop.adapters import DetectorAdapter
from annoloop.controller import RoundController
from annoloop.io import digest, read, write
from annoloop.projects import ProjectStore
from annoloop.reused.checkpoint_policy import with_final_checkpoint
from annoloop.selection import SelectionPolicy
from annoloop.native_data import signed_labels


class FixtureAdapter(DetectorAdapter):
    def __init__(self, key, events):
        self.key, self.events = key, events

    def predict_target(self, checkpoint, frames):
        self.events.append((self.key, "predict"))
        return {f["frame_id"]: dict(boxes=[dict(id="same_id", class_name="car", box=[4., 0., 0., 4., 2., 2., 0.],
                     score=.8, cls_score=.8, iou_score=.8)]) for f in frames}

    def predict_source_validation(self, checkpoint):
        return dict(domain="source_validation", boxes=[dict(class_name="car", cls_score=i/20, iou_score=i/20,
                  source_gt_iou=i/20) for i in range(20)])

    def train_target(self, checkpoint, labels_path, output, policy):
        self.events.append((self.key, "train", digest(labels_path), policy))
        output = Path(output)
        output.mkdir()
        checkpoint = output / "test_fixture.pth"
        checkpoint.write_bytes(b'controller test only, not actual model weights')
        return dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), optimizer_steps=1,
                    target_gold_used=False, shared_labels_sha256=digest(labels_path))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset = self.root / "dataset"
        dataset.mkdir()
        write(dataset / "class_schema.json", dict(data_type="3d", annotation_type="cuboid", classes=["car"]))
        np.save(dataset / "f0.npy", np.array([[4., 0., 0.]] * 10, np.float32))
        self.store = ProjectStore(self.root / "data")
        self.dataset = dataset
        self.events = []
        self.adapters = {key: FixtureAdapter(key, self.events) for key in ("A", "B")}
        self.initial = {}
        for key in ("A", "B"):
            path = self.root / (key + ".pth")
            path.write_bytes(b'initial test fixture')
            self.initial[key] = str(path)

    def controller(self, project, **kwargs):
        return RoundController(self.store, project, self.adapters, self.initial,
            point_loader=lambda frame: np.load(frame["media"]), gpu_guard=lambda: None, **kwargs)

    def test_automatic_runs_budget_sequentially_with_shared_labels(self):
        project = self.store.create("fixture", self.dataset, max_rounds=2, execution_mode="automatic",
                                    policy=SelectionPolicy(mode="automatic"))
        results = self.controller(project).run()
        self.assertEqual([r["version"] for r in results], ["M1", "M2"])
        trained = [e for e in self.events if e[1] == "train"]
        self.assertEqual([e[0] for e in trained], ["A", "B", "A", "B"])
        self.assertEqual(trained[0][2], trained[1][2])
        self.assertEqual(trained[2][2], trained[3][2])
        self.assertTrue(all(e[3]["ckpt_interval"] == 10 and e[3]["save_final"] for e in trained))
        self.assertEqual(self.store.get(project)["selected_version"], "M2")

    def test_manual_execution_runs_only_one_round_even_when_selection_automatic(self):
        project = self.store.create("fixture", self.dataset, max_rounds=3, execution_mode="manual",
                                    policy=SelectionPolicy(mode="automatic"))
        self.assertEqual(len(self.controller(project).run()), 1)

    def test_both_source_calibrations_precede_target_predictions(self):
        project=self.store.create('fixture',self.dataset)
        for key,adapter in self.adapters.items():
            original=adapter.predict_source_validation
            def source(checkpoint, key=key, original=original):
                self.events.append((key,'source_validation'))
                return original(checkpoint)
            adapter.predict_source_validation=source
        self.controller(project).run_one()
        self.assertEqual(self.events[:4],[('A','source_validation'),('B','source_validation'),
                                         ('A','predict'),('B','predict')])

    def test_insufficient_source_calibration_stops_before_any_target_prediction_or_training(self):
        project=self.store.create('fixture',self.dataset)
        self.adapters['A'].predict_source_validation=lambda _: dict(domain='source_validation',
            boxes=[dict(class_name='car',cls_score=.1,iou_score=.2,source_gt_iou=0.) for _ in range(5)])
        with self.assertRaisesRegex(ValueError,'car: 5/10'):
            self.controller(project).run_one()
        self.assertEqual(self.events,[])
        report=read(self.store.path(project)/'work/M1/before/A_calibration_report.json')
        self.assertFalse(report['ready'])
        self.assertEqual(report['classes'][0]['samples'],5)
        self.assertFalse((self.store.path(project)/'work/M1/before/A_calibration.json').exists())
        self.assertEqual(self.store.get(project)['completed_rounds'],[])

    def test_class_subset_keeps_motorcycle_as_signed_ignore_and_preserves_model_class_ids(self):
        classes=['car','pedestrian','motorcycle']
        write(self.dataset/'class_schema.json',dict(data_type='3d',annotation_type='cuboid',classes=classes))
        project=self.store.create('fixture',self.dataset)
        for adapter in self.adapters.values():
            target=adapter.predict_target;source=adapter.predict_source_validation
            def predict(checkpoint,frames,target=target):
                raw=target(checkpoint,frames)
                for doc in raw.values():
                    doc['boxes'].append(dict(id='moto',class_name='motorcycle',box=[12.,0.,0.,2.,1.,1.,0.],
                                             score=.9,cls_score=.9,iou_score=.9))
                return raw
            def validation(checkpoint,source=source):
                records=source(checkpoint)
                records['boxes'] += [dict(b,class_name='pedestrian') for b in records['boxes']]
                return records  # No motorcycle calibration samples, deliberately.
            adapter.predict_target=predict;adapter.predict_source_validation=validation
        self.controller(project,adaptation_classes=['car','pedestrian']).run_one()
        labels=read(self.store.path(project)/'rounds/M1/labels.json')
        frame=labels['frames'][0]
        self.assertEqual(labels['adaptation_classes'],['car','pedestrian'])
        self.assertEqual(labels['excluded_classes'],['motorcycle'])
        self.assertEqual({b['class_name'] for b in frame['boxes']},{'car'})
        self.assertTrue(frame['ignored'])
        self.assertTrue(all(b['class_name']=='motorcycle' and b['reason']=='excluded_adaptation_class'
                            and 'quality' not in b and 'confidence' not in b for b in frame['ignored']))
        signed,_,_=signed_labels(frame,classes)
        self.assertIn(1.,signed[:,-1]);self.assertIn(-3.,signed[:,-1])
        self.assertNotIn(3.,signed[:,-1])
        self.assertEqual(self.store.get(project)['schema']['classes'],classes)
        metrics=read(self.store.path(project)/'rounds/M1/round.json')['metrics']
        self.assertEqual(metrics['agreement'],1.)

    def test_automatic_execution_pauses_for_manual_label_review(self):
        project = self.store.create("fixture", self.dataset, max_rounds=3, execution_mode="automatic")
        self.assertEqual(len(self.controller(project).run()), 1)
        self.assertIsNone(self.store.get(project)["selected_version"])

    def test_failed_adapter_artifacts_are_preserved_and_not_auto_restarted(self):
        project = self.store.create("fixture", self.dataset)
        def fail(*args):
            raise RuntimeError("fixture failure")
        self.adapters["A"].train_target = fail
        controller = self.controller(project)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            controller.run_one()
        self.assertTrue((self.store.path(project) / "work/M1/initial_shared.json").exists())
        with self.assertRaisesRegex(ValueError, "artifacts"):
            controller.run_one()

    def test_explicit_new_attempt_preserves_failed_round_and_records_work_path(self):
        project = self.store.create('fixture', self.dataset)
        original = self.adapters['A'].train_target
        def fail(*args):
            raise RuntimeError('interrupted fixture')
        self.adapters['A'].train_target = fail
        with self.assertRaisesRegex(RuntimeError, 'interrupted fixture'):
            self.controller(project).run_one()
        base = self.store.path(project) / 'work/M1'
        preserved = {str(p.relative_to(base)): p.read_bytes() for p in base.rglob('*') if p.is_file()}
        self.adapters['A'].train_target = original
        result = self.controller(project, attempt_id='explicit_retry').run_one()
        self.assertEqual(result['version'], 'M1')
        for name, content in preserved.items():
            self.assertEqual((base / name).read_bytes(), content)
        work = base / 'attempts/explicit_retry'
        self.assertTrue((work / 'A/test_fixture.pth').exists())
        self.assertEqual(read(self.store.path(project) / 'rounds/M1/round.json')['work_directory'], str(work))
        self.assertEqual(self.store.get(project)['completed_rounds'], ['M1'])

    def test_attempt_identifier_cannot_escape_project(self):
        project = self.store.create('fixture', self.dataset)
        with self.assertRaisesRegex(ValueError, 'Attempt ID'):
            self.controller(project, attempt_id='../escape')

    def test_resume_uses_completed_pair_without_training_again(self):
        project=self.store.create('fixture',self.dataset)
        original=self.adapters['B'].predict_target;calls=[]
        def interrupt_after(checkpoint,frames):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('interrupted after training')
            return original(checkpoint,frames)
        self.adapters['B'].predict_target=interrupt_after
        with self.assertRaisesRegex(RuntimeError,'after training'):self.controller(project).run_one()
        old=self.store.path(project)/'work/M1'
        keep={str(p):digest(p) for p in old.rglob('*') if p.is_file()}
        continuation=dict(index=1,before_cache=str(old/'before/predictions.json'),before_sha256=digest(old/'before/predictions.json'),
            labels_path=str(old/'initial_shared.json'),labels_sha256=digest(old/'initial_shared.json'),
            completed={k:read(old/(k+'_trained.json')) for k in ('A','B')},checkpoints={})
        self.adapters['B'].predict_target=original
        result=self.controller(project,attempt_id='continued',resume_plan=continuation).run_one()
        self.assertEqual(result['version'],'M1')
        self.assertEqual([e[0] for e in self.events if e[1]=='train'],['A','B'])
        for path,checksum in keep.items():self.assertEqual(digest(path),checksum)

    def test_resume_keeps_completed_a_and_retrains_only_b_without_checkpoint(self):
        project=self.store.create('fixture',self.dataset)
        original=self.adapters['B'].train_target
        def interrupt_b(*args):raise RuntimeError('B interrupted before save')
        self.adapters['B'].train_target=interrupt_b
        with self.assertRaisesRegex(RuntimeError,'B interrupted'):self.controller(project).run_one()
        old=self.store.path(project)/'work/M1'
        keep={str(p):digest(p) for p in old.rglob('*') if p.is_file()}
        continuation=dict(index=1,before_cache=str(old/'before/predictions.json'),before_sha256=digest(old/'before/predictions.json'),
            labels_path=str(old/'initial_shared.json'),labels_sha256=digest(old/'initial_shared.json'),
            completed={'A':read(old/'A_trained.json')},checkpoints={})
        self.adapters['B'].train_target=original
        result=self.controller(project,attempt_id='continued_b',resume_plan=continuation).run_one()
        self.assertEqual(result['version'],'M1')
        self.assertEqual([e[0] for e in self.events if e[1]=='train'],['A','B'])
        self.assertNotIn('resume_checkpoint',next(e[3] for e in self.events if e[0]=='B' and e[1]=='train'))
        for path,checksum in keep.items():self.assertEqual(digest(path),checksum)

    def test_automatic_resume_accepts_m1_and_trains_m2_without_manual_selection(self):
        project=self.store.create('fixture',self.dataset,max_rounds=2,execution_mode='automatic',
                                  policy=SelectionPolicy(mode='automatic'))
        original=self.adapters['B'].predict_target;calls=[]
        def interrupt_after(checkpoint,frames):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('interrupted after training')
            return original(checkpoint,frames)
        self.adapters['B'].predict_target=interrupt_after
        with self.assertRaisesRegex(RuntimeError,'after training'):self.controller(project).run_one()
        old=self.store.path(project)/'work/M1'
        continuation=dict(index=1,before_cache=str(old/'before/predictions.json'),before_sha256=digest(old/'before/predictions.json'),
            labels_path=str(old/'initial_shared.json'),labels_sha256=digest(old/'initial_shared.json'),
            completed={k:read(old/(k+'_trained.json')) for k in ('A','B')},checkpoints={})
        self.adapters['B'].predict_target=original
        controller=self.controller(project,attempt_id='continued_auto',resume_plan=continuation)
        result=controller.run()
        self.assertEqual([r['version'] for r in result],['M1','M2'])
        self.assertTrue(all(r['decision']['action']=='accept' for r in result))
        self.assertEqual(self.store.get(project)['selected_version'],'M2')
        self.assertIsNone(controller.resume_plan)
        self.assertEqual([e[0] for e in self.events if e[1]=='train'],['A','B','A','B'])
        before_m2=read(self.store.path(project)/'work/M2/attempts/continued_auto/before/predictions.json')
        self.assertEqual(before_m2['checkpoint_hashes'],
                         {k:m['sha256'] for k,m in read(self.store.path(project)/'rounds/M1/round.json')['checkpoints'].items()})

    def test_first_automatic_round_rejected_by_threshold_does_not_train_m2(self):
        project=self.store.create('fixture',self.dataset,max_rounds=3,execution_mode='automatic',
                                  policy=SelectionPolicy(mode='automatic',min_agreement=.6))
        controller=self.controller(project)
        original=controller.shared
        def low_agreement(*args):
            labels,metrics=original(*args);metrics['agreement']=.005250965250965251
            return labels,metrics
        controller.shared=low_agreement
        results=controller.run()
        self.assertEqual(len(results),1)
        self.assertEqual(results[0]['decision']['action'],'reject')
        self.assertIn('agreement_below_threshold',results[0]['decision']['reasons'])
        self.assertIsNone(self.store.get(project)['selected_version'])
        self.assertEqual([e[0] for e in self.events if e[1]=='train'],['A','B'])
        self.assertFalse((self.store.path(project)/'work/M2').exists())

    def test_memory_retention_does_not_inflate_agreement_or_force_auto_selection(self):
        project=self.store.create('fixture',self.dataset,max_rounds=2,execution_mode='automatic',
                                  policy=SelectionPolicy(mode='automatic'))
        for adapter in self.adapters.values():
            original=adapter.predict_target
            def disappear(checkpoint,frames,original=original):
                raw=original(checkpoint,frames)
                if '/work/M2/' in str(checkpoint):
                    for document in raw.values():document['boxes']=[]
                return raw
            adapter.predict_target=disappear
        results=self.controller(project).run()
        root=self.store.path(project)
        first=read(root/'rounds/M1/labels.json')['frames'][0]['boxes'][0]
        last=read(root/'rounds/M2/labels.json')['frames'][0]
        self.assertEqual(last['boxes'][0]['id'],first['id'])
        self.assertEqual(last['boxes'][0]['reason'],'memory_retained_missing')
        self.assertFalse(last['ignored'])
        receipt=read(root/'rounds/M2/round.json')
        self.assertEqual(receipt['metrics']['box_count'],1)
        self.assertEqual(receipt['metrics']['current_consensus_box_count'],0)
        self.assertEqual(receipt['metrics']['agreement'],0)
        self.assertEqual(receipt['metrics']['memory']['retained'],1)
        self.assertEqual(results[1]['decision']['action'],'reject')
        self.assertEqual(self.store.get(project)['selected_version'],'M1')
        self.assertEqual(receipt['memory_sha256'],digest(root/'rounds/M2/memory.json'))
        self.assertEqual(len(read(root/'rounds/M2/memory.json')['frames'][0]['objects'][0]['history']),3)

    def test_rejected_round_memory_is_observed_but_not_used_as_training_labels(self):
        project=self.store.create('fixture',self.dataset,max_rounds=2,execution_mode='automatic',
                                  policy=SelectionPolicy(mode='automatic'))
        controller=self.controller(project);original=controller.shared;calls=[]
        def reject_second(*args):
            labels,metrics=original(*args);calls.append(1)
            if len(calls)==3:metrics['agreement']=.1
            return labels,metrics
        controller.shared=reject_second
        controller.run()
        root=self.store.path(project)
        self.assertEqual(self.store.get(project)['selected_version'],'M1')
        self.store.configure(project,'fixture',3,'automatic')
        self.controller(project).run_one()
        before=read(root/'work/M3/memory_before.json')
        self.assertEqual(before['round_index'],2)
        training=[e for e in self.events if e[1]=='train'][-2:]
        self.assertTrue(all(e[2]==digest(root/'rounds/M1/labels.json') for e in training))

    def test_memory_can_be_disabled_without_changing_normal_round_selection(self):
        project=self.store.create('fixture',self.dataset,max_rounds=1,execution_mode='automatic',
                                  policy=SelectionPolicy(mode='automatic'))
        self.controller(project,memory_policy={'enabled':False}).run()
        root=self.store.path(project)
        self.assertFalse((root/'rounds/M1/memory.json').exists())
        self.assertEqual(self.store.get(project)['selected_version'],'M1')

    def test_final_checkpoint_policy_does_not_extend_epochs(self):
        class Optimizer:
            def step(self):
                return None
        optimizer = Optimizer()
        steps = []
        def trainer(model, optimizer, total_epochs, ckpt_save_dir, start_iter=0, rank=0):
            for _ in range(total_epochs):
                optimizer.step()
                steps.append(1)
        def state(model, optimizer, epoch, iterations):
            return dict(epoch=epoch, iterations=iterations)
        def save(value, filename):
            write(filename + ".pth", value)
        wrapped = with_final_checkpoint(trainer, state, save)
        wrapped(None, optimizer, 13, self.root)
        self.assertEqual(len(steps), 13)
        self.assertEqual(read(self.root / "checkpoint_epoch_13.pth")["iterations"], 13)

    def test_failed_trainer_never_claims_final_checkpoint(self):
        class Optimizer:
            def step(self):
                pass
        def trainer(optimizer, total_epochs, ckpt_save_dir):
            raise RuntimeError("interrupted")
        wrapped = with_final_checkpoint(trainer, lambda *args: {}, lambda *args, **kwargs: self.fail())
        with self.assertRaises(RuntimeError):
            wrapped(Optimizer(), 13, self.root)
        self.assertFalse((self.root / "checkpoint_epoch_13.pth").exists())


if __name__ == "__main__":
    unittest.main()
