"""Saved partial previews are read-only and never imply completed adaptation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from annoloop.io import digest, read, write
from annoloop.projects import ProjectStore
from annoloop import previews


class PreviewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        target = self.root / 'target'; target.mkdir()
        write(target / 'class_schema.json', dict(data_type='3d', annotation_type='cuboid', classes=['car']))
        for fid in ('one', 'two'):
            np.save(target / (fid + '.npy'), np.arange(300, dtype=np.float32).reshape(-1, 3))
        (target / 'annotations.json').write_text('gold must never be parsed')
        self.store = ProjectStore(self.root / 'data'); self.pid = self.store.create('fixture', target)
        self.project = self.store.path(self.pid)
        self.frames = read(self.project / 'dataset.json')['frames']
        self.job = self.project / 'adaptation_jobs/fixture'
        self.work = self.project / 'work/M1/attempts/fixture'
        models = {k: dict(path=k+'.pth', checkpoint_sha256='before_'+k, config_sha256='cfg_'+k,
                         runtime='runtime', classes=['car']) for k in ('A', 'B')}
        self.models = models
        write(self.job / 'spec.json', dict(mode='run', attempt_id='fixture',
            dataset_sha256=self.store.get(self.pid)['dataset_sha256'],
            config=dict(models=models, bin_columns=4, adaptation_classes=['car'])))
        write(self.job / 'state.json', dict(status='running', mode='run', pid=999999, process_start='stale',
            created_at='2026-10-14T00:00:00Z', stage='predict_target', active_detector='B', work_directory=str(self.work)))
        write(self.work / 'before/checkpoints.json', dict(checkpoint_hashes={k:'before_'+k for k in models}))
        for key in models:
            write(self.work / (key+'_trained.json'), dict(checkpoint_sha256='after_'+key, epochs=3))
            for phase in ('before', 'after'):
                directory = self.job / key / 'target' / phase
                spec = dict(models[key], mode='target_sample', checkpoint_sha256=phase+'_'+key, frames=self.frames)
                write(directory / 'spec.json', spec)
                count = 1 if phase=='after' and key=='B' else 2
                for i, frame in enumerate(self.frames[:count]):
                    write(directory / 'frames' / f'{i:06d}.json', dict(frame_id=frame['frame_id'], target_gold_read=False,
                        boxes=[dict(id='box', class_name='car', box=[4.,0.,0.,4.,2.,2.,0.], score=.8)]))
                write(directory / 'inspection.json', dict(status='running', total_frames=2, completed_frames=count))
                if count==2:
                    write(directory / 'result.json', dict(domain='target_unlabelled', checkpoint_sha256=phase+'_'+key))

    def test_interrupted_round_exposes_partial_after_predictions_without_snapshot_or_dispatch(self):
        identity = digest(self.project / 'project.json')
        with patch('annoloop.adaptation.process_identity', return_value=None), \
             patch('annoloop.adaptation.start_job') as dispatch:
            attempts = previews.saved_attempts(self.store, self.pid)
            dispatch.assert_not_called()
        self.assertEqual(len(attempts), 1)
        attempt = attempts[0]
        self.assertEqual(attempt['state']['status'], 'interrupted')
        self.assertTrue(attempt['phases']['after']['A']['complete'])
        b = attempt['phases']['after']['B']
        self.assertEqual((b['completed'], b['total'], b['complete']), (1,2,False))
        self.assertEqual(previews.prediction(b, 'one')['frame_id'], 'one')
        self.assertIsNone(previews.prediction(b, 'two'))
        self.assertEqual(digest(self.project / 'project.json'), identity)
        self.assertFalse((self.project / 'rounds').exists())

    def test_other_checkpoint_or_source_mode_never_appears_as_after_target_prediction(self):
        path = self.job / 'B/target/after/spec.json'; original = read(path)
        for fields in (dict(checkpoint_sha256='wrong'), dict(mode='validation'), dict(config_sha256='wrong')):
            with self.subTest(fields=fields):
                write(path, dict(original, **fields))
                self.assertNotIn('B', previews.saved_attempts(self.store, self.pid)[0]['phases']['after'])

    def test_prediction_wrong_frame_or_gold_is_rejected(self):
        run = previews.saved_attempts(self.store, self.pid)[0]['phases']['after']['B']
        path = Path(run['available']['one']); document = read(path)
        write(path, dict(document, frame_id='two'))
        with self.assertRaisesRegex(ValueError, 'không khớp'):
            previews.prediction(run, 'one')
        write(path, dict(document, target_gold_read=True))
        with self.assertRaisesRegex(ValueError, 'gold'):
            previews.prediction(run, 'one')

    def test_changed_input_frame_evidence_is_not_used_for_overlay(self):
        path = self.job / 'B/target/after/spec.json'; spec = read(path)
        spec['frames'][0]['sha256'] = 'changed'; write(path, spec)
        self.assertNotIn('B', previews.saved_attempts(self.store, self.pid)[0]['phases']['after'])
        frame = self.frames[0]
        points, total = previews.point_preview(frame, 4, max_points=12)
        self.assertEqual(total, 100); self.assertEqual(points.shape, (12,3))
        Path(frame['media']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'đã thay đổi'):
            previews.point_preview(frame, 4)

    def test_inference_progress_keeps_stopped_counts_and_never_reports_live(self):
        state = read(self.job / 'state.json'); state['process_alive'] = False
        progress = previews.inference_progress(self.job, state)
        self.assertEqual((progress['completed'], progress['total'], progress['live']), (1,2,False))

    def test_changed_dataset_snapshot_cannot_be_previewed(self):
        path = self.job / 'spec.json'; spec = read(path); spec['dataset_sha256']='other'; write(path,spec)
        self.assertEqual(previews.saved_attempts(self.store,self.pid),[])

    def test_each_round_is_linked_to_its_own_trained_checkpoint(self):
        next_work=self.project/'work/M2/attempts/fixture'
        write(next_work/'before/checkpoints.json',dict(checkpoint_hashes={k:'after_'+k for k in self.models}))
        for key in self.models:
            write(next_work/(key+'_trained.json'),dict(checkpoint_sha256='M2_'+key,epochs=3))
            root=self.job/key/'target/M2_after'
            write(root/'spec.json',dict(self.models[key],mode='target_sample',checkpoint_sha256='M2_'+key,frames=self.frames))
            for i,frame in enumerate(self.frames):
                write(root/'frames'/f'{i:06d}.json',dict(frame_id=frame['frame_id'],boxes=[
                    dict(id='box',class_name='car',box=[8.,0.,0.,4.,2.,2.,0.],score=.7)]))
            write(root/'result.json',dict(domain='target_unlabelled',checkpoint_sha256='M2_'+key))
        attempts=previews.saved_attempts(self.store,self.pid)
        self.assertEqual([a['version'] for a in attempts],['M2','M1'])
        for a in attempts:
            expected='M2_A' if a['version']=='M2' else 'after_A'
            self.assertEqual(a['phases']['after']['A']['checkpoint_sha256'],expected)
        self.assertEqual(previews.prediction(attempts[0]['phases']['after']['A'],'one')['boxes'][0]['box'][0],8.)

    def test_multi_round_scan_reuses_run_index_but_sees_new_and_removed_frames(self):
        self.test_each_round_is_linked_to_its_own_trained_checkpoint()
        with patch.object(previews,'_target_run',wraps=previews._target_run) as inspect:
            previews.saved_attempts(self.store,self.pid)
            paths=[str(call.args[0]) for call in inspect.call_args_list]
            self.assertEqual(len(paths),len(set(paths)))
        path=self.job/'B/target/after/frames/000001.json'
        write(path,dict(frame_id=self.frames[1]['frame_id'],boxes=[]))
        attempt=next(a for a in previews.saved_attempts(self.store,self.pid) if a['version']=='M1')
        self.assertEqual(attempt['phases']['after']['B']['completed'],2)
        path.unlink()
        attempt=next(a for a in previews.saved_attempts(self.store,self.pid) if a['version']=='M1')
        self.assertEqual(attempt['phases']['after']['B']['completed'],1)


if __name__ == '__main__':
    unittest.main()
