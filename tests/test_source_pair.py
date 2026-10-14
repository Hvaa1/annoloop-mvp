"""Early stop handoff tests with mocked launch/PIDs; no real trainers."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from annoloop import jobs
from annoloop.io import read, write


class SourcePairTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.pair = self.root / 'pair'
        self.runs = {key: self.root / key for key in ('A', 'B')}
        for directory in self.runs.values():
            directory.mkdir()
        write(self.pair / 'pair.json', dict(specs={key: dict(model=key) for key in self.runs},
                                          jobs={}, status='starting', outcomes={}, stop_requested=False))
        self.events = []

    def launch(self, spec):
        key = spec['model']
        self.events.append('launch_' + key)
        return str(self.runs[key])

    def snapshot(self, state, alive=False):
        return dict(training=dict(status=state), process_alive=alive)

    def completed(self, key):
        write(self.runs[key] / 'checkpoint.metadata.json', dict(test_fixture=True))

    def invoke(self, status, idle=None):
        idle = idle or (lambda: self.events.append('gpu_idle'))
        with patch('annoloop.jobs.start_pretrain', side_effect=self.launch), \
             patch('annoloop.jobs.status', side_effect=status), \
             patch('annoloop.jobs.gpu_idle', side_effect=idle), \
             patch('annoloop.jobs.time.sleep'):
            jobs.pair_worker(self.pair)

    def test_stopped_a_waits_for_process_exit_then_starts_b_and_keeps_checkpoint(self):
        weight = self.runs['A'] / 'checkpoints/checkpoint_epoch_10.pth'
        weight.parent.mkdir();weight.write_bytes(b'existing interrupted checkpoint fixture')
        self.completed('B')
        states = iter([self.snapshot('stopped', True), self.snapshot('stopped', False)])
        def status(directory):
            if Path(directory).name == 'A':
                return next(states)
            return self.snapshot('completed')
        self.invoke(status)
        result = read(self.pair / 'pair.json')
        self.assertEqual(self.events, ['launch_A', 'gpu_idle', 'launch_B', 'gpu_idle'])
        self.assertEqual(result['status'], 'completed_with_stops')
        self.assertEqual(result['outcomes']['A']['status'], 'stopped')
        self.assertEqual(result['outcomes']['B']['status'], 'completed')
        self.assertEqual(weight.read_bytes(), b'existing interrupted checkpoint fixture')
        self.assertFalse((self.runs['A'] / 'checkpoint.metadata.json').exists())

    def test_failed_or_interrupted_a_does_not_start_b(self):
        for state in ('failed', 'interrupted'):
            with self.subTest(state=state):
                self.events = []
                with self.assertRaisesRegex(RuntimeError, state):
                    self.invoke(lambda directory: self.snapshot(state))
                self.assertEqual(self.events, ['launch_A'])
                self.assertEqual(read(self.pair / 'pair.json')['status'], 'failed')

    def test_busy_gpu_after_a_exit_blocks_b(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU busy'):
            self.invoke(lambda directory: self.snapshot('stopped'),
                        idle=lambda: (_ for _ in ()).throw(RuntimeError('GPU busy')))
        self.assertEqual(self.events, ['launch_A'])

    def test_cancelling_whole_sequence_never_launches_b(self):
        def status(directory):
            current = read(self.pair / 'pair.json')
            write(self.pair / 'pair.json', dict(current, stop_requested=True))
            return self.snapshot('stopped')
        self.invoke(status)
        self.assertEqual(self.events, ['launch_A', 'gpu_idle'])
        self.assertEqual(read(self.pair / 'pair.json')['status'], 'cancelled')

    def test_b_stopped_finishes_sequence_without_restarting_either_model(self):
        self.completed('A')
        self.invoke(lambda directory: self.snapshot('completed' if Path(directory).name == 'A' else 'stopped'))
        result = read(self.pair / 'pair.json')
        self.assertEqual(result['status'], 'completed_with_stops')
        self.assertEqual(result['outcomes']['B']['status'], 'stopped')
        self.assertEqual(self.events.count('launch_A'), 1)
        self.assertEqual(self.events.count('launch_B'), 1)

    def test_stopping_one_trainer_twice_sends_only_one_signal(self):
        run = self.runs['A']
        write(run / 'training.json', dict(pid=123, process_start='identity'))
        with patch('annoloop.jobs.process_identity', return_value='identity'), patch('annoloop.jobs.os.kill') as kill:
            self.assertTrue(jobs.stop(run));self.assertTrue(jobs.stop(run))
            self.assertEqual(kill.call_count, 1)


if __name__ == '__main__':
    unittest.main()
