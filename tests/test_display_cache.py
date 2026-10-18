"""Page-view hash caches must follow file identity and never replace dispatch hashing."""
from pathlib import Path
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from annoloop import adaptation
from annoloop.io import digest, display_digest


class DisplayCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'weights.pth'
        self.path.write_bytes(b'first')
        # Pretend a minute has passed so the file is outside the racy window.
        later = time.time_ns() + 60_000_000_000
        self.clock = patch('annoloop.io.time.time_ns', return_value=later)
        self.clock.start()

    def tearDown(self):
        self.clock.stop()
        self.tmp.cleanup()

    def test_display_digest_reuses_until_file_changes(self):
        self.assertEqual(display_digest(self.path), digest(self.path))
        with patch('annoloop.io.digest', side_effect=AssertionError('rehashed')):
            self.assertEqual(display_digest(self.path), digest(self.path))
        self.path.write_bytes(b'second, longer')
        self.assertEqual(display_digest(self.path), digest(self.path))

    def test_recently_changed_file_is_never_cached(self):
        self.clock.stop()
        self.path.write_bytes(b'fresh')
        display_digest(self.path)
        with patch('annoloop.io.digest', return_value='rehashed') as full:
            self.assertEqual(display_digest(self.path), 'rehashed')
            full.assert_called_once()
        self.clock.start()

    def test_same_size_rewrite_with_restored_mtime_is_rehashed(self):
        # Real clock: a same-tick rewrite keeps the signature, so the racy rule must catch it.
        self.clock.stop()
        display_digest(self.path)
        stat = self.path.stat()
        self.path.write_bytes(b'FIRST')
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(display_digest(self.path), digest(self.path))
        self.clock.start()

    def test_display_contract_invalidates_on_config_or_dataset_change(self):
        root = Path(self.tmp.name)
        (root / 'dataset.json').write_text('{"frames": []}')
        project = dict(dataset_sha256='hash')
        config = dict(bin_columns=4, coordinates_confirmed=True)
        first = adaptation._display_contract(root, project, config)
        with patch('annoloop.adaptation._point_contract', side_effect=AssertionError('recomputed')):
            self.assertEqual(adaptation._display_contract(root, project, config), first)
        with patch('annoloop.adaptation._point_contract', return_value=dict(changed=True)) as full:
            self.assertEqual(adaptation._display_contract(root, project, dict(config, bin_columns=3)),
                             dict(changed=True))
            full.assert_called_once()
        with patch('annoloop.adaptation._point_contract', return_value=dict(fresh=True)) as full:
            (root / 'dataset.json').write_text('{"frames": [], "edited": true}')
            self.assertEqual(adaptation._display_contract(root, project, config), dict(fresh=True))
            full.assert_called_once()


if __name__ == '__main__':
    unittest.main()
