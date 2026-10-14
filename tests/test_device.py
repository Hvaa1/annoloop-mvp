"""Hardware guards with mocked nvidia-smi; never launch GPU work."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from annoloop.device import gpu_idle


class DeviceTests(unittest.TestCase):
    @patch("annoloop.device.subprocess.run")
    def test_busy_gpu_blocks_second_worker(self, command):
        command.return_value = SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        with self.assertRaisesRegex(RuntimeError, "12345"):
            gpu_idle()

    @patch("annoloop.device.subprocess.run")
    def test_unsupported_pid_query_is_not_treated_as_idle(self, command):
        command.return_value = SimpleNamespace(returncode=0, stdout="Not Supported\n", stderr="")
        with self.assertRaisesRegex(RuntimeError, "chưa xác minh"):
            gpu_idle()


if __name__ == "__main__":
    unittest.main()
