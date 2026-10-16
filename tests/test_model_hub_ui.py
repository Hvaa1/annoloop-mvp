"""The architecture-neutral library works before project setup and is action-gated."""
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ui'))


class LibraryUITest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.data=Path(self.tmp.name)/'state'
        env=patch.dict(os.environ,{'ANNOLOOP_DATA_DIR':str(self.data)});env.start();self.addCleanup(env.stop)
        self.app=AppTest.from_file(str(ROOT/'ui/app.py'),default_timeout=20)
    def open_library(self):
        self.app.run()
        next(r for r in self.app.radio if r.label=='Không gian làm việc').set_value('Kho model').run()
        self.assertFalse(self.app.exception)
    def test_view_and_filter_never_download_or_run_models(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('No network on view')):
            self.open_library()
            next(m for m in self.app.multiselect if m.label=='Class bạn cần gán').set_value(['car','pedestrian']).run()
            next(s for s in self.app.selectbox if s.label=='Kiến trúc').select('CenterPoint').run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any('adapter' in w.value.lower() for w in self.app.warning))
        self.assertFalse((self.data/'model_cache').exists())
        self.assertTrue(any(b.label=='Tải / tiếp tục tải gói model' for b in self.app.button))
    def test_explicit_download_dispatches_worker_once(self):
        with patch('model_library.start_download') as launch:
            self.open_library();self.assertEqual(launch.call_count,0)
            next(b for b in self.app.button if b.label=='Tải / tiếp tục tải gói model').click().run()
            self.assertFalse(self.app.exception);self.assertEqual(launch.call_count,1)
    def test_uncovered_class_does_not_silently_select_a_model(self):
        self.open_library()
        next(m for m in self.app.multiselect if m.label=='Class bạn cần gán').set_value(['unknown_new_class']).run()
        next(c for c in self.app.checkbox if c.label=='Chỉ hiện model hỗ trợ đủ class đã chọn').check().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(any(b.label=='Tải / tiếp tục tải gói model' for b in self.app.button))
    def test_adapter_preparation_requires_explicit_selection(self):
        with patch('model_library.start_download') as launch:
            self.open_library();launch.assert_not_called()
            next(b for b in self.app.button if b.label=='Chọn model & chuẩn bị adapter').click().run()
            self.assertFalse(self.app.exception);self.assertEqual(launch.call_count,1)
            self.assertTrue(launch.call_args.kwargs['prepare'])


if __name__=='__main__':unittest.main()
