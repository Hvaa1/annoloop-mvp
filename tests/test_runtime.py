"""Import preflight and model-Python selection; no subprocess GPU or trainer."""
import json
import subprocess
import tempfile
from pathlib import Path
import unittest
import types
import os
from unittest.mock import patch
from annoloop.runtime import probe_runtime,worker_environment,configure_sparse_headers
from annoloop.settings import Settings
from annoloop.jobs import start_pretrain
from annoloop.io import read


class RuntimeTests(unittest.TestCase):
    def test_nvrtc_include_path_uses_selected_complete_toolkit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            header=root/'include/cuda/std/cassert';header.parent.mkdir(parents=True);header.touch()
            (root/'lib').mkdir()
            cumm=types.ModuleType('cumm');common=types.ModuleType('cumm.common');cumm.common=common
            with patch.dict(os.environ,{'ANNOLOOP_CUDA_TOOLKIT':str(root)}),patch.dict('sys.modules',{'cumm':cumm,'cumm.common':common}):
                self.assertEqual(configure_sparse_headers(),str(root))
                self.assertEqual(common._CACHED_CUDA_INCLUDE_LIB,(root/'include',root/'lib'))

    def test_missing_libcudacxx_header_fails_preflight_clearly(self):
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'ANNOLOOP_CUDA_TOOLKIT':temporary}):
            with self.assertRaisesRegex(ValueError,'cuda/std/cassert'):configure_sparse_headers()

    def test_worker_exposes_hashed_nvrtc_library_without_changing_wheel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);native=root/'venv/lib/python3.10/site-packages/torch/lib';native.mkdir(parents=True)
            library=native/'libnvrtc-fixture.so.11.8';library.write_bytes(b'fixture')
            settings=Settings(root/'data',root/'ui/python',root/'launcher',root/'venv/bin/python')
            environment=worker_environment(settings)
            aliases=list((settings.data/'runtime_libraries').glob('*/libnvrtc.so'))
            self.assertEqual(len(aliases),1)
            self.assertEqual(aliases[0].resolve(),library)
            self.assertIn(str(aliases[0].parent),environment['LD_LIBRARY_PATH'].split(':'))
            self.assertFalse((native/'libnvrtc.so').exists())

    def test_probe_runs_model_python_with_native_dependency_path(self):
        settings=Settings(Path('/data'),Path('/ui/python'),Path('/launcher'),Path('/model/python'),Path('/tmp'))
        result=subprocess.CompletedProcess([],0,stdout=json.dumps(dict(ready=True)),stderr='')
        with patch('annoloop.runtime.subprocess.run',return_value=result) as run:
            self.assertTrue(probe_runtime({'runtime':'/runtime'},settings)['ready'])
        self.assertEqual(run.call_args.args[0][0],'/model/python')
        self.assertIn('/tmp',run.call_args.kwargs['env']['PYTHONPATH'].split(':'))

    def test_registered_trainer_uses_model_python_while_launcher_uses_ui(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            settings=Settings(root/'data',Path('/ui/python'),Path('/launcher'),Path('/model/python'))
            files={key:str(root/key) for key in ('config','runtime','source_train_annotations','source_validation_annotations')}
            for path in files.values():Path(path).touch()
            with patch('annoloop.runtime.require_runtime',return_value=dict(ready=True)),patch('annoloop.jobs.subprocess.run') as run:
                directory=start_pretrain(files,settings)
            command=run.call_args.args[0]
            self.assertEqual(command[0],'/ui/python')
            self.assertEqual(command[command.index('--')+1],'/model/python')
            self.assertEqual(read(Path(directory)/'job.json')['python'],'/model/python')

    def test_missing_dependency_creates_no_training_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);settings=Settings(root/'data',Path('/python'),Path('/launcher'))
            files={key:str(root/key) for key in ('config','runtime','source_train_annotations','source_validation_annotations')}
            for path in files.values():Path(path).touch()
            with patch('annoloop.runtime.require_runtime',side_effect=ValueError('missing dependency')),patch('annoloop.jobs.subprocess.run') as run:
                with self.assertRaises(ValueError):start_pretrain(files,settings)
                run.assert_not_called()
            self.assertFalse((settings.data/'jobs').exists())
