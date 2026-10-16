"""Real fixture file connection, no downloads outside mocked upstream or GPU."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from annoloop.connect_model import connect,main
from annoloop.io import read,write
from annoloop.model_hub import download,model_root
from test_model_hub import model,Response,CONFIG


class ConnectTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.data=Path(self.tmp.name)
        self.model=dict(model(),framework='mmdetection3d',modality='point_cloud')

    def install(self):
        return download(self.data,self.model,opener=lambda request,timeout:
            Response(CONFIG,url=request.full_url) if 'raw.githubusercontent' in request.full_url else Response())

    def test_installed_model_connects_to_real_raw_handler_without_claiming_training(self):
        self.install()
        with patch('subprocess.run',side_effect=AssertionError('no process')), \
             patch('urllib.request.urlopen',side_effect=AssertionError('no network')):
            report=connect(self.data,self.model)
        self.assertEqual(report['status'],'connected_unverified')
        self.assertEqual(report['entrypoints']['predict_raw'],'annoloop.adapter_factory:FrameworkAdapter.predict_raw')
        self.assertIsNone(report['entrypoints']['train_target'])
        self.assertFalse(report['capabilities']['calibration_implemented'])
        self.assertFalse(report['capabilities']['signed_ignore_implemented'])
        self.assertFalse(report['adaptation_ready']);self.assertFalse(report['gpu_verified'])
        self.assertTrue(Path(report['adapter']).exists())

    def test_missing_or_corrupted_weights_do_not_get_connection(self):
        with self.assertRaises(ValueError):connect(self.data,self.model)
        receipt=self.install();Path(receipt['weights']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):connect(self.data,self.model)
        self.assertFalse((model_root(self.data,self.model['id'])/'connection.json').exists())

    def test_reconnection_preserves_earlier_reports(self):
        self.install();first=connect(self.data,self.model);before=Path(first['receipt']).read_bytes()
        second=connect(self.data,self.model)
        self.assertNotEqual(first['receipt'],second['receipt'])
        self.assertEqual(before,Path(first['receipt']).read_bytes())
        self.assertEqual(read(model_root(self.data,self.model['id'])/'connection.json'),second)

    def test_unsupported_modality_report_does_not_invent_binding(self):
        self.model['modality']='lidar_camera';self.install()
        report=connect(self.data,self.model)
        self.assertEqual(report['status'],'unsupported');self.assertIsNone(report['adapter'])
        self.assertFalse((model_root(self.data,self.model['id'])/'adapter.json').exists())

    def test_all_installed_cli_never_downloads_absent_model(self):
        self.install();other=dict(self.model,id='absent')
        with patch('annoloop.connect_model.load_catalog',return_value={'models':[self.model,other]}), \
             patch('urllib.request.urlopen',side_effect=AssertionError('no download')),redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--all-installed','--data',str(self.data)]),0)
        self.assertIn('connected_unverified',output.getvalue())
        self.assertFalse((model_root(self.data,other['id'])/'adapter.json').exists())

    def test_project_check_does_not_change_selection_or_read_gold(self):
        from annoloop.projects import ProjectStore
        import numpy as np
        folder=self.data/'target';folder.mkdir()
        np.save(folder/'one.npy',np.ones((3,3),np.float32))
        store=ProjectStore(self.data)
        from annoloop.schema import ClassSchema
        pid=store.create('fixture',folder,schema_override=ClassSchema('3d','cuboid',('car',)))
        write(store.path(pid)/'setup.json',dict(source_counts={'train':{'car':2},'validation':{'car':1}}))
        write(store.path(pid)/'adaptation.json',dict(models={},model_selection={},adaptation_classes=['car']))
        original=(store.path(pid)/'adaptation.json').read_bytes()
        self.install();report=connect(self.data,self.model,pid)
        self.assertEqual(report['project_check']['input_profile']['features'],['x','y','z'])
        self.assertFalse(report['target_gold_read']);self.assertFalse(report['adaptation_ready'])
        self.assertEqual(original,(store.path(pid)/'adaptation.json').read_bytes())
        self.assertFalse((store.path(pid)/'native_verified.json').exists())

    def test_download_cli_calls_connection_after_install(self):
        import sys
        from annoloop.model_hub import main as hub_main
        # Download is replaced only at transport boundary; connection consumes
        # installed fixture hashes/configs, just as in the real CLI.
        with patch('annoloop.model_hub.load_catalog',return_value={'models':[self.model]}), \
             patch('annoloop.model_hub.download',side_effect=lambda *a,**k:self.install()), \
             patch.object(sys,'argv',['model_hub','download',self.model['id'],'--data',str(self.data)]), \
             redirect_stdout(io.StringIO()):
            hub_main()
        self.assertTrue((model_root(self.data,self.model['id'])/'connection.json').exists())


if __name__=='__main__':unittest.main()
