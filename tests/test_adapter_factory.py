"""Lazy preparation must not run/download anything before an explicit action."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import numpy as np
from annoloop.adapter_factory import recipe_for,prepare_adapter,select_and_prepare,load_adapter
from annoloop.framework_worker import points_for,boxes_document
from annoloop.io import read,write,digest
from annoloop.model_hub import download,model_root
from annoloop.settings import Settings
from test_model_hub import model,Response,CONFIG


class AdapterFactoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.m=dict(model(),framework='mmdetection3d',modality='point_cloud')
        self.settings=Settings(self.root,Path('/ui-python'),Path('/launcher'),Path('/model-python'))
    def opener(self,request,timeout):
        return Response(CONFIG,url=request.full_url) if 'raw.githubusercontent' in request.full_url else Response()
    def install_fixture(self,data,m,**kwargs):return download(data,m,opener=self.opener,**kwargs)
    def test_recipe_lookup_is_lazy(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('No network')):
            r=recipe_for(self.m)
        self.assertEqual(r['backend'],'mmdetection3d');self.assertFalse(list(self.root.iterdir()))
    def test_no_install_means_no_prepared_adapter(self):
        with self.assertRaises(ValueError):prepare_adapter(self.root,self.m)
        self.assertFalse(list(self.root.iterdir()))
    def test_explicit_setup_downloads_then_binds_without_gpu(self):
        with patch('subprocess.run',side_effect=AssertionError('No process for binding')):
            p=select_and_prepare(self.root,self.m,downloader=self.install_fixture)
        self.assertEqual(p['status'],'prepared_unverified');self.assertFalse(p['runtime_verified'])
        self.assertEqual(p['classes'],['car','pedestrian'])
        self.assertFalse(p['recipe']['supports_generic_target_training'])
        with patch('annoloop.model_hub.load_catalog',return_value={'models':[self.m]}):a=load_adapter(self.root,'test-model',self.settings)
        self.assertEqual(a.profile['checkpoint_sha256'],digest(a.profile['checkpoint']))
    def test_unknown_framework_rejects_before_download(self):
        m=dict(self.m,framework='Unknown');downloader=Mock()
        with self.assertRaises(ValueError):select_and_prepare(self.root,m,downloader=downloader)
        downloader.assert_not_called();self.assertFalse(list(self.root.iterdir()))
    def test_camera_model_rejects_before_download(self):
        downloader=Mock()
        with self.assertRaises(ValueError):select_and_prepare(self.root,dict(self.m,modality='lidar_camera'),downloader=downloader)
        downloader.assert_not_called()
    def test_failure_preserves_receipt_without_claiming_adapter_ready(self):
        with self.assertRaises(OSError):select_and_prepare(self.root,self.m,downloader=Mock(side_effect=OSError('offline')))
        r=read(model_root(self.root,'test-model')/'selection.json')
        self.assertEqual(r['status'],'failed');self.assertFalse((model_root(self.root,'test-model')/'adapter.json').exists())
    def test_binding_rejects_changed_config(self):
        p=select_and_prepare(self.root,self.m,downloader=self.install_fixture);Path(p['config']).write_text('tampered')
        with patch('annoloop.model_hub.load_catalog',return_value={'models':[self.m]}),self.assertRaises(ValueError):load_adapter(self.root,'test-model',self.settings)
    def test_predict_requires_coordinates_and_does_not_start_implicitly(self):
        select_and_prepare(self.root,self.m,downloader=self.install_fixture)
        with patch('annoloop.model_hub.load_catalog',return_value={'models':[self.m]}):a=load_adapter(self.root,'test-model',self.settings)
        with patch('subprocess.run') as run,self.assertRaises(ValueError):a.predict_raw([{}],self.root/'outputs')
        run.assert_not_called()
    def test_binding_cannot_redirect_to_an_unrelated_config(self):
        p=select_and_prepare(self.root,self.m,downloader=self.install_fixture)
        other=self.root/'other.py';other.write_text('unrelated = True')
        p.update(config=str(other),config_sha256=digest(other))
        write(model_root(self.root,'test-model')/'adapter.json',p)
        with patch('annoloop.model_hub.load_catalog',return_value={'models':[self.m]}),self.assertRaises(ValueError):load_adapter(self.root,'test-model',self.settings)
    def test_features_not_fabricated_or_silently_reordered(self):
        p=self.root/'points.npy';np.save(p,np.ones((2,3),dtype=np.float32));f=dict(media=str(p),sha256=digest(p))
        np.testing.assert_equal(points_for(f,['x','y','z']),np.ones((2,3)))
        with self.assertRaises(ValueError):points_for(f,['x','y','z','intensity'])
        np.save(p,np.array([[10,20,30,40]],dtype=np.float32));f.update(sha256=digest(p),feature_names=['intensity','z','y','x'])
        np.testing.assert_equal(points_for(f,['x','y','z','intensity']),[[40,30,20,10]])
    def test_ambiguous_four_column_array_is_blocked(self):
        p=self.root/'points.npy';np.save(p,np.ones((2,4)));f=dict(media=str(p),sha256=digest(p))
        with self.assertRaises(ValueError):points_for(f,['x','y','z','intensity'])
    def test_serialization_preserves_centers_and_has_no_fake_iou_score(self):
        r=boxes_document([[1,2,3]],[[4,2,1]],[.2],[.8],[1],['car','pedestrian'],frame_id='f',coordinate_frame='LIDAR')
        b=r['boxes'][0];self.assertEqual(b['class_name'],'pedestrian');self.assertEqual(b['box'][:3],[1,2,3])
        self.assertNotIn('iou_score',b);self.assertFalse(r['adaptation_ready'])
    def test_invalid_framework_labels_are_rejected(self):
        with self.assertRaises(ValueError):boxes_document([[0,0,0]],[[1,1,1]],[0],[.9],[3],['car'],frame_id='f',coordinate_frame='LIDAR')


if __name__=='__main__':unittest.main()
