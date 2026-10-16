"""Model downloads: real byte contracts, safe resume and class coverage semantics."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from annoloop.io import read,write
from annoloop.model_hub import (recommendations,load_catalog,download,installed,model_root,status)


BODY=b'PK\x03\x04'+bytes(range(256))*40
CONFIG=b'class_names = ["car", "pedestrian"]\n'


def model(classes=('car','pedestrian')):
    return dict(id='test-model',name='test',architecture='AnyDetector',framework='test',dataset='test',
        weights_url='https://download.openmmlab.com/test.pth',classes=list(classes),runtime_verified=False,
        runtime_status='adapter_required',revision='pinned',source_url='https://github.com/open-mmlab/mmdetection3d',
        config_path='configs/test.py',config_files={'configs/test.py':dict(url='https://raw.githubusercontent.com/a/b/pinned/configs/test.py',sha256=hashlib.sha256(CONFIG).hexdigest())},
        publisher_sha256=None,download_verification=dict(status='binary_reachable',prefix_bytes=4096,prefix_sha256=hashlib.sha256(BODY[:4096]).hexdigest()))


class Response(io.BytesIO):
    def __init__(self,body=BODY,code=200,headers=None,url='https://download.openmmlab.com/test.pth'):
        super().__init__(body);self.status=code;self.headers=headers or {'Content-Length':str(len(body)),'ETag':'"v1"'};self.url=url
    def geturl(self):return self.url


class HubTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.data=Path(self.tmp.name)
    def opener(self,request,timeout):
        return Response(CONFIG,url=request.full_url) if 'raw.githubusercontent' in request.full_url else Response()
    def test_bundled_registry_has_only_verified_binaries_and_diverse_models(self):
        catalog=load_catalog();self.assertGreaterEqual(len(catalog['models']),40)
        self.assertGreaterEqual(len({m['architecture'] for m in catalog['models']}),10)
        self.assertTrue(all(m['download_verification']['prefix_sha256'] for m in catalog['models']))
        self.assertTrue(all(not m['runtime_verified'] for m in catalog['models']))
    def test_vehicle_not_equivalent_to_car_and_cyclist_not_motorcycle(self):
        row=recommendations(['car','motorcycle'],catalog={'models':[model(['Vehicle','Cyclist'])]})[0]
        self.assertEqual(row['missing_classes'],['car','motorcycle']);self.assertTrue(row['mapping_notes'])
    def test_multimodal_not_recommended_for_point_cloud_input(self):
        a=dict(model(),modality='lidar_camera')
        self.assertFalse(recommendations(['car'],catalog={'models':[a]}))
        self.assertEqual(len(recommendations(['car'],catalog={'models':[a]},modality='lidar_camera')),1)
    def test_case_normalization_and_coverage_ranking_not_cross_dataset_ap(self):
        a=model(['Car','Pedestrian']);b=dict(model(['Car']),id='car-only',upstream_metrics={'AP':100})
        rows=recommendations(['car','pedestrian'],catalog={'models':[b,a]})
        self.assertEqual(rows[0]['id'],'test-model');self.assertTrue(rows[0]['all_classes_supported'])
    def test_install_records_digest_and_configs_without_claiming_native_ready(self):
        m=model();receipt=download(self.data,m,opener=self.opener)
        self.assertEqual(receipt['sha256'],hashlib.sha256(BODY).hexdigest())
        self.assertTrue(installed(self.data,m));self.assertFalse(receipt['runtime_verified'])
        with patch('annoloop.model_hub.urllib.request.urlopen',side_effect=AssertionError('No network')):
            download(self.data,m,opener=lambda *a,**k:self.fail('Cached install made network request'))
    def test_resume_validated_range_appends_only_tail(self):
        m=model();root=model_root(self.data,m['id']);root.mkdir(parents=True)
        (root/'weights.pth.part').write_bytes(BODY[:5000]);write(root/'partial.json',dict(url=m['weights_url'],etag='"v1"'))
        def opener(request,timeout):
            if 'raw.githubusercontent' in request.full_url:return self.opener(request,timeout)
            self.assertEqual(request.get_header('Range'),'bytes=5000-')
            self.assertEqual(request.get_header('If-range'),'"v1"')
            return Response(BODY[5000:],206,{'Content-Range':f'bytes 5000-{len(BODY)-1}/{len(BODY)}','ETag':'"v1"'})
        download(self.data,m,opener=opener)
        self.assertEqual((root/'weights.pth').read_bytes(),BODY)
    def test_server_ignoring_range_restarts_instead_of_concatenating(self):
        m=model();root=model_root(self.data,m['id']);root.mkdir(parents=True)
        (root/'weights.pth.part').write_bytes(BODY[:5000]);write(root/'partial.json',dict(url=m['weights_url'],etag='"v1"'))
        download(self.data,m,opener=self.opener);self.assertEqual((root/'weights.pth').read_bytes(),BODY)
    def test_bad_range_or_changed_version_does_not_promote(self):
        for headers in [{'Content-Range':f'bytes 4999-{len(BODY)-1}/{len(BODY)}','ETag':'"v1"'},
                        {'Content-Range':f'bytes 5000-{len(BODY)-1}/{len(BODY)}','ETag':'"v2"'}]:
            m=model();root=model_root(self.data,m['id']);root.mkdir(parents=True,exist_ok=True)
            (root/'weights.pth.part').write_bytes(BODY[:5000]);write(root/'partial.json',dict(url=m['weights_url'],etag='"v1"'))
            with self.assertRaises(ValueError):download(self.data,m,opener=lambda *a,**k:Response(BODY[5000:],206,headers))
            self.assertFalse((root/'installed.json').exists());self.assertEqual((root/'weights.pth.part').stat().st_size,5000)
    def test_html_auth_page_is_not_a_checkpoint(self):
        with self.assertRaises(ValueError):download(self.data,model(),opener=lambda *a,**k:Response(b'<html>Login</html>'))
        self.assertFalse(installed(self.data,model()));self.assertEqual(status(self.data,'test-model')['status'],'failed')
    def test_incomplete_transfer_keeps_part_and_never_promotes(self):
        with self.assertRaises(ValueError):download(self.data,model(),opener=lambda *a,**k:Response(BODY[:4096],headers={'Content-Length':str(len(BODY))}))
        root=model_root(self.data,'test-model');self.assertTrue((root/'weights.pth.part').exists());self.assertFalse((root/'weights.pth').exists())
    def test_hash_mismatch_and_cache_corruption(self):
        m=model();m['publisher_sha256']='0'*64
        with self.assertRaises(ValueError):download(self.data,m,opener=self.opener)
        m=model();download(self.data,m,opener=self.opener)
        (model_root(self.data,m['id'])/'weights.pth').write_bytes(b'corrupt')
        self.assertFalse(installed(self.data,m))
    def test_config_hash_mismatch_never_installs(self):
        m=model();m['config_files']['configs/test.py']['sha256']='0'*64
        with self.assertRaises(ValueError):download(self.data,m,opener=self.opener)
        self.assertFalse((model_root(self.data,m['id'])/'installed.json').exists())
    def test_cancel_preserves_partial_and_no_final(self):
        root=model_root(self.data,'test-model');root.mkdir(parents=True);(root/'cancel').touch()
        with self.assertRaises(InterruptedError):download(self.data,model(),opener=self.opener)
        self.assertEqual(status(self.data,'test-model')['status'],'paused');self.assertFalse((root/'weights.pth').exists())
    def test_disallowed_urls_and_path_traversal(self):
        m=model();m['weights_url']='http://127.0.0.1/private'
        with self.assertRaises(ValueError):download(self.data,m,opener=self.opener)
        with self.assertRaises(ValueError):model_root(self.data,'../../escape')


if __name__=='__main__':unittest.main()
