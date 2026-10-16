"""Pinned ST3D source initializers, import/download contracts and project lineage."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import yaml
from annoloop.io import read,write,digest
from annoloop.native_sources import catalog,install,root_for
from annoloop.reused.range_zip import RemoteZip
import test_setup as setup_fixtures


class NativeSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.body=b'fixture checkpoint bytes only'
        self.model=dict(catalog()['models'][0],id='lsf-test',bytes=len(self.body),
                        sha256=hashlib.sha256(self.body).hexdigest(),member='ckpt/test.pth')
        p=patch('annoloop.native_sources.entry',return_value=self.model);p.start();self.addCleanup(p.stop)

    def test_catalog_pins_profiles_and_keeps_vehicle_semantics(self):
        models=catalog()['models'];self.assertEqual(len(models),2)
        self.assertTrue(all(m['classes']==['Vehicle'] and m['features']==['x','y','z'] for m in models))
        self.assertTrue(all('remote_not_rechecked' in m['availability'] for m in models))

    def test_import_is_explicit_and_does_not_fake_gpu_or_map_car(self):
        path=self.root/'local.pth';path.write_bytes(self.body)
        with patch('urllib.request.urlopen',side_effect=AssertionError('no network')), \
             patch('subprocess.Popen',side_effect=AssertionError('no worker')):
            metadata=read(install(self.root/'state',self.model['id'],path))
        self.assertEqual(metadata['classes'],['Vehicle']);self.assertFalse(metadata['runtime_verified'])
        self.assertFalse(metadata['forward_backward_verified'])
        self.assertEqual(digest(metadata['checkpoint']),self.model['sha256'])

    def test_wrong_payload_never_registers_class_verified_metadata(self):
        path=self.root/'wrong.pth';path.write_bytes(b'wrong')
        with self.assertRaises(ValueError):install(self.root/'state',self.model['id'],path)
        root=root_for(self.root/'state',self.model['id'])
        self.assertFalse((root/'checkpoint.metadata.json').exists())
        self.assertEqual(read(root/'download.json')['status'],'failed')

    def test_zip_fetch_reads_only_selected_member_and_checks_final_sha(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('unrelated.pth',b'not this one');archive.writestr(self.model['member'],self.body)
        result=install(self.root/'state',self.model['id'],remote_factory=lambda m:io.BytesIO(data.getvalue()))
        self.assertEqual(Path(read(result)['checkpoint']).read_bytes(),self.body)
        self.assertFalse((self.root/'unrelated.pth').exists())

    def test_range_reader_rejects_whole_file_and_changed_range(self):
        class Response(io.BytesIO):
            status=206
            headers={'Content-Range':'bytes 0-0/100','ETag':'"one"'}
        r=RemoteZip('https://drive.usercontent.google.com/test',opener=lambda *a,**k:Response(b'x'))
        with self.assertRaises(ValueError):r.read(10)
        class Whole(Response):status=200
        with self.assertRaises(ValueError):RemoteZip('https://drive.usercontent.google.com/test',opener=lambda *a,**k:Whole(b'x'))


class NativeBindingTests(unittest.TestCase):
    # Reuse source/target setup helpers without inheriting unrelated tests.
    setUp=setup_fixtures.SetupTests.setUp
    models=setup_fixtures.SetupTests.models
    project=setup_fixtures.SetupTests.project

    def fixture_bind(self):
        from annoloop.native_source_binding import bind
        pid=self.project();weight=self.root/'initializer.pth';weight.write_bytes(b'source initializer fixture')
        fixture=dict(catalog()['models'][0],bytes=weight.stat().st_size,sha256=digest(weight))
        a=patch('annoloop.native_sources.entry',return_value=fixture)
        b=patch('annoloop.native_source_binding.entry',return_value=fixture)
        a.start();b.start();self.addCleanup(a.stop);self.addCleanup(b.stop)
        install(self.settings.data,fixture['id'],weight)
        return pid,fixture,bind(self.store,pid,self.settings,'A',fixture['id'])

    def test_selection_uses_only_checkpoints_from_the_selected_initializer(self):
        from annoloop import adaptation
        pid,fixture,setup=self.fixture_bind()
        old=self.root/'old.pth';old.write_bytes(b'old different initializer')
        old_meta=self.root/'old.metadata.json'
        write(old_meta,dict(checkpoint=str(old),checkpoint_sha256=digest(old),classes=['car','pedestrian'],classes_verified=True))
        run=self.root/'run';write(run/'job.json',dict(spec=dict(checkpoint_metadata=str(old_meta))))
        bmeta=read(setup['model_metadata']['B']);cfg_b=self.root/'B.yaml';cfg_b.write_text('fixture: true\n')
        a=dict(path=str(old),config=setup['specs']['A']['config'],runtime='fixture',classes=['car','pedestrian'],name='old',origin_run=str(run))
        b=dict(path=bmeta['checkpoint'],config=str(cfg_b),runtime='fixture',classes=['car','pedestrian'],name='B')
        intent={'A':dict(kind='source',source_id=fixture['id']),'B':dict(kind='local',path=b['path'])}
        with patch('annoloop.adaptation.model_choices',return_value={'A':[a],'B':[b]}):
            result=adaptation.save_model_selection(self.store,pid,self.settings,intent)
            self.assertNotIn('A',result['models'])
            # A newly trained child of this exact initializer can satisfy intent.
            write(run/'job.json',dict(spec=dict(checkpoint_metadata=setup['specs']['A']['checkpoint_metadata'])))
            result=adaptation.save_model_selection(self.store,pid,self.settings,intent)
            self.assertIn('A',result['models'])
        self.assertFalse((self.store.path(pid)/'native_verified.json').exists())
    def test_bind_missing_classes_creates_training_spec_without_launch(self):
        from annoloop.native_source_binding import bind
        pid=self.project();project_before=read(self.store.path(pid)/'project.json')
        fixture=catalog()['models'][0];weight=self.root/'init.pth';weight.write_bytes(b'fixture only')
        fixture=dict(fixture,bytes=weight.stat().st_size,sha256=digest(weight))
        with patch('annoloop.native_sources.entry',return_value=fixture), \
             patch('annoloop.native_source_binding.entry',return_value=fixture), \
             patch('annoloop.jobs.start_pretrain') as launch:
            install(self.settings.data,fixture['id'],weight)
            setup=bind(self.store,pid,self.settings,'A',fixture['id'])
            self.assertTrue(setup['plan']['models']['A']['pretrain_required'])
            self.assertEqual(setup['plan']['models']['A']['missing_classes'],['car','pedestrian'])
            config=yaml.safe_load(Path(setup['specs']['A']['config']).read_text())
            self.assertEqual(config['CLASS_NAMES'],['car','pedestrian'])
            metadata=read(setup['specs']['A']['checkpoint_metadata'])
            self.assertEqual(metadata['classes'],['Vehicle'])
            self.assertNotIn('SELF_TRAIN',config);self.assertNotIn('DATA_CONFIG_TAR',config)
            self.assertEqual(project_before,read(self.store.path(pid)/'project.json'))
            self.assertFalse((self.store.path(pid)/'native_verified.json').exists())
            launch.assert_not_called()
            before=(self.store.path(pid)/'setup.json').read_bytes()
            with patch('annoloop.adaptation.active_job',return_value='active'),self.assertRaises(ValueError):
                bind(self.store,pid,self.settings,'A',fixture['id'])
            self.assertEqual(before,(self.store.path(pid)/'setup.json').read_bytes())


if __name__=='__main__':unittest.main()
