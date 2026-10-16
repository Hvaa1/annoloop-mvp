"""Folder-first UI; no actual train or CVAT requests."""
import json
import os
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from streamlit.testing.v1 import AppTest
from annoloop.projects import ProjectStore
from annoloop.io import digest,read,write
from annoloop.selection import SelectionPolicy
from annoloop.settings import Settings
from annoloop.pseudo_memory import empty as empty_memory, update as update_memory, MemoryPolicy
from annoloop.memory_store import context as memory_context
from annoloop.model_filter import dataset_profile

ROOT=Path(__file__).resolve().parents[1]


class UITest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.env=patch.dict(os.environ,{'ANNOLOOP_DATA_DIR':str(self.root/'state')})
        self.env.start();self.addCleanup(self.env.stop)
        preflight=patch('annoloop.runtime.require_runtime',return_value=dict(ready=True))
        preflight.start();self.addCleanup(preflight.stop)
        self.target=self.root/'target';self.target.mkdir()
        (self.target/'001.bin').write_bytes(b'fixture only')
        self.source=self.root/'source'/'v1.0-mini';self.source.mkdir(parents=True)
        for split in ('train','val'):
            infos=[dict(token=split,gt_names=np.array(['car','pedestrian']))]
            (self.source/f'nuscenes_infos_1sweeps_{split}.pkl').write_bytes(pickle.dumps(infos))
        write(self.source/'v1.0-mini/sample.json',[dict(token='train',scene_token='one'),dict(token='val',scene_token='two')])
        self.models={}
        for key,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]:
            weight=self.root/'state/models'/(key+'.pth');weight.parent.mkdir(parents=True,exist_ok=True);weight.write_bytes(b'fixture only')
            path=weight.with_suffix('.metadata.json')
            write(path,dict(checkpoint=str(weight),checkpoint_sha256=digest(weight),classes=['car','pedestrian'],classes_verified=True,architecture=arch))
            self.models[key]=str(path)
        self.app=AppTest.from_file(str(ROOT/'ui/app.py'),default_timeout=20)

    def start_wizard(self):
        self.app.run().radio(key='navigation').set_value('Dữ liệu').run()
        self.app.text_input(key='quick_source').set_value(str(self.source.parent))
        self.app.text_input(key='quick_target').set_value(str(self.target)).run()
        next(b for b in self.app.button if b.label=='Đọc hai thư mục').click().run()
        self.assertFalse(self.app.error,str(self.app.error))
        self.app.multiselect[0].set_value(['car','pedestrian']).run()
        # Catalog also contains the workspace Waymo checkpoints; choose our fixture.
        self.app.selectbox(key='quick_model_A').set_value(self.models['A'])
        self.app.selectbox(key='quick_model_B').set_value(self.models['B'])

    def create(self):
        self.start_wizard()
        next(b for b in self.app.button if b.label=='Lưu dữ liệu & chọn model →').click().run()
        self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
        self.assertEqual(self.app.radio(key='navigation').value,'Class & nguồn')
        # Legacy source-page tests intentionally inspect the existing source plan.
        self.app.radio(key='navigation').set_value('Huấn luyện nguồn').run()
        return ProjectStore(self.root/'state').list()[0]

    def test_empty_pages_and_no_accidental_jobs(self):
        self.app.run()
        for page in ['Tổng quan','Dữ liệu','Class & nguồn','Huấn luyện nguồn','Vòng adaptation','So sánh phiên bản','Xuất CVAT','Hướng dẫn & thiết bị']:
            self.app.radio(key='navigation').set_value(page).run()
            self.assertFalse(self.app.exception,str(self.app.exception))
        self.assertFalse((self.root/'state/jobs').exists())

    def test_local_ensemble_displays_downloads_without_running_or_promoting_them(self):
        p=self.create()
        imported=dict(path='/fixture/downloaded/weights.pth',name='ST3D downloaded fixture',
            architecture='SECONDNetIoU',classes=['Vehicle'],origin='ST3D chính thức · đã tải',
            bytes=143000000,eligible=False,reasons=['Chưa strict-load / calibration'])
        with patch('annoloop.local_ensemble.discover',return_value=[imported]), \
             patch('annoloop.local_ensemble.start') as start:
            self.app.radio(key='navigation').set_value('Class & nguồn').run()
            self.app.radio(key='ensemble_mode_'+p['id']).set_value('Ghép checkpoint local').run()
            self.assertFalse(self.app.exception,str(self.app.exception))
            self.assertTrue(any('ST3D downloaded fixture' in str(frame.value) for frame in self.app.dataframe))
            self.assertTrue(next(b for b in self.app.button if b.label=='Lưu nhóm model & tiếp tục →').disabled)
            self.app.run();start.assert_not_called()

    def test_final_prediction_button_is_explicit_and_page_refresh_never_dispatches(self):
        p=self.create()
        spec=dict(version='M1',frame_ids=['001'],config=dict(models=[{},{}],classes=['car']))
        with patch('annoloop.final_delivery.plan',return_value=spec), \
             patch('annoloop.final_delivery.start') as start:
            self.app.radio(key='navigation').set_value('Xuất CVAT').run()
            self.assertFalse(self.app.exception,str(self.app.exception))
            self.assertFalse(any('Chưa sẵn sàng bàn giao:' in message.value for message in self.app.warning))
            start.assert_not_called()
            next(b for b in self.app.button if b.label=='Chạy final predict & tạo gói bàn giao').click().run()
            self.assertEqual(start.call_count,1)
            self.app.run();self.assertEqual(start.call_count,1)

    def test_cvat_login_works_without_final_package_and_does_not_upload_or_repeat(self):
        p=self.create();pid=p['id']
        account=dict(host='http://localhost:8080',username='fixture_user',user_id=7,
                     checked_at='2026-10-15T00:00:00+00:00')
        with patch('delivery_page.sdk_available',return_value=True), \
             patch('delivery_page.check_login',return_value=account) as login, \
             patch('annoloop.final_delivery.upload') as upload:
            self.app.radio(key='navigation').set_value('Xuất CVAT').run()
            check=next(b for b in self.app.button if b.label=='Kiểm tra đăng nhập CVAT')
            self.assertFalse(check.disabled)
            self.assertTrue(next(b for b in self.app.button if b.label=='Tạo task CVAT').disabled)
            self.app.text_input(key='delivery_user_'+pid).set_value('fixture_user')
            self.app.text_input(key='delivery_password_'+pid).set_value('fixture_secret')
            check.click().run()
            self.assertFalse(self.app.exception,str(self.app.exception))
            login.assert_called_once_with('http://localhost:8080','fixture_user','fixture_secret')
            self.assertTrue(any('đăng nhập thành công: fixture_user' in msg.value for msg in self.app.success))
            self.assertNotIn('fixture_secret',str(self.app.session_state['delivery_login_'+pid]))
            self.app.run();self.assertEqual(login.call_count,1);upload.assert_not_called()

    def test_final_package_visible_and_cvat_upload_requires_explicit_submission(self):
        from annoloop.final_delivery import prepare_export
        from annoloop.schema import ClassSchema
        (self.target/'001.bin').write_bytes(np.zeros((8,4),dtype='<f4').tobytes())
        p=self.create();store=ProjectStore(self.root/'state');pid=p['id']
        root=store.path(pid)/'delivery_jobs/fixture';root.mkdir(parents=True)
        doc=dict(frame_id='001',boxes=[dict(id='car',class_name='car',box=[0.,0.,0.,4.,2.,2.,0.],score=.9)])
        write(root/'frames/0.json',doc)
        inventory=read(store.path(pid)/'dataset.json')
        prepare_export(root,inventory,[doc],ClassSchema('3d','cuboid',('car',)),4)
        write(root/'result.json',dict(purpose='final_delivery',version='M1',scope='non_holdout',frame_ids=['001'],
            frames=[dict(frame_id='001',file='frames/0.json')],
            export_manifest_sha256=digest(root/'exports/shape/export.json')))
        write(root/'state.json',dict(status='completed',result_sha256=digest(root/'result.json')))
        with patch('delivery_page.sdk_available',return_value=True), \
             patch('annoloop.final_delivery.upload',return_value=dict(status='verified',url='http://localhost:8080/tasks/7')) as upload:
            self.app.radio(key='navigation').set_value('Xuất CVAT').run()
            self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
            self.assertTrue(any('1 frame' in msg.value for msg in self.app.success))
            self.app.run();upload.assert_not_called()
            self.app.text_input(key='delivery_user_'+pid).set_value('fixture user')
            self.app.text_input(key='delivery_password_'+pid).set_value('fixture password')
            next(b for b in self.app.button if b.label=='Tạo task CVAT').click().run()
            upload.assert_called_once()
            self.assertTrue(any('Task đã được kiểm tra' in msg.value for msg in self.app.success))

    def test_final_ignore_view_keeps_version_comparison_and_rollback_controls(self):
        from annoloop.final_delivery import prepare_export
        from annoloop.schema import ClassSchema
        (self.target/'001.bin').write_bytes(np.zeros((8,4),dtype='<f4').tobytes())
        p=self.create();store=ProjectStore(self.root/'state');pid=p['id']
        root=store.path(pid)/'delivery_jobs/fixture';root.mkdir(parents=True)
        car=dict(id='car',class_name='car',box=[0.,0.,0.,4.,2.,2.,0.],score=.9)
        ped=dict(id='ped',class_name='pedestrian',box=[10.,0.,0.,.7,.7,1.7,0.],score=.4,
                 confidence=.4,quality=.25,training_policy='ignore_region',reason='low_calibrated_quality')
        doc=dict(frame_id='001',boxes=[car],ignored=[ped])
        write(root/'frames/0.json',doc)
        write(root/'spec.json',dict(project_id=pid,config=dict(bin_columns=4)))
        prepare_export(root,read(store.path(pid)/'dataset.json'),[doc],ClassSchema('3d','cuboid',('car','pedestrian')),4)
        write(root/'result.json',dict(purpose='final_delivery',version='M1',scope='non_holdout',frame_ids=['001'],
            class_names=['car','pedestrian'],frames=[dict(frame_id='001',file='frames/0.json')],
            export_manifest_sha256=digest(root/'exports/shape/export.json')))
        write(root/'state.json',dict(status='completed',result_sha256=digest(root/'result.json')))
        weight=self.root/'round_fixture.pth';weight.write_bytes(b'fixture only')
        for index in (1,2):
            labels=dict(target_gold_read=False,frames=[dict(frame_id='001',boxes=[car])])
            metrics=dict(domain='target_unlabelled_proxy',target_gold_read=False,agreement=.8,supported_fraction=.9,box_count=1)
            store.record_round(pid,index,labels,metrics,{'A':weight,'B':weight},{'001':dict(boxes=[car])})
        store.select(pid,'M1')
        before=(root/'frames/0.json').read_bytes();export_before=(root/'exports/shape/annotations.json').read_bytes()
        with patch('annoloop.final_delivery.start') as start,patch('annoloop.final_delivery.upload') as upload:
            self.app.radio(key='navigation').set_value('So sánh phiên bản').run()
            self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
            key='compare_final_'+pid+'_fixture'
            self.app.multiselect(key='result_classes_'+key).set_value(['pedestrian']).run()
            self.app.radio(key='result_policy_'+key).set_value('Chỉ ignore').run()
            self.assertTrue(any(s.label=='Trước' for s in self.app.selectbox))
            self.assertTrue(any(s.label=='Sau' for s in self.app.selectbox))
            self.assertTrue(any(b.label=='Dùng phiên bản này' for b in self.app.button))
            chart=json.loads(self.app.get('plotly_chart')[0].proto.spec)
            self.assertTrue(any(t.get('name')=='pedestrian · ignore' for t in chart['data']))
            self.assertFalse(any(t.get('name','').startswith('car ·') for t in chart['data']))
            self.assertEqual(store.get(pid)['selected_version'],'M1')
            self.app.run();start.assert_not_called();upload.assert_not_called()
        self.assertEqual(before,(root/'frames/0.json').read_bytes())
        self.assertEqual(export_before,(root/'exports/shape/annotations.json').read_bytes())

    def test_folders_and_labels_create_project_without_manual_json_yaml(self):
        p=self.create()
        self.assertEqual(p['schema']['classes'],['car','pedestrian'])
        self.assertFalse((self.target/'class_schema.json').exists())
        self.assertEqual(self.app.radio(key='navigation').value,'Huấn luyện nguồn')
        self.assertFalse(any(b.label=='Học các class còn thiếu' for b in self.app.button))
        self.assertTrue(any('Không cần train nguồn lại' in message.value for message in self.app.success))
        for page in ['Class & nguồn','Vòng adaptation','So sánh phiên bản','Xuất CVAT']:
            self.app.radio(key='navigation').set_value(page).run()
            self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
        self.assertTrue(next(b for b in self.app.button if b.label=='Tạo task CVAT').disabled)

    def test_invalid_folder_shows_recoverable_error(self):
        self.app.run().radio(key='navigation').set_value('Dữ liệu').run()
        self.app.text_input(key='quick_source').set_value(str(self.root/'missing'))
        self.app.text_input(key='quick_target').set_value(str(self.target)).run()
        next(b for b in self.app.button if b.label=='Đọc hai thư mục').click().run()
        self.assertTrue(self.app.error);self.assertFalse(self.app.exception)
        self.assertNotIn('quick_data',self.app.session_state)

    def test_new_project_clears_old_draft_and_preserves_existing_project(self):
        previous=self.create();store=ProjectStore(self.root/'state')
        root=store.path(previous['id'])
        before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
        next(b for b in self.app.button if b.label=='＋ Tạo dự án mới').click().run()
        self.assertFalse(self.app.exception,str(self.app.exception))
        self.assertIsNone(self.app.selectbox(key='project_id').value)
        self.assertEqual(self.app.radio(key='navigation').value,'Dữ liệu')
        self.assertEqual(self.app.text_input(key='quick_source').value,'')
        self.assertEqual(self.app.text_input(key='quick_target').value,'')
        self.assertNotIn('quick_data',self.app.session_state)
        self.assertTrue(self.app.button(key='workflow_nav_1').disabled)
        self.app.run()  # A rerun must not silently reselect the old project.
        self.assertIsNone(self.app.selectbox(key='project_id').value)
        self.app.text_input(key='quick_name').set_value('Dự án thứ hai')
        self.app.text_input(key='quick_source').set_value(str(self.source.parent))
        self.app.text_input(key='quick_target').set_value(str(self.target)).run()
        next(b for b in self.app.button if b.label=='Đọc hai thư mục').click().run()
        self.app.multiselect[0].set_value(['car','pedestrian']).run()
        self.assertIsNone(self.app.selectbox(key='quick_model_A').value)
        self.assertIsNone(self.app.selectbox(key='quick_model_B').value)
        # Dataset creation must not validate a checkpoint until one is chosen.
        with patch('annoloop.setup.checkpoint_metadata',side_effect=AssertionError('unselected checkpoint')):
            next(b for b in self.app.button if b.label=='Lưu dữ liệu & chọn model →').click().run()
        self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
        current=self.app.selectbox(key='project_id').value
        self.assertNotEqual(current,previous['id'])
        self.assertEqual(store.get(current)['name'],'Dự án thứ hai')
        self.assertEqual(len(store.list()),2)
        self.assertEqual(self.app.radio(key='navigation').value,'Class & nguồn')
        self.assertEqual(before,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
        self.assertFalse((self.root/'state/jobs').exists())

    def test_draft_can_return_to_existing_project_and_blank_name_is_blocked(self):
        previous=self.create()
        next(b for b in self.app.button if b.label=='＋ Tạo dự án mới').click().run()
        self.app.selectbox(key='project_id').set_value(previous['id']).run()
        self.assertEqual(self.app.radio(key='navigation').value,'Tổng quan')
        self.assertFalse(self.app.exception,str(self.app.exception))
        next(b for b in self.app.button if b.label=='＋ Tạo dự án mới').click().run()
        self.start_wizard()
        self.app.text_input(key='quick_name').set_value('   ').run()
        self.assertTrue(next(b for b in self.app.button if b.label=='Lưu dữ liệu & chọn model →').disabled)
        self.assertEqual(len(ProjectStore(self.root/'state').list()),1)

    def test_single_workflow_select_before_download_and_never_select_again_in_adaptation(self):
        from test_model_hub import model
        m=dict(model(),framework='mmdetection3d',architecture='CenterPoint',modality='point_cloud')
        with patch('project_models.load_catalog',return_value={'models':[m]}),patch('annoloop.hub_selection.load_catalog',return_value={'models':[m]}),patch('annoloop.model_hub.load_catalog',return_value={'models':[m]}),patch('model_library.start_download') as download:
            self.start_wizard()
            next(b for b in self.app.button if b.label=='Lưu dữ liệu & chọn model →').click().run()
            self.assertEqual(self.app.radio(key='navigation').value,'Class & nguồn')
            pid=ProjectStore(self.root/'state').list()[0]['id']
            self.app.checkbox(key=f'workflow_compatible_{pid}').uncheck().run()
            for key in ['A','B']:self.app.selectbox(key=f'workflow_model_{pid}_{key}').set_value('hub:'+m['id'])
            self.app.run()
            next(b for b in self.app.button if b.label=='Xác nhận model & tiếp tục →').click().run()
            self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
            self.assertEqual(self.app.radio(key='navigation').value,'Huấn luyện nguồn')
            root=self.root/'state/projects'/pid;config=read(root/'adaptation.json')
            self.assertEqual(config['model_selection'],{k:dict(kind='hub',id=m['id']) for k in ['A','B']})
            self.assertEqual(config['models'],{})
            self.app.run();download.assert_not_called()
            self.app.button(key='workflow_nav_3').click().run()
            self.assertFalse(self.app.exception,str(self.app.exception))
            self.assertFalse(any('workflow_model_' in (s.key or '') or 'adapt_model_' in (s.key or '') for s in self.app.selectbox))
            self.assertTrue(any('CenterPoint' in s.value for s in self.app.markdown))
            self.assertTrue(next(b for b in self.app.button if b.label=='Chạy vòng tiếp theo').disabled)
            self.assertEqual(config,read(root/'adaptation.json'));download.assert_not_called()

    def test_model_filter_excludes_hub_before_download_and_keeps_old_selection_for_repair(self):
        p=self.create();pid=p['id'];store=ProjectStore(self.root/'state')
        from test_model_hub import model
        m=dict(model(),framework='mmdetection3d',architecture='CenterPoint',modality='point_cloud')
        native={}
        for key,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]:
            cfg=self.root/(key+'_filter.yaml')
            cfg.write_text('DATA_CONFIG:\n  POINT_FEATURE_ENCODING:\n    used_feature_list: [x, y, z]\n')
            native[key]=[dict(path=read(self.models[key])['checkpoint'],config=str(cfg),runtime='fixture',
                classes=['car','pedestrian'],architecture=arch,name=arch,intermediate=False)]
        # Header scan only; never inspect annotations, import a detector or download.
        with patch('project_models.load_catalog',return_value={'models':[m]}), \
             patch('annoloop.adaptation.model_choices',return_value=native), \
             patch('project_models.dataset_profile',wraps=dataset_profile) as scan, \
             patch('model_library.start_download') as download:
            # Keep the immutable media inventory intact; mock just the header reader.
            with patch('annoloop.model_filter.point_features',return_value=['x','y','z','rgb']):
                self.app.radio(key='navigation').set_value('Class & nguồn').run()
                self.assertFalse(self.app.exception,str(self.app.exception))
                self.assertTrue(self.app.checkbox(key=f'workflow_compatible_{pid}').value)
                widget=self.app.selectbox(key=f'workflow_model_{pid}_A')
                self.assertTrue(any('SECONDNetIoU' in option for option in widget.options))
                self.assertFalse(any('CenterPoint' in option for option in widget.options))
                table=next(d.value for d in self.app.dataframe if 'Lý do chưa phù hợp' in d.value.columns)
                self.assertTrue(any('adapter adaptation' in reason for reason in table['Lý do chưa phù hợp']))
                # Legacy saved unsupported choices remain visible, never silently replaced.
                write(store.path(pid)/'adaptation.json',dict(model_selection={k:dict(kind='hub',id=m['id']) for k in ['A','B']}))
                self.app.radio(key='navigation').set_value('Tổng quan').run()
                self.app.radio(key='navigation').set_value('Class & nguồn').run()
                self.assertEqual(self.app.selectbox(key=f'workflow_model_{pid}_A').value,'hub:'+m['id'])
                self.assertTrue(next(b for b in self.app.button if b.label=='Xác nhận model & tiếp tục →').disabled)
                self.app.radio(key='navigation').set_value('Huấn luyện nguồn').run()
                self.assertFalse(any(b.label=='Tải & chuẩn bị model' for b in self.app.button))
                self.assertTrue(next(b for b in self.app.button if b.label=='Cập nhật model đã chuẩn bị & kiểm tra bước tiếp →').disabled)
                download.assert_not_called();self.assertTrue(scan.called)

    def test_native_st3d_selection_prepares_only_after_explicit_download(self):
        from annoloop.native_sources import catalog
        p=self.create();pid=p['id']
        with patch('annoloop.model_filter.point_features',return_value=['x','y','z']), \
             patch('native_sources_ui.start_download') as download:
            self.app.radio(key='navigation').set_value('Class & nguồn').run()
            models=catalog()['models']
            for key,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]:
                model=next(m for m in models if m['architecture']==arch)
                self.app.selectbox(key=f'workflow_model_{pid}_{key}').set_value('source:'+model['id'])
            self.app.run()
            next(b for b in self.app.button if b.label=='Xác nhận model & tiếp tục →').click().run()
            self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
            saved=read(self.root/'state/projects'/pid/'adaptation.json')
            self.assertEqual(saved['model_selection']['A']['source_id'],models[0]['id'])
            self.assertEqual(saved['models'],{})
            self.app.run();download.assert_not_called()
            self.assertFalse((self.root/'state/native_model_cache').exists())
            self.app.button(key='download_native_'+models[0]['id']).click().run()
            self.assertEqual(download.call_count,1)
            self.app.run();self.assertEqual(download.call_count,1)

    def test_missing_model_submit_once_and_refresh_never_restarts(self):
        model=read(self.models['B']);model['classes']=['Vehicle'];write(self.models['B'],model)
        p=self.create()
        with patch('annoloop.jobs.start_pretrain',return_value='mock / no real process') as start:
            next(b for b in self.app.button if b.label=='Học các class còn thiếu').click().run()
            self.assertFalse(self.app.error,str(self.app.error))
            self.assertEqual(start.call_count,1)
            self.app.run()
            self.app.radio(key='navigation').set_value('Tổng quan').run()
            self.app.radio(key='navigation').set_value('Huấn luyện nguồn').run()
            self.assertEqual(start.call_count,1)
        self.assertFalse((self.root/'state/jobs').exists())

    def test_review_snapshots_and_rollback_keeps_both(self):
        p=self.create();store=ProjectStore(self.root/'state');pid=p['id']
        weight=self.root/'fixture.pth';weight.write_bytes(b'test only')
        for index in (1,2):
            box=dict(id='car1',class_name='car',box=[float(index),0.,0.,4.,2.,2.,0.],score=.9)
            labels=dict(target_gold_read=False,frames=[dict(frame_id='001',boxes=[box])])
            metrics=dict(domain='target_unlabelled_proxy',target_gold_read=False,agreement=.8,supported_fraction=.9,box_count=1)
            store.record_round(pid,index,labels,metrics,{'A':weight,'B':weight},{'001':dict(boxes=[box])})
        self.app.radio(key='navigation').set_value('So sánh phiên bản').run()
        self.assertFalse(self.app.error,str(self.app.error));self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.get('plotly_chart')),2)
        next(w for w in self.app.selectbox if w.label=='Phiên bản nhãn muốn sử dụng').set_value('M1')
        next(b for b in self.app.button if b.label=='Dùng phiên bản này').click().run()
        self.assertEqual(store.get(pid)['selected_version'],'M1')
        self.assertEqual(store.get(pid)['completed_rounds'],['M1','M2'])

    def test_rejected_first_round_explains_threshold_and_manual_override(self):
        p=self.create();store=ProjectStore(self.root/'state');pid=p['id']
        store.configure(pid,p['name'],3,'automatic',SelectionPolicy(mode='automatic'))
        weight=self.root/'fixture.pth';weight.write_bytes(b'test only')
        box=dict(id='car1',class_name='car',box=[4.,0.,0.,4.,2.,2.,0.],score=.9)
        labels=dict(target_gold_read=False,frames=[dict(frame_id='001',boxes=[box])])
        metrics=dict(domain='target_unlabelled_proxy',target_gold_read=False,
                     agreement=.005250965250965251,supported_fraction=1.,box_count=1)
        store.record_round(pid,1,labels,metrics,{'A':weight,'B':weight},{'001':dict(boxes=[box])})
        with patch('annoloop.adaptation.start_job') as dispatch:
            self.app.radio(key='navigation').set_value('So sánh phiên bản').run()
            self.assertFalse(self.app.exception,str(self.app.exception))
            self.assertTrue(any('M1 không được tự động chọn' in w.value for w in self.app.warning))
            self.assertTrue(any('Chưa có phiên bản nhãn được chấp nhận' in w.value for w in self.app.info))
            table=next(w.value for w in self.app.dataframe if 'Tiêu chí' in w.value.columns)
            self.assertEqual(table.iloc[0]['Kết quả'],'0.53%')
            self.assertEqual(table.iloc[0]['Ngưỡng'],'≥ 60.00%')
            receipt=read(store.path(pid)/'rounds/M1/round.json')
            self.assertEqual(receipt['selection_policy']['mode'],'automatic')
            next(b for b in self.app.button if b.label=='Dùng phiên bản này').click().run()
            self.assertTrue(any('Bạn đã chọn M1 bằng tay' in w.value for w in self.app.info))
            dispatch.assert_not_called()
        self.assertEqual(store.get(pid)['completed_rounds'],['M1'])
        self.assertEqual(store.get(pid)['selected_version'],'M1')

    def test_memory_history_is_visible_without_starting_a_model(self):
        p=self.create();store=ProjectStore(self.root/'state');pid=p['id']
        policy=MemoryPolicy()
        memory=empty_memory(store.get(pid)['dataset_sha256'],p['schema']['classes'],policy)
        box=dict(id='original',class_name='car',box=[4.,0.,0.,4.,2.,2.,0.],score=.8,
                 confidence=.8,quality=.8,training_policy='positive')
        original=dict(target_gold_read=False,frames=[dict(frame_id='001',boxes=[box])])
        raw={k:{'001':[box]} for k in ('A','B')}
        for index in (0,1):
            memory,labels=update_memory(memory,original,raw,memory_context(index,{'A':'A'+str(index),'B':'B'+str(index)}),
                lambda f:np.asarray([[4.,0.,0.]]*20,np.float32),policy)
        weight=self.root/'memory_fixture.pth';weight.write_bytes(b'fixture only')
        metrics=dict(domain='target_unlabelled_proxy',target_gold_read=False,
                     agreement=.9,supported_fraction=1.,box_count=1,memory=memory['summary'])
        store.record_round(pid,1,labels,metrics,{'A':weight,'B':weight},
            {'001':dict(boxes=labels['frames'][0]['boxes'])},memory=memory)
        with patch('annoloop.adaptation.start_job') as dispatch:
            self.app.radio(key='navigation').set_value('So sánh phiên bản').run()
            self.assertFalse(self.app.exception,str(self.app.exception))
            self.assertFalse(self.app.error,str(self.app.error))
            self.assertEqual(self.app.selectbox(key='memory_version_'+pid).value,'M1')
            table=next(w.value for w in self.app.dataframe if 'IoU lịch sử' in w.value.columns)
            self.assertEqual(list(table['Vòng']),['M0','M1'])
            self.assertEqual(list(table['Quality']),[.8,.8])
            dispatch.assert_not_called()

    def test_partial_post_train_predictions_visible_without_snapshot_or_new_gpu_job(self):
        (self.target/'001.bin').unlink()
        for fid in ('one','two'):
            np.save(self.target/(fid+'.npy'),np.ones((20,3),dtype=np.float32))
        write(self.target/'class_schema.json',dict(data_type='3d',annotation_type='cuboid',classes=['car']))
        store=ProjectStore(self.root/'state');pid=store.create('preview fixture',self.target)
        frames=read(store.path(pid)/'dataset.json')['frames']
        job=store.path(pid)/'adaptation_jobs/preview';work=store.path(pid)/'work/M1/attempts/preview'
        models={k:dict(path=k+'.pth',checkpoint_sha256='before_'+k,config_sha256='cfg_'+k,
                       runtime='fixture',classes=['car']) for k in ('A','B')}
        write(job/'spec.json',dict(mode='run',attempt_id='preview',dataset_sha256=store.get(pid)['dataset_sha256'],
            config=dict(models=models,bin_columns=4,adaptation_classes=['car'])))
        write(job/'state.json',dict(status='interrupted',mode='run',stage='predict_target',active_detector='B',
                                   work_directory=str(work)))
        write(work/'before/checkpoints.json',dict(checkpoint_hashes={k:'before_'+k for k in models}))
        for key in models:
            write(work/(key+'_trained.json'),dict(checkpoint_sha256='after_'+key,epochs=3))
            for phase in ('before','after'):
                root=job/key/'target'/phase
                write(root/'spec.json',dict(models[key],mode='target_sample',frames=frames,checkpoint_sha256=phase+'_'+key))
                count=1 if phase=='after' and key=='B' else 2
                for i,frame in enumerate(frames[:count]):
                    write(root/'frames'/f'{i:06d}.json',dict(frame_id=frame['frame_id'],boxes=[
                        dict(id='box',class_name='car',box=[1.,1.,1.,4.,2.,2.,0.],score=.8)]))
                if count==2:
                    write(root/'result.json',dict(domain='target_unlabelled',checkpoint_sha256=phase+'_'+key))
        original=digest(store.path(pid)/'project.json')
        self.app.session_state['project_id']=pid;self.app.session_state['navigation']='So sánh phiên bản'
        with patch('annoloop.adaptation.start_job') as train,patch('annoloop.inspection.start_inspection') as predict:
            self.app.run()
            self.assertFalse(self.app.error,str(self.app.error));self.assertFalse(self.app.exception)
            self.assertEqual(len(self.app.get('plotly_chart')),4)
            table=next(df.value for df in self.app.dataframe if 'Frame đã predict' in df.value.columns)
            self.assertIn('1/2',table['Frame đã predict'].tolist())
            self.app.selectbox(key='preview_frame_'+pid).set_value('two').run()
            self.assertEqual(len(self.app.get('plotly_chart')),3)
            self.assertTrue(any('chưa có prediction' in message.value for message in self.app.info))
            train.assert_not_called();predict.assert_not_called()
        self.assertEqual(digest(store.path(pid)/'project.json'),original)
        self.assertEqual(store.get(pid)['completed_rounds'],[])

    def test_prediction_view_refreshes_page_when_another_round_completes(self):
        self.app.run()
        import prediction_page
        from types import SimpleNamespace
        stale=dict(id='fixture',completed_rounds=[],selected_version=None)
        store=SimpleNamespace(get=lambda _:dict(stale,completed_rounds=['M1']))
        with patch.object(prediction_page.st,'rerun',side_effect=RuntimeError('refresh completed round')) as rerun:
            with self.assertRaisesRegex(RuntimeError,'refresh completed round'):
                prediction_page.saved_predictions.__wrapped__(store,stale)
            rerun.assert_called_once_with(scope='app')

    def test_folder_picker_exists_for_source_and_target(self):
        self.app.run().radio(key='navigation').set_value('Dữ liệu').run()
        self.assertEqual(len([b for b in self.app.button if b.label=='📂 Chọn']),2)
        # Browse only; neither reading labels nor training runs is triggered.
        self.app.session_state['browse_quick_target']=str(self.target)
        self.app.button(key='quick_target_browse').click().run()
        self.assertFalse(self.app.exception,str(self.app.exception))
        self.assertTrue(any(b.label=='Dùng thư mục này' for b in self.app.button))
        next(b for b in self.app.button if b.label=='Dùng thư mục này').click().run()
        self.assertFalse(self.app.exception,str(self.app.exception))
        self.assertEqual(self.app.text_input(key='quick_target').value,str(self.target))
        self.assertNotIn('quick_data',self.app.session_state)

    def test_failed_launch_is_explicitly_historical_and_does_not_auto_retry(self):
        model=read(self.models['B']);model['classes']=['Vehicle'];write(self.models['B'],model)
        p=self.create();store=ProjectStore(self.root/'state')
        path=store.path(p['id'])/'setup.json';setup=read(path)
        setup['launches']=[dict(status='failed',error='Detector B: failed',path='old-run')]
        write(path,setup)
        with patch('annoloop.jobs.start_pretrain') as start:
            self.app.run()
            self.assertTrue(any('Chưa có lượt thử lại mới' in w.value for w in self.app.warning))
            self.assertTrue(any(b.label=='Thử lại với môi trường đã sửa' for b in self.app.button))
            start.assert_not_called()

    def test_intermediate_checkpoint_page_can_evaluate_without_training_or_target_confirmation(self):
        for path in self.models.values():
            value=read(path);value['classes']=['Vehicle'];write(path,value)
        p=self.create();store=ProjectStore(self.root/'state');setup=read(store.path(p['id'])/'setup.json')
        job=self.root/'state/jobs/checkpoint-fixture';job.mkdir(parents=True)
        write(job/'job.json',dict(spec=setup['specs']['A']))
        weight=job/'checkpoints/checkpoint_epoch_10.pth';weight.parent.mkdir();weight.write_bytes(b'fixture only')
        self.app.radio(key='navigation').set_value('Kiểm tra model').run()
        self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
        self.assertFalse(next(b for b in self.app.button if b.label=='Đánh giá source validation').disabled)
        self.assertTrue(next(b for b in self.app.button if b.label=='Predict sample target').disabled)
        with patch('inspection_page.start_inspection',return_value='mock / no process') as inspect,patch('annoloop.jobs.start_pretrain') as train:
            next(b for b in self.app.button if b.label=='Đánh giá source validation').click().run()
            inspect.assert_called_once();train.assert_not_called()

    def test_current_project_is_selected_and_settings_can_be_saved(self):
        p=self.create()
        self.assertEqual(self.app.selectbox(key='project_id').value,p['id'])
        self.assertNotIn(None,self.app.selectbox(key='project_id').options)
        next(w for w in self.app.sidebar.text_input if w.label=='Tên dự án').set_value('Project renamed')
        next(w for w in self.app.sidebar.number_input if w.label=='Số vòng tối đa').set_value(5)
        next(b for b in self.app.button if b.label=='Lưu dự án').click().run()
        saved=ProjectStore(self.root/'state').get(p['id'])
        self.assertEqual(saved['name'],'Project renamed');self.assertEqual(saved['max_rounds'],5)
        self.assertFalse(self.app.exception,str(self.app.exception))

    def test_direct_streamlit_launch_watches_backend_files(self):
        from streamlit.file_util import file_in_pythonpath
        with patch.dict(os.environ, {'PYTHONPATH': ''}):
            self.app.run()
            self.assertTrue(file_in_pythonpath(str(ROOT/'src/annoloop/projects.py')))
            self.assertFalse(self.app.exception,str(self.app.exception))

    def test_training_monitor_uses_architecture_not_filename_or_tqdm_cursor_a(self):
        self.create()
        job=self.root/'state/jobs/monitor-fixture'
        # File name deliberately disagrees with the actual model architecture.
        config=self.root/'A_source.yaml';config.write_text('MODEL:\n  NAME: PVRCNN\n')
        write(job/'job.json',dict(status='planned',spec=dict(config=str(config))))
        (job/'train.log').write_text('train: 1/323, total_it=1\x1b[A\nepochs: 0/20, loss=8.65\n')
        self.app.radio(key='navigation').set_value('Huấn luyện nguồn').run()
        self.assertFalse(self.app.exception,str(self.app.exception))
        self.assertTrue(any(m.value=='Model: B · PVRCNN' for m in self.app.subheader))
        self.assertTrue(any('B · PVRCNN' in option for w in self.app.selectbox if w.label=='Run đã đăng ký' for option in w.options))
        self.assertFalse(any('\x1b[A' in code.value for code in self.app.code))

    def test_adaptation_explains_missing_model_and_never_auto_launches(self):
        p=self.create()
        with patch('annoloop.adaptation.start_job') as start:
            self.app.radio(key='navigation').set_value('Vòng adaptation').run()
            self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
            self.assertTrue(next(b for b in self.app.button if b.label=='Chạy vòng tiếp theo').disabled)
            self.assertTrue(next(b for b in self.app.button if b.label=='Kiểm tra adapter GPU A → B').disabled)
            self.assertFalse(any('concrete DetectorAdapter' in m.value for m in self.app.warning))
            self.app.run();start.assert_not_called()

    def test_downloaded_hub_model_selectable_and_saved_with_training_blocker(self):
        from annoloop.model_hub import download,model_root
        from test_model_hub import model,Response,CONFIG
        p=self.create();pid=p['id'];m=dict(model(['Car','Pedestrian']),framework='mmdetection3d',architecture='CenterPoint',modality='point_cloud')
        def opener(request,timeout):return Response(CONFIG,url=request.full_url) if 'raw.githubusercontent' in request.full_url else Response()
        download(self.root/'state',m,opener=opener)
        weight=str(model_root(self.root/'state',m['id'])/'weights.pth')
        with patch('annoloop.hub_selection.load_catalog',return_value={'models':[m]}),patch('project_models.load_catalog',return_value={'models':[m]}),patch('annoloop.model_hub.load_catalog',return_value={'models':[m]}),patch('annoloop.adaptation.start_job') as start:
            self.app.radio(key='navigation').set_value('Class & nguồn').run()
            self.app.checkbox(key=f'workflow_compatible_{pid}').uncheck().run()
            widget=self.app.selectbox(key=f'workflow_model_{pid}_A')
            self.assertTrue(any('CenterPoint' in option for option in widget.options))
            widget.set_value('hub:'+m['id'])
            self.app.selectbox(key=f'workflow_model_{pid}_B').set_value('hub:'+m['id']).run()
            next(b for b in self.app.button if b.label=='Xác nhận model & tiếp tục →').click().run()
            self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
            saved=read(self.root/'state/projects'/pid/'adaptation.json')
            self.assertEqual(saved['models']['A']['path'],weight)
            self.assertEqual(self.app.radio(key='navigation').value,'Huấn luyện nguồn')
            self.app.radio(key='navigation').set_value('Vòng adaptation').run()
            self.assertTrue(next(b for b in self.app.button if b.label=='Kiểm tra adapter GPU A → B').disabled)
            self.assertTrue(next(b for b in self.app.button if b.label=='Chạy vòng tiếp theo').disabled)
            self.assertTrue(any('chưa nối train target' in w.value for w in self.app.warning))
            self.assertFalse(any(s.key==f'adapt_model_{pid}_A' for s in self.app.selectbox))
            start.assert_not_called()

    def test_adaptation_buttons_dispatch_only_after_saved_checks_and_explicit_click(self):
        p=self.create();store=ProjectStore(self.root/'state');pid=p['id']
        candidates={}
        for key,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]:
            model=read(self.models[key]);config=self.root/(key+'.yaml');config.write_text('UI fixture')
            candidates[key]=[dict(path=model['checkpoint'],config=str(config),runtime='fixture-runtime',
                classes=['car','pedestrian'],architecture=arch,name=key,intermediate=False)]
        with patch('annoloop.adaptation.model_choices',return_value=candidates):
            from annoloop import adaptation
            adaptation.save_setup(store,pid,Settings.load(),{k:candidates[k][0]['path'] for k in ('A','B')},False)
            self.app.radio(key='navigation').set_value('Vòng adaptation').run()
            self.app.checkbox(key=f'adapt_axes_{pid}').set_value(True).run()
            next(b for b in self.app.button if b.label=='Lưu thiết lập adaptation').click().run()
            self.assertFalse(self.app.error,str(self.app.error))
            self.assertFalse(next(b for b in self.app.button if b.label=='Kiểm tra adapter GPU A → B').disabled)
            self.assertTrue(next(b for b in self.app.button if b.label=='Chạy vòng tiếp theo').disabled)
            with patch('annoloop.adaptation.start_job',return_value='fixture no process') as start:
                next(b for b in self.app.button if b.label=='Kiểm tra adapter GPU A → B').click().run()
                self.app.run();self.assertEqual(start.call_count,1)
                self.assertEqual(start.call_args.kwargs['mode'],'verify')
            config=read(store.path(pid)/'adaptation.json')
            write(store.path(pid)/'native_verified.json',dict(models={k:{f:m[f] for f in ('checkpoint_sha256','config_sha256','runtime')}
                for k,m in config['models'].items()},optimizer_steps=0,source_evidence=read(store.path(pid)/'setup.json')['source_evidence'],
                point_contract=dict(dataset_sha256=store.get(pid)['dataset_sha256'],bin_columns=config['bin_columns'],coordinates_confirmed=True)))
            self.app.run()
            self.assertFalse(next(b for b in self.app.button if b.label=='Chạy vòng tiếp theo').disabled)
            with patch('annoloop.adaptation.start_job',return_value='fixture no process') as start:
                next(b for b in self.app.button if b.label=='Chạy vòng tiếp theo').click().run()
                self.app.run();self.assertEqual(start.call_count,1)
            self.app.number_input(key=f'adapt_epochs_{pid}').set_value(2).run()
            self.assertTrue(next(b for b in self.app.button if b.label=='Chạy vòng tiếp theo').disabled)
            self.app.number_input(key=f'adapt_epochs_{pid}').set_value(1).run()
            old=store.path(pid)/'work/M1/state.json'
            write(old,dict(status='running',interrupted_fixture=True))
            original=old.read_bytes()
            store.request_finish(pid)
            self.app.run()
            self.assertTrue(next(b for b in self.app.button if b.label=='Chạy vòng tiếp theo').disabled)
            with patch('annoloop.adaptation.start_job') as start:
                next(b for b in self.app.button if b.label=='Cho phép chạy tiếp').click().run()
                start.assert_not_called()
            self.assertFalse(store.get(pid)['stop_after_round'])
            self.app.run()
            self.assertFalse(next(b for b in self.app.button if b.label=='Chạy lại vòng M1').disabled)
            with patch('annoloop.adaptation.start_job',return_value='fixture no process') as start:
                next(b for b in self.app.button if b.label=='Chạy lại vòng M1').click().run()
                self.app.run();self.assertEqual(start.call_count,1)
                self.assertTrue(start.call_args.kwargs['retry'])
            self.assertEqual(old.read_bytes(),original)
            self.assertFalse(self.app.exception,str(self.app.exception))
            report=dict(domain='source_validation',minimum_samples=10,ready=False,classes=[
                dict(class_name='car',samples=20,matched_at_iou_0_3=12,enough_samples=True),
                dict(class_name='pedestrian',samples=5,matched_at_iou_0_3=0,enough_samples=False)])
            def current_scope(*args):
                chosen=read(store.path(pid)/'adaptation.json')['adaptation_classes']
                rows=[r for r in report['classes'] if r['class_name'] in chosen]
                return {'A':dict(report,classes=rows,ready=all(r['enough_samples'] for r in rows))}
            with patch('annoloop.adaptation.calibration_reports',side_effect=current_scope), \
                 patch('annoloop.adaptation.start_job') as start:
                self.app.run()
                self.assertTrue(next(b for b in self.app.button if b.label=='Chạy lại vòng M1').disabled)
                self.assertFalse(next(b for b in self.app.button if b.label=='Kiểm tra adapter GPU A → B').disabled)
                table=next(df.value for df in self.app.dataframe if 'Số box dự đoán' in df.value.columns)
                self.assertEqual(table['Số box dự đoán'].tolist(),[20,5])
                self.assertTrue(any('5/10' in text.value for text in self.app.markdown))
                self.app.multiselect(key=f'adapt_classes_{pid}').set_value(['car']).run()
                self.assertTrue(next(b for b in self.app.button if b.label=='Chạy lại vòng M1').disabled)
                next(b for b in self.app.button if b.label=='Lưu thiết lập adaptation').click().run()
                self.assertEqual(read(store.path(pid)/'adaptation.json')['adaptation_classes'],['car'])
                self.assertFalse(next(b for b in self.app.button if b.label=='Chạy lại vòng M1').disabled)
                start.assert_not_called()
            self.assertFalse(self.app.exception,str(self.app.exception))
            recovery=dict(work=str(old.parent),completed={'A':{},'B':{}},checkpoints={})
            with patch('adaptation_page.resume_options',return_value=[recovery]), \
                 patch('annoloop.adaptation.start_job',return_value='fixture no process') as start:
                self.app.run()
                self.assertFalse(next(b for b in self.app.button if b.label=='Tiếp tục vòng M1').disabled)
                next(b for b in self.app.button if b.label=='Tiếp tục vòng M1').click().run()
                self.app.run()
                start.assert_called_once()
                self.assertEqual(start.call_args.kwargs['resume_from'],str(old.parent))
                self.assertNotIn('retry',start.call_args.kwargs)

    def test_finished_background_check_refreshes_controls_without_dispatch(self):
        self.app.run()  # Loads the same UI module used by Streamlit.
        import adaptation_page
        waiting_key='adaptation_waiting_fixture_project'
        session={waiting_key:'fixture_job'}
        with patch.object(adaptation_page.st,'session_state',session), \
             patch.object(adaptation_page.st,'rerun',side_effect=RuntimeError('full app refresh')) as rerun, \
             patch('annoloop.adaptation.job_status',return_value={'process_alive':False}), \
             patch('annoloop.adaptation.start_job') as start:
            with self.assertRaisesRegex(RuntimeError,'full app refresh'):
                adaptation_page.monitor.__wrapped__(Path('/fixture_project'))
            rerun.assert_called_once_with(scope='app')
            self.assertNotIn(waiting_key,session)
            start.assert_not_called()


if __name__=='__main__':unittest.main()
