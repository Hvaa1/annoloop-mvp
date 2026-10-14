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
from annoloop.pseudo_memory import empty as empty_memory, update as update_memory, MemoryPolicy
from annoloop.memory_store import context as memory_context

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
        next(b for b in self.app.button if b.label=='Tạo dự án và tự chuẩn bị').click().run()
        self.assertFalse(self.app.exception,str(self.app.exception));self.assertFalse(self.app.error,str(self.app.error))
        return ProjectStore(self.root/'state').list()[0]

    def test_empty_pages_and_no_accidental_jobs(self):
        self.app.run()
        for page in ['Tổng quan','Dữ liệu','Class & nguồn','Huấn luyện nguồn','Vòng adaptation','So sánh phiên bản','Xuất CVAT','Hướng dẫn & thiết bị']:
            self.app.radio(key='navigation').set_value(page).run()
            self.assertFalse(self.app.exception,str(self.app.exception))
        self.assertFalse((self.root/'state/jobs').exists())

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
        self.app.button(key='quick_target_browse').click().run()
        self.assertFalse(self.app.exception,str(self.app.exception))
        self.assertTrue(any(b.label=='Dùng thư mục này' for b in self.app.button))

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

    def test_adaptation_buttons_dispatch_only_after_saved_checks_and_explicit_click(self):
        p=self.create();store=ProjectStore(self.root/'state');pid=p['id']
        candidates={}
        for key,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]:
            model=read(self.models[key]);config=self.root/(key+'.yaml');config.write_text('UI fixture')
            candidates[key]=[dict(path=model['checkpoint'],config=str(config),runtime='fixture-runtime',
                classes=['car','pedestrian'],architecture=arch,name=key,intermediate=False)]
        with patch('annoloop.adaptation.model_choices',return_value=candidates):
            self.app.radio(key='navigation').set_value('Vòng adaptation').run()
            for key in ('A','B'):
                self.app.selectbox(key=f'adapt_model_{pid}_{key}').set_value(candidates[key][0]['path'])
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
