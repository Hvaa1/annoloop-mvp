"""MMDet recipes and loss masks on CPU; no model/network/GPU dispatch."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from annoloop.io import read,write,digest
from annoloop.mmdet_recipe import describe,current,input_blockers,config_tree
from annoloop.mmdet_ignore import install,inside_xy
from annoloop.mmdet_quality import consistency,rotate_points,ANGLE


class MMDetTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)

    def recipe(self,head='CenterHead',sweeps=False):
        path=self.root/'model.py'
        text="class_names=['car','pedestrian']\nvoxel_size=[.1,.1,.2]\nmodel=dict(type='CenterPoint',pts_bbox_head=dict(type='"+head+"',in_channels=sum([128,128]),voxel=voxel_size[:2]))\ntest_pipeline=[dict(type='LoadPointsFromFile',coord_type='LIDAR',load_dim=4,use_dim=4)]\n"
        if sweeps:text+="test_pipeline=[dict(type='LoadPointsFromFile',coord_type='LIDAR',load_dim=5,use_dim=5),dict(type='LoadPointsFromMultiSweeps',sweeps_num=9,use_dim=[0,1,2,3,4])]\n"
        path.write_text(text);return describe(path,self.root)

    def test_recipe_inheritance_no_python_execution_and_input_contract(self):
        recipe=self.recipe();self.assertTrue(recipe['implementation_supported']);self.assertTrue(recipe['input_supported'])
        self.assertEqual(recipe['features'],['x','y','z','intensity']);self.assertTrue(current(recipe))
        child=self.root/'child.py';child.write_text("_base_=['model.py']\nmodel=dict(pts_bbox_head=dict(in_channels=64))\n")
        cfg,_=config_tree(child,self.root);self.assertEqual(cfg['model']['pts_bbox_head']['type'],'CenterHead')
        self.assertEqual(cfg['model']['pts_bbox_head']['in_channels'],64)
        child.write_text("_base_=['model.py']\nimport os\nos.system('must-never-execute')\n")
        with patch('os.system',side_effect=AssertionError('Config execution')):describe(child,self.root)
        child.write_text("_base_=['../outside.py']\n")
        with self.assertRaisesRegex(ValueError,'ngoài cache'):describe(child,self.root)
        (self.root/'model.py').write_text('changed');self.assertFalse(current(recipe))

    def test_downloaded_unconnected_model_uses_pinned_config_only(self):
        from annoloop.mmdet_recipe import cached_model
        self.recipe()
        cache=self.root/'data/model_cache/mmdet3d-fixture';upstream=cache/'upstream'
        upstream.mkdir(parents=True);config=upstream/'model.py'
        config.write_text((self.root/'model.py').read_text())
        model=dict(id='mmdet3d-fixture',framework='mmdetection3d',weights_url='https://x/w.pth',config_path='model.py',
                   config_files={'model.py':dict(sha256=digest(config))})
        self.assertNotIn('mmdet_recipe',cached_model(self.root/'data',model))  # not downloaded
        write(cache/'installed.json',dict(model_id=model['id'],weights_url=model['weights_url'],sha256='w'))
        found=cached_model(self.root/'data',model)
        self.assertEqual(found['mmdet_recipe']['head'],'CenterHead')
        self.assertEqual(found['mmdet_recipe_source'],'static_config_unconnected')
        config.write_text(config.read_text()+'\n# edited\n')  # tampered config is ignored
        self.assertNotIn('mmdet_recipe',cached_model(self.root/'data',model))

    def test_worker_configures_nvrtc_headers_before_any_mode(self):
        from annoloop import mmdet_worker
        write(self.root/'spec.json',{})
        order=[]
        with patch('importlib.util.find_spec',return_value=object()), \
             patch('annoloop.runtime.configure_sparse_headers',side_effect=lambda:order.append('headers')), \
             patch.object(mmdet_worker,'verify',side_effect=lambda root,spec:order.append('verify')):
            mmdet_worker.run(self.root,'verify')
        self.assertEqual(order,['headers','verify'])

    def test_verify_target_backward_uses_bounded_top_scored_boxes(self):
        from annoloop import mmdet_worker
        class Value:
            def backward(self):pass
            def detach(self):return self
            def cpu(self):return self
            def __float__(self):return 1.
        masks=dict(masked=0);calls=[]
        def fake_loss(torch,model,head,masks_,array,truth,ignored):
            calls.append(truth);masks_['masked']=3;return Value()
        source_box=[1.,1.,0.,4.,2.,2.,0.]
        raw=[dict(box=[float(i),0,0,1,1,1,0],score=i/500,class_name='car') for i in range(400)]
        spec=dict(classes=['car'],frames=[dict(frame_id='t')],checkpoint_sha256='c',config_sha256='k',mmdet_recipe={})
        fake_model=SimpleNamespace(zero_grad=lambda set_to_none:None)
        with patch.object(mmdet_worker,'prepare',return_value=(None,fake_model,None,masks,None)), \
             patch.object(mmdet_worker,'sources',return_value=[dict(source_boxes=[source_box],source_names=['car'])]), \
             patch.object(mmdet_worker,'points',return_value=np.zeros((10,4),dtype=np.float32)), \
             patch('annoloop.reused.core.points_in_box',return_value=np.ones(10,bool)), \
             patch.object(mmdet_worker,'loss',side_effect=fake_loss), \
             patch.object(mmdet_worker,'predictions',return_value=raw), \
             patch('annoloop.native_training.checked_gradients',return_value=5):
            mmdet_worker.verify(self.root,spec)
        target=calls[1][0]
        self.assertEqual(len(target),mmdet_worker.VERIFY_TARGET_BOXES)
        self.assertEqual(target[0][0],399.)  # highest score first
        self.assertEqual(read(self.root/'verified.json')['target_verify_boxes'],mmdet_worker.VERIFY_TARGET_BOXES)

    def test_v0_checkpoints_are_blocked_for_legacy_box_convention(self):
        from annoloop.mmdet_recipe import legacy_blocker,with_legacy,LEGACY_CONVENTION
        from annoloop.model_filter import assess
        old=dict(weights_url='https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/x.pth')
        new=dict(weights_url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/x.pth')
        self.assertEqual(legacy_blocker(old),[LEGACY_CONVENTION]);self.assertEqual(legacy_blocker(new),[])
        recipe=with_legacy(old,self.recipe());self.assertFalse(recipe['input_supported'])
        profile=dict(complete=True,features=['x','y','z','intensity'],sweeps_available=False)
        report=assess(dict(old,classes=['car'],framework='mmdetection3d',mmdet_recipe=recipe),['car'],profile)
        self.assertFalse(report['selectable']);self.assertIn(LEGACY_CONVENTION,report['reasons'])
        self.assertTrue(assess(dict(new,classes=['car'],framework='mmdetection3d',mmdet_recipe=self.recipe()),['car'],profile)['eligible'])

    def test_chunked_overlaps_match_full_computation_exactly(self):
        from annoloop.mmdet_ignore import ChunkedOverlaps,chunk_assigners
        def inner(a,b,mode='iou',is_aligned=False):
            return (a[:,None,0]-b[None,:,0]).abs()+(1 if mode=='iof' else 0)
        gt=torch.rand(7,7);anchors=torch.rand(1000,7);chunked=ChunkedOverlaps(inner,chunk=64)
        self.assertTrue(torch.equal(chunked(gt,anchors),inner(gt,anchors)))
        self.assertTrue(torch.equal(chunked(anchors,gt,mode='iof'),inner(anchors,gt,'iof')))
        head=SimpleNamespace(bbox_assigner=[SimpleNamespace(iou_calculator=inner),SimpleNamespace(iou_calculator=inner)])
        chunk_assigners(head);chunk_assigners(head)
        self.assertTrue(all(isinstance(a.iou_calculator,ChunkedOverlaps) and a.iou_calculator.inner is inner for a in head.bbox_assigner))

    def test_multisweep_and_missing_intensity_are_not_silently_filled(self):
        recipe=self.recipe(sweeps=True);self.assertFalse(recipe['input_supported']);self.assertEqual(recipe['sweeps'],10)
        reasons=input_blockers(recipe,dict(complete=True,features=['x','y','z','intensity']))
        self.assertTrue(any('time' in r for r in reasons))
        recipe=self.recipe();self.assertTrue(input_blockers(recipe,dict(complete=True,features=['x','y','z','rgb'])))
        unknown=self.recipe('FreeAnchor3DHead');self.assertFalse(unknown['implementation_supported'])

    def test_source_nuscenes_contract_and_real_intensity_are_bound_to_infos(self):
        import pickle
        from annoloop.mmdet_worker import sources,points
        media=self.root/'cloud.bin';np.array([[1.,2.,3.,.75,12.]],dtype='<f4').tofile(media)
        path=self.root/'infos.pkl'
        row=dict(token='source',sweeps=[],lidar_path='cloud.bin',gt_names=np.array(['car']),gt_boxes=np.array([[1.,2.,3.,4.,2.,2.,0.]]))
        path.write_bytes(pickle.dumps([row]))
        spec=dict(bin_columns=4,source_setup=dict(loader_root=str(self.root),source_root=str(self.root),
            source_evidence=dict(train=dict(path=str(path),sha256=digest(path)))),mmdet_recipe=dict(features=['x','y','z','intensity']))
        rows=sources(spec,'train');array=points(spec,rows[0])
        self.assertEqual(array.shape,(1,4));self.assertEqual(array[0,-1],.75)
        self.assertEqual(rows[0]['feature_names'][-1],'ring_index')
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'infos đã đổi'):sources(spec,'train')
        row.pop('sweeps');path.write_bytes(pickle.dumps([row]));spec['source_setup']['source_evidence']['train']['sha256']=digest(path)
        with self.assertRaisesRegex(ValueError,'không đoán layout'):sources(spec,'train')

    def test_current_scan_time_is_derived_only_in_explicit_mode_and_preserves_measurements(self):
        from annoloop.mmdet_recipe import effective
        from annoloop.mmdet_worker import points
        recipe=self.recipe(sweeps=True)
        frame_path=self.root/'target.bin';raw=np.array([[1.,2.,3.,.75]],dtype='<f4');raw.tofile(frame_path)
        frame=dict(frame_id='target',media=str(frame_path),sha256=digest(frame_path),feature_names=['x','y','z','intensity'])
        spec=dict(bin_columns=4,mmdet_recipe=recipe)
        with self.assertRaisesRegex(ValueError,'thiếu feature'):points(spec,frame)
        recipe=effective(recipe,'current_scan_only');self.assertTrue(recipe['input_supported']);self.assertEqual(recipe['declared_sweeps'],10)
        self.assertFalse(input_blockers(recipe,dict(complete=True,features=frame['feature_names'])))
        spec['mmdet_recipe']=recipe;array=points(spec,frame)
        np.testing.assert_array_equal(array[:,:4],raw);self.assertEqual(array[0,4],0.)
        # Intensity can never be invented, even in current-scan mode.
        frame_path.write_bytes(np.ones((1,3),dtype='<f4').tobytes());frame.update(sha256=digest(frame_path),feature_names=['x','y','z']);spec['bin_columns']=3
        with self.assertRaisesRegex(ValueError,'thiếu feature'):points(spec,frame)
        array=np.array([[1.,2.,3.,.75,-.1]],dtype='<f4');array.tofile(frame_path)
        frame.update(sha256=digest(frame_path),feature_names=['x','y','z','intensity','time']);spec['bin_columns']=5
        with self.assertRaisesRegex(ValueError,'elapsed time khác0'):points(spec,frame)

    def test_rotation_quality_is_independent_measured_iou_not_confidence(self):
        first=[dict(id='one',class_name='car',box=[2.,0.,0.,4.,2.,2.,0.],score=.2)]
        second=[dict(first[0],box=[2*np.cos(ANGLE),2*np.sin(ANGLE),0.,4.,2.,2.,ANGLE])]
        result=consistency(first,second)[0]
        self.assertAlmostEqual(result['iou_score'],1.,places=6);self.assertEqual(result['cls_score'],.2)
        self.assertFalse(result['quality_is_native_iou_head'])
        self.assertEqual(consistency(first,[])[0]['iou_score'],0.)
        self.assertEqual(consistency(first,[dict(second[0],class_name='pedestrian')])[0]['iou_score'],0.)
        array=np.array([[1.,2.,3.,.7]],dtype=np.float32);np.testing.assert_allclose(rotate_points(array,ANGLE)[:,2:],array[:,2:])

    def test_source_iou_only_in_calibration_target_results_never_contain_gold(self):
        from annoloop.mmdet_worker import infer
        from annoloop.predictions import assert_no_gold
        box=dict(id='one',class_name='car',box=[0.,0.,0.,4.,2.,2.,0.],score=.8,cls_score=.8,iou_score=.9)
        source=dict(frame_id='source',source_boxes=[box['box']],source_names=['car'])
        spec=dict(mode='validation',classes=['car'],checkpoint_sha256='fixture',mmdet_recipe={})
        out=self.root/'source'
        with patch('annoloop.mmdet_worker.prepare',return_value=(None,None,None,None,None)), \
             patch('annoloop.mmdet_worker.points',return_value=np.ones((1,3))), \
             patch('annoloop.mmdet_worker.predictions',return_value=[box]), \
             patch('annoloop.mmdet_worker.sources',return_value=[source]):
            infer(out,spec)
            result=read(out/'result.json');self.assertAlmostEqual(result['calibration_records']['boxes'][0]['source_gt_iou'],1.)
            assert_no_gold(read(out/'frames/000000.json'))
            spec.update(mode='target_sample',frames=[dict(frame_id='target')])
            infer(self.root/'target',spec)
            result=read(self.root/'target/result.json');self.assertIsNone(result['calibration_records']);assert_no_gold(result)
            assert_no_gold(read(self.root/'target/frames/000000.json'))

    def test_anchor_ignore_zeroes_only_background_and_velocity_regression(self):
        def targets(assigner,anchors,gt,ignored,meta):
            return (torch.tensor([0,2,2]),torch.ones(3),torch.zeros((3,9)),
                    torch.tensor([[1.]*9,[0.]*9,[0.]*9]),torch.zeros(3),torch.ones(3),torch.tensor([0]),torch.tensor([1,2]))
        head=SimpleNamespace(anchor_target_single_assigner=targets,train_cfg={});context=install(head,'Anchor3DHead')
        context['boxes']=torch.tensor([[0.,0.,0.,2.,2.,2.,0.]])
        anchors=torch.zeros(3,9);anchors[2,0]=10.
        result=head.anchor_target_single_assigner(None,anchors,None,None,None)
        self.assertEqual(result[1].tolist(),[1.,0.,1.]);self.assertEqual(context['masked'],1)
        self.assertEqual(result[3][:,7:].sum(),0)

    def test_center_ignore_changes_negative_gradient_keeps_positive(self):
        class Loss:
            def forward(self,pred,target,weight=None,**kw):return ((pred-target).square()*(weight if weight is not None else 1)).sum()
        head=SimpleNamespace(loss_cls=Loss(),train_cfg=dict(point_cloud_range=[0,0,-5,4,4,5],voxel_size=[1,1,1],out_size_factor=1,code_weights=[1.]*10))
        context=install(head,'CenterHead');context['boxes']=torch.tensor([[.5,.5,0.,3.,3.,3.,0.]])
        pred=torch.full((1,1,4,4),.2,requires_grad=True);target=torch.zeros_like(pred);target[0,0,0,0]=1
        head.loss_cls.forward(pred,target).backward()
        self.assertEqual(pred.grad[0,0,1,1],0.);self.assertNotEqual(pred.grad[0,0,0,0],0.);self.assertNotEqual(pred.grad[0,0,3,3],0.)
        self.assertGreater(context['masked'],0);self.assertEqual(head.train_cfg['code_weights'][-2:],[0.,0.])

    def test_native_and_framework_are_routed_to_different_adapters(self):
        from annoloop.detector_factory import create
        args=([],[],self.root,SimpleNamespace(),4,{})
        with patch('annoloop.native_adapter.NativeDetectorAdapter') as native,patch('annoloop.mmdet_adapter.MMDetDetectorAdapter') as mm:
            create(dict(backend='native_st3d'),*args);native.assert_called_once();mm.assert_not_called()
            create(dict(backend='registry_framework',framework='mmdetection3d',mmdet_recipe=dict(implementation_supported=True)),*args);mm.assert_called_once()
        with self.assertRaisesRegex(ValueError,'Chưa có DetectorAdapter'):create(dict(backend='registry_framework'),*args)

    def test_mmdet_preflight_uses_model_venv_for_command_and_library_environment(self):
        from annoloop.settings import Settings
        from annoloop.runtime import require_runtime
        import os
        settings=Settings(self.root/'data',Path('/ui/bin/python'),Path('/launcher'),Path('/native/bin/python'))
        def environment(actual):
            self.assertEqual(actual.trainer_python,Path('/mmdet/bin/python'));return {'fixture':'env'}
        with patch.dict(os.environ,{'ANNOLOOP_MMDET3D_PYTHON':'/mmdet/bin/python'}), \
             patch('annoloop.mmdet_adapter.worker_environment',side_effect=environment), \
             patch('annoloop.mmdet_adapter.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='{"ready":true}',stderr='')) as run:
            require_runtime(dict(framework='mmdetection3d',mmdet_recipe=dict(implementation_supported=True)),settings)
            self.assertEqual(run.call_args.args[0][0],'/mmdet/bin/python')
            self.assertEqual(run.call_args.kwargs['env'],{'fixture':'env'})

    def test_download_connect_and_project_check_use_supported_recipe_without_pretrain(self):
        from test_model_hub import model,Response
        from test_adaptation import NativeAdaptationTests
        from annoloop.model_hub import download
        from annoloop.connect_model import connect
        from annoloop import adaptation,preparation
        import hashlib
        fixture=NativeAdaptationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        setup=read(fixture.store.path(fixture.pid)/'setup.json')
        setup['source_counts']={s:{'car':20,'pedestrian':20} for s in ('train','validation')};write(fixture.store.path(fixture.pid)/'setup.json',setup)
        body=b"class_names=['car','pedestrian']\nmodel=dict(type='CenterPoint',pts_bbox_head=dict(type='CenterHead'))\ntest_pipeline=[dict(type='LoadPointsFromFile',load_dim=3,use_dim=3,coord_type='LIDAR')]\n"
        m=dict(model(),framework='mmdetection3d',architecture='CenterPoint',modality='point_cloud')
        m['config_files'][m['config_path']]['sha256']=hashlib.sha256(body).hexdigest()
        def opener(request,timeout):return Response(body,url=request.full_url) if 'raw.githubusercontent' in request.full_url else Response()
        download(fixture.settings.data,m,opener=opener)
        with patch('annoloop.model_hub.load_catalog',return_value=dict(models=[m])), \
             patch('annoloop.hub_selection.load_catalog',return_value=dict(models=[m])), \
             patch('annoloop.adaptation.checkpoint_choices',return_value=[]), \
             patch('annoloop.adaptation.start_job',return_value='no worker') as dispatch:
            connected=connect(fixture.settings.data,m,fixture.pid)
            self.assertTrue(connected['capabilities']['target_training_implemented'])
            self.assertFalse(connected['adaptation_ready'])
            selection={r:dict(kind='hub',id=m['id']) for r in ('A','B')}
            config=adaptation.save_model_selection(fixture.store,fixture.pid,fixture.settings,selection)
            report=preparation.check_prepared(fixture.store,fixture.pid,fixture.settings,True,4)
            self.assertEqual(report['status'],'checking');dispatch.assert_called_once()
            self.assertEqual(dispatch.call_args.kwargs['mode'],'verify')
            self.assertEqual(config['models']['A']['mmdet_recipe']['head'],'CenterHead')
            self.assertFalse(adaptation.readiness(fixture.store,fixture.pid)['ready'])
            self.assertFalse((fixture.store.path(fixture.pid)/'native_verified.json').exists())
            config=adaptation.save_setup(fixture.store,fixture.pid,fixture.settings,
                {r:config['models'][r]['path'] for r in ('A','B')},True,mmdet_input_mode='current_scan_only')
            self.assertEqual(config['mmdet_input_mode'],'current_scan_only')
            self.assertEqual(config['models']['A']['mmdet_recipe']['input_mode'],'current_scan_only')
            # Ordinary LR/epoch edits preserve the explicit input choice.
            saved=adaptation.save_setup(fixture.store,fixture.pid,fixture.settings,
                {r:config['models'][r]['path'] for r in ('A','B')},True)
            self.assertEqual(saved['mmdet_input_mode'],'current_scan_only')
            fixture.mark_verified(config)
            self.assertFalse(adaptation.readiness(fixture.store,fixture.pid)['gpu_verified'])

    def test_training_saves_tenth_and_final_and_resumes_optimizer(self):
        from annoloop.mmdet_worker import train
        model=torch.nn.Linear(1,1);checkpoint=self.root/'initial.pth';checkpoint.write_bytes(b'fixture')
        media=self.root/'frame.npy';np.save(media,np.ones((8,3),dtype=np.float32))
        frame=dict(frame_id='frame',media=str(media),sha256=digest(media))
        box=dict(id='box',class_name='car',box=[1.,1.,1.,5.,5.,5.,0.],score=.9,training_policy='positive')
        labels=self.root/'labels.json';write(labels,dict(target_gold_read=False,frames=[dict(frame_id='frame',boxes=[box],ignored=[])]))
        spec=dict(classes=['car'],architecture='CenterPoint',frames=[frame],labels_path=str(labels),shared_labels_sha256=digest(labels),
            path=str(checkpoint),config_sha256='config',policy=dict(epochs=11,ckpt_interval=10,save_final=True))
        source=dict(source_boxes=[box['box']],source_names=['car'])
        def value(*args):return model(torch.ones((1,1))).square().mean()+.1
        output=self.root/'run';output.mkdir()
        with patch('annoloop.mmdet_worker.prepare',return_value=(torch,model,None,{},{})), \
             patch('annoloop.mmdet_worker.sources',return_value=[source]), \
             patch('annoloop.mmdet_worker.points',return_value=np.ones((8,3),dtype=np.float32)), \
             patch('annoloop.mmdet_worker.loss',side_effect=value):
            train(output,spec)
            self.assertEqual(sorted(p.name for p in (output/'checkpoints').glob('*.pth')),['checkpoint_epoch_10.pth','checkpoint_epoch_11.pth'])
            saved=torch.load(output/'checkpoints/checkpoint_epoch_10.pth',weights_only=False)
            spec['policy']['resume_checkpoint']={'path':str(output/'checkpoints/checkpoint_epoch_10.pth')}
            resume=self.root/'resume';resume.mkdir()
            with patch('annoloop.mmdet_worker.prepare',return_value=(torch,model,None,{},saved)):
                train(resume,spec)
            self.assertEqual(read(resume/'trained.json')['optimizer_steps'],11)
            self.assertEqual(read(resume/'trained.json')['shared_labels_sha256'],digest(labels))
            spec['mmdet_recipe']=dict(input_mode='current_scan_only')
            with patch('annoloop.mmdet_worker.prepare',return_value=(torch,model,None,{},saved)):
                with self.assertRaisesRegex(ValueError,'khác input mode'):train(self.root/'different-mode',spec)


if __name__=='__main__':unittest.main()
