"""Resume contracts using temporary files/CPU optimizer state; no GPU training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from annoloop.io import read, write, digest
from annoloop.projects import ProjectStore
from annoloop.resume import plan, options
from annoloop.inference_cache import seed_target, cached_frame, finish_cached_target
from annoloop.native_training import restore_optimizer
from annoloop.native_adapter import NativeDetectorAdapter
from annoloop.pseudo_memory import empty, MemoryPolicy


class ResumeTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        target=self.root/'target';target.mkdir()
        write(target/'class_schema.json',dict(data_type='3d',annotation_type='cuboid',classes=['car']))
        for name in ('one','two'):np.save(target/(name+'.npy'),np.ones((5,3),np.float32))
        self.store=ProjectStore(self.root/'data');self.pid=self.store.create('fixture',target)
        root=self.store.path(self.pid);self.job=root/'adaptation_jobs/old';self.work=root/'work/M1/attempts/old'
        self.frames=read(root/'dataset.json')['frames']
        models={}
        for k,arch in [('A','SECONDNetIoU'),('B','PVRCNN')]:
            weight=self.root/(k+'.pth');weight.write_bytes(b'fixture '+k.encode())
            cfg=self.root/(k+'.yaml');cfg.write_text('fixture config')
            models[k]=dict(path=str(weight),checkpoint_sha256=digest(weight),config=str(cfg),
                config_sha256=digest(cfg),runtime='fixture',classes=['car'],architecture=arch)
        config=dict(models=models,epochs=3,learning_rate=.0001,source_weight=.5,tracking_mode='off',
                    bin_columns=4,coordinates_confirmed=True,adaptation_classes=['car'])
        write(root/'adaptation.json',config)
        write(root/'setup.json',dict(source_evidence={'fixture':'source only'}))
        write(self.job/'spec.json',dict(mode='run',config=config,source_evidence={'fixture':'source only'},
            dataset_sha256=self.store.get(self.pid)['dataset_sha256']))
        write(self.job/'state.json',dict(status='interrupted',mode='run'))
        self.box=dict(id='box',class_name='car',box=[4.,0.,0.,4.,2.,2.,0.],score=.8,confidence=.8,quality=.8)
        labels=dict(target_gold_read=False,frames=[dict(frame_id=f['frame_id'],boxes=[self.box]) for f in self.frames])
        write(self.work/'initial_shared.json',labels)
        write(self.work/'before/predictions.json',dict(target_gold_read=False,
            checkpoint_hashes={k:m['checkpoint_sha256'] for k,m in models.items()},
            models={k:{f['frame_id']:[self.box] for f in self.frames} for k in models}))
        for k,m in models.items():
            weight=self.work/k/'checkpoints/checkpoint_epoch_3.pth';weight.parent.mkdir(parents=True);weight.write_bytes(b'trained fixture '+k.encode())
            write(self.work/(k+'_trained.json'),dict(checkpoint=str(weight),checkpoint_sha256=digest(weight),epochs=3,
                optimizer_steps=6,classes=['car'],architecture=m['architecture'],source_assisted=True,
                target_gold_used=False,shared_labels_sha256=digest(self.work/'initial_shared.json')))
        self.config=config

    def test_completed_pair_is_recoverable_without_rewriting_old_files(self):
        before={str(p):digest(p) for p in self.work.rglob('*') if p.is_file()}
        p=plan(self.store,self.pid,self.work)
        self.assertEqual(set(p['completed']),{'A','B'});self.assertEqual(p['checkpoints'],{})
        self.assertEqual(len(options(self.store,self.pid)),1)
        self.assertEqual(before,{str(p):digest(p) for p in self.work.rglob('*') if p.is_file()})

    def test_changed_labels_checkpoint_config_or_live_worker_block_resume(self):
        with patch('annoloop.adaptation.job_status',return_value={'process_alive':True}):
            with self.assertRaisesRegex(ValueError,'còn sống'):plan(self.store,self.pid,self.work)
        path=self.work/'initial_shared.json';old=read(path);write(path,dict(old,changed=True))
        with self.assertRaisesRegex(ValueError,'không hợp lệ'):plan(self.store,self.pid,self.work)
        write(path,old)
        weight=Path(read(self.work/'A_trained.json')['checkpoint']);original=weight.read_bytes();weight.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'không hợp lệ'):plan(self.store,self.pid,self.work)
        weight.write_bytes(original)
        config=dict(self.config,epochs=4);write(self.store.path(self.pid)/'adaptation.json',config)
        with self.assertRaisesRegex(ValueError,'Thiết lập đã đổi'):plan(self.store,self.pid,self.work)

    def test_partial_model_with_no_checkpoint_does_not_claim_optimizer_recovery(self):
        (self.work/'B_trained.json').unlink()
        for path in (self.work/'B/checkpoints').glob('*.pth'):path.unlink()
        m=self.config['models']['B']
        write(self.work/'B/spec.json',dict(m,shared_labels_sha256=digest(self.work/'initial_shared.json')))
        p=plan(self.store,self.pid,self.work)
        self.assertEqual(list(p['completed']),['A']);self.assertEqual(p['checkpoints'],{})

    def test_repeated_interruption_keeps_last_optimizer_checkpoint(self):
        (self.work/'B_trained.json').unlink()
        for path in (self.work/'B/checkpoints').glob('*.pth'):path.unlink()
        checkpoint=self.root/'previous_checkpoint_epoch_1.pth';checkpoint.write_bytes(b'previous saved state')
        resume=dict(path=str(checkpoint),sha256=digest(checkpoint),epoch=1)
        job=read(self.job/'spec.json');job['resume_plan']=dict(checkpoints={'B':resume});write(self.job/'spec.json',job)
        m=self.config['models']['B']
        write(self.work/'B/spec.json',dict(m,path=str(checkpoint),checkpoint_sha256=resume['sha256'],
            shared_labels_sha256=digest(self.work/'initial_shared.json'),policy=dict(resume_checkpoint=resume)))
        self.assertEqual(plan(self.store,self.pid,self.work)['checkpoints']['B'],resume)
        newer=self.work/'B/checkpoints/checkpoint_epoch_2.pth';newer.write_bytes(b'newer saved state')
        self.assertEqual(plan(self.store,self.pid,self.work)['checkpoints']['B']['epoch'],2)

    def test_labels_without_matching_before_cache_cannot_be_reused(self):
        (self.work/'before/predictions.json').unlink()
        with self.assertRaisesRegex(ValueError,'thiếu cache'):plan(self.store,self.pid,self.work)

    def test_resume_binds_memory_snapshot_and_rejects_policy_changes(self):
        memory=empty(self.store.get(self.pid)['dataset_sha256'],['car'],MemoryPolicy())
        memory.update(round_index=0,frames=[dict(frame_id=f['frame_id'],objects=[]) for f in self.frames])
        write(self.work/'memory_before.json',memory)
        p=plan(self.store,self.pid,self.work)
        self.assertEqual(p['memory_before_sha256'],digest(self.work/'memory_before.json'))
        config=dict(self.config,memory_policy={'enabled':False})
        write(self.store.path(self.pid)/'adaptation.json',config)
        with self.assertRaisesRegex(ValueError,'memory đã đổi'):plan(self.store,self.pid,self.work)

    def test_real_saved_frames_are_reused_and_missing_frames_remain_missing(self):
        spec=dict(self.config['models']['A'],mode='target_sample',frames=self.frames,score_threshold=.05,
                  iou_threshold=.5,bin_columns=4,coordinates_confirmed=True,require_components=True)
        old=self.job/'A/target/native';write(old/'spec.json',spec)
        box=dict(self.box,cls_score=.8,iou_score=.8)
        write(old/'frames/000000.json',dict(frame_id='one',boxes=[box]))
        fresh=self.root/'new_native'
        spec['reused_frames']=seed_target(self.job,'A',spec,fresh)
        self.assertEqual(len(spec['reused_frames']),1)
        self.assertEqual(cached_frame(fresh,spec,0)['boxes'],[box])
        self.assertIsNone(cached_frame(fresh,spec,1))
        with self.assertRaisesRegex(ValueError,'thiếu frame'):finish_cached_target(fresh,spec)
        write(old/'frames/000001.json',dict(frame_id='two',boxes=[box]))
        spec['reused_frames']=seed_target(self.job,'A',spec,fresh)
        result=finish_cached_target(fresh,spec)
        self.assertEqual((result['reused_frames'],result['new_forward_frames']),(2,0))
        (fresh/'frames/000000.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'đã thay đổi'):cached_frame(fresh,spec,0)

    def test_cache_from_other_model_never_skips_forward(self):
        spec=dict(self.config['models']['A'],mode='target_sample',frames=self.frames,score_threshold=.05,
                  iou_threshold=.5,bin_columns=4,coordinates_confirmed=True,require_components=True)
        old=self.job/'A/target/native';write(old/'spec.json',dict(spec,checkpoint_sha256='different'))
        write(old/'frames/000000.json',dict(frame_id='one',boxes=[]))
        self.assertEqual(seed_target(self.job,'A',spec,self.root/'new'),{})

    def test_adapter_with_all_saved_target_frames_launches_no_gpu_worker(self):
        adapter=NativeDetectorAdapter(self.config['models']['A'],['car'],self.frames,self.root/'fresh_job/A',None)
        spec=dict(adapter.spec(self.config['models']['A']['path']),mode='target_sample')
        old=self.job/'A/target/native';write(old/'spec.json',spec)
        box=dict(self.box,cls_score=.8,iou_score=.8)
        for i,frame in enumerate(self.frames):
            write(old/'frames'/f'{i:06d}.json',dict(frame_id=frame['frame_id'],boxes=[box]))
        adapter.reuse_job=self.job
        with patch.object(adapter,'call',side_effect=AssertionError('Must not launch GPU')) as call:
            predicted=adapter.predict_target(self.config['models']['A']['path'],self.frames)
        call.assert_not_called()
        self.assertEqual(set(predicted),{'one','two'})
        result=read(next((self.root/'fresh_job/A/target').glob('*/result.json')))
        self.assertEqual(result['new_forward_frames'],0)
        self.assertEqual(result['reused_frames'],2)

    def test_optimizer_moments_and_epoch_are_restored_from_checkpoint(self):
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=torch.optim.Adam([parameter],lr=.001)
        parameter.grad=torch.ones_like(parameter);optimizer.step()
        checkpoint=self.root/'optimizer.pth'
        torch.save(dict(epoch=10,it=50,optimizer_state=optimizer.state_dict()),checkpoint)
        fresh=torch.optim.Adam([torch.nn.Parameter(torch.tensor([2.]))],lr=.001)
        spec=dict(path=str(checkpoint),policy=dict(resume_checkpoint=dict(sha256=digest(checkpoint),epoch=10)))
        self.assertEqual(restore_optimizer(torch,spec,fresh,20),(10,50))
        self.assertTrue(torch.equal(fresh.state_dict()['state'][0]['exp_avg'],optimizer.state_dict()['state'][0]['exp_avg']))
        with self.assertRaisesRegex(ValueError,'epoch/optimizer'):restore_optimizer(torch,spec,fresh,5)


if __name__=='__main__':unittest.main()
