"""Memory transitions and persistence on CPU fixtures; no model jobs/gold reads."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from annoloop.io import read, write, digest
from annoloop.pseudo_memory import MemoryPolicy, empty, update, validate
from annoloop.memory_store import context, load
from annoloop.projects import ProjectStore
from annoloop.native_data import signed_labels
from annoloop.selection import SelectionPolicy


def box(identity='car', name='car', x=4., confidence=.8, quality=.8):
    return dict(id=identity, class_name=name, box=[x,0.,0.,4.,2.,2.,0.], score=.8,
                confidence=confidence, quality=quality, training_policy='positive')


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.policy=MemoryPolicy();self.classes=['car','pedestrian']
        self.memory=empty('dataset',self.classes,self.policy)
        self.points=np.asarray([[4.,0.,0.]]*20,np.float32)
        self.index=-1

    def step(self, boxes, ignored=None, raw=None, weights=None, points=None, frame='f'):
        self.index+=1
        labels=dict(target_gold_read=False,frames=[dict(frame_id=frame,boxes=boxes,ignored=ignored or [])])
        raw=raw if raw is not None else {'A':{frame:boxes},'B':{frame:boxes}}
        weights=weights or {'A':'A'+str(self.index),'B':'B'+str(self.index)}
        self.memory,result=update(self.memory,labels,raw,context(self.index,weights),
                                 lambda f:self.points if points is None else points,self.policy)
        return result['frames'][0]

    def stable(self):
        self.step([box('first')]);return self.step([box('second')])

    def test_stable_id_and_history_follow_matching_geometry_not_native_id(self):
        first=self.step([box('first')]);identity=first['boxes'][0]['id']
        second=self.step([box('different',x=4.1)])
        self.assertEqual(second['boxes'][0]['id'],identity)
        entry=self.memory['frames'][0]['objects'][0]
        self.assertEqual(entry['hit_count'],2);self.assertEqual(entry['stable_hits'],2)
        self.assertEqual([h['round_index'] for h in entry['history']],[0,1])
        self.assertGreater(entry['history'][-1]['match_iou'],.9)
        self.assertEqual(entry['box']['box'][0],4.1)

    def test_stable_box_retained_once_then_ignored_then_expired(self):
        identity=self.stable()['boxes'][0]['id']
        missing=self.step([])
        self.assertEqual(missing['boxes'][0]['id'],identity)
        self.assertEqual(missing['boxes'][0]['reason'],'memory_retained_missing')
        self.assertEqual(missing['boxes'][0]['confidence'],.8)
        self.assertIsNone(self.memory['frames'][0]['objects'][0]['history'][-1]['confidence'])
        ignored=self.step([])
        self.assertFalse(ignored['boxes']);self.assertEqual(ignored['ignored'][0]['training_policy'],'ignore_region')
        expired=self.step([])
        self.assertFalse(expired['boxes']);self.assertFalse(expired['ignored'])
        self.assertEqual(self.memory['summary']['expired'],1)
        self.assertEqual(len(self.memory['frames'][0]['objects'][0]['history']),5)

    def test_single_observation_cannot_be_carried_as_positive(self):
        self.step([box()]);missing=self.step([])
        self.assertFalse(missing['boxes']);self.assertTrue(missing['ignored'])

    def test_same_checkpoint_cannot_fabricate_multiple_confirmations(self):
        weights={'A':'fixedA','B':'fixedB'}
        self.step([box()],weights=weights);self.step([box()],weights=weights)
        entry=self.memory['frames'][0]['objects'][0]
        self.assertEqual(entry['hit_count'],1);self.assertEqual(entry['stable_hits'],1)
        self.assertFalse(entry['history'][-1]['independent_evidence'])
        self.assertFalse(self.step([])['boxes'])

    def test_current_low_quality_or_class_conflict_overrides_history(self):
        self.stable()
        low=box('weak',quality=.2)
        result=self.step([],raw={'A':{'f':[low]},'B':{'f':[low]}})
        self.assertFalse(result['boxes']);self.assertEqual(result['ignored'][0]['reason'],'memory_conflict')
        self.assertEqual(result['ignored'][0]['memory']['stable_hits'],0)

    def test_class_changes_do_not_automatically_relabel_an_old_box(self):
        self.stable()
        pedestrian=box('new-class',name='pedestrian')
        result=self.step([pedestrian])
        self.assertFalse(result['boxes'])
        self.assertEqual({b['class_name'] for b in result['ignored']},{'car','pedestrian'})

    def test_retention_requires_real_current_point_support(self):
        self.stable()
        result=self.step([],points=np.zeros((20,3),np.float32))
        self.assertFalse(result['boxes']);self.assertEqual(result['ignored'][0]['point_count'],0)

    def test_single_detector_support_can_keep_a_stable_box_boundedly(self):
        self.stable()
        strong=box('A-only')
        result=self.step([],raw={'A':{'f':[strong]},'B':{'f':[]}})
        self.assertEqual(result['boxes'][0]['reason'],'memory_retained_supported')
        self.assertEqual(self.memory['summary']['retained'],1)
        self.assertFalse(self.step([],raw={'A':{'f':[strong]},'B':{'f':[]}})['boxes'])

    def test_large_calibrated_score_drop_is_ignored_not_overwritten_as_positive(self):
        self.stable()
        result=self.step([box(confidence=.65)])
        self.assertFalse(result['boxes'])
        self.assertEqual(self.memory['frames'][0]['objects'][0]['conflict_count'],1)

    def test_matching_but_unstable_geometry_does_not_earn_retention(self):
        self.step([box()]);self.step([box(x=5.5)])
        self.assertEqual(self.memory['frames'][0]['objects'][0]['stable_hits'],1)
        self.assertFalse(self.step([])['boxes'])

    def test_expired_box_can_reappear_with_same_id_but_needs_new_stability(self):
        identity=self.stable()['boxes'][0]['id']
        for _ in range(3):self.step([])
        result=self.step([box('reappeared')])
        self.assertEqual(result['boxes'][0]['id'],identity)
        self.assertEqual(result['boxes'][0]['memory']['stable_hits'],1)

    def test_retained_positive_does_not_also_become_signed_ignore(self):
        self.stable()
        ignored=dict(box('old-raw'),training_policy='ignore_region',reason='unmatched')
        result=self.step([],ignored=[ignored])
        signed,_,_=signed_labels(result,self.classes)
        self.assertEqual(len(signed),1);self.assertEqual(signed[0,-1],1.)

    def test_frames_are_separate_and_invalid_gold_or_history_is_rejected(self):
        labels=dict(target_gold_read=False,frames=[dict(frame_id=f,boxes=[box()]) for f in ('f','g')])
        raw={k:{f:[box()] for f in ('f','g')} for k in ('A','B')}
        memory,result=update(self.memory,labels,raw,context(0,{'A':'A','B':'B'}),
                             lambda f:self.points,self.policy)
        self.assertNotEqual(result['frames'][0]['boxes'][0]['id'],result['frames'][1]['boxes'][0]['id'])
        corrupted=deepcopy(memory);corrupted['frames'][0]['objects'][0]['history'].append(
            corrupted['frames'][0]['objects'][0]['history'][0])
        with self.assertRaisesRegex(ValueError,'history'):validate(corrupted,'dataset',self.classes,['f','g'])
        with self.assertRaisesRegex(ValueError,'khác dataset'):validate(memory,'other',self.classes)
        with self.assertRaisesRegex(ValueError,'gold'):
            update(self.memory,dict(labels,gold=[]),raw,context(0,{}),lambda f:self.points,self.policy)

    def test_update_is_immutable_and_cannot_reprocess_a_round(self):
        before=deepcopy(self.memory);self.step([box()])
        self.assertEqual(before['round_index'],-1);self.assertEqual(before['frames'],[])
        with self.assertRaisesRegex(ValueError,'theo thứ tự'):
            update(self.memory,dict(target_gold_read=False,frames=[]),{},context(0,{}),lambda f:self.points,self.policy)

    def test_disabled_memory_leaves_labels_unchanged(self):
        labels=dict(target_gold_read=False,frames=[dict(frame_id='f',boxes=[box()])])
        memory,result=update(self.memory,labels,{},context(0,{}),lambda f:self.points,MemoryPolicy(enabled=False))
        self.assertIsNone(memory);self.assertEqual(result,labels)

    def test_policy_rejects_unbounded_retention_or_one_hit_stability(self):
        for kwargs in (dict(stable_hits=1),dict(keep_misses=3),dict(match_iou=.8,stable_iou=.5),dict(min_points=0)):
            with self.assertRaises(ValueError):MemoryPolicy(**kwargs)


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.dataset=self.root/'dataset';self.dataset.mkdir()
        write(self.dataset/'class_schema.json',dict(data_type='3d',annotation_type='cuboid',classes=['car']))
        np.save(self.dataset/'f.npy',np.asarray([[4.,0.,0.]]*20,np.float32))
        write(self.dataset/'gold.json',{'ground_truth':'must never be read'})
        self.store=ProjectStore(self.root/'storage');self.pid=self.store.create('fixture',self.dataset,max_rounds=4,
            policy=SelectionPolicy(mode='automatic'))
        self.frames=read(self.store.path(self.pid)/'dataset.json')['frames'];self.policy=MemoryPolicy()
        self.hash=self.store.get(self.pid)['dataset_sha256']

    def checkpoint(self,index):
        result={}
        for key in ('A','B'):
            path=self.root/(key+str(index)+'.pth');path.write_bytes(b'fixture '+key.encode()+str(index).encode())
            result[key]=path
        return result

    def record(self,index,memory=None):
        labels=dict(target_gold_read=False,frames=[dict(frame_id='f',boxes=[box()])])
        points=lambda f:np.load(f['media'])
        checkpoints=self.checkpoint(index)
        if memory is not None:
            memory,labels=update(memory,labels,{'A':{'f':[box()]},'B':{'f':[box()]}},
                context(index,{k:digest(v) for k,v in checkpoints.items()}),lambda f:points(self.frames[0]),self.policy)
        metrics=dict(domain='target_unlabelled_proxy',target_gold_read=False,box_count=1,agreement=.9,supported_fraction=1.)
        self.store.record_round(self.pid,index,labels,metrics,checkpoints,{'f':dict(boxes=labels['frames'][0]['boxes'])},memory=memory)
        return memory

    def test_legacy_replay_keeps_snapshots_unchanged_and_never_opens_target_gold(self):
        self.record(1);self.record(2)
        root=self.store.path(self.pid)
        before={str(p):digest(p) for p in (root/'rounds').rglob('*') if p.is_file()}
        point_reads=[]
        def guarded_read(path):
            self.assertNotEqual(Path(path),self.dataset/'gold.json')
            return read(path)
        def points(frame):point_reads.append(frame['frame_id']);return np.load(frame['media'])
        with patch('annoloop.memory_store.read',side_effect=guarded_read):
            memory=load(self.store,self.pid,self.frames,['car'],self.policy,points)
        self.assertEqual(memory['round_index'],2)
        self.assertEqual(memory['frames'][0]['objects'][0]['stable_hits'],2)
        self.assertEqual(point_reads,['f'])
        self.assertEqual(before,{str(p):digest(p) for p in (root/'rounds').rglob('*') if p.is_file()})
        self.assertFalse((root/'rounds/M2/memory.json').exists())

    def test_memory_snapshot_hash_and_geometry_bind_selection_and_loading(self):
        memory=empty(self.hash,['car'],self.policy)
        memory,_=update(memory,dict(target_gold_read=False,frames=[dict(frame_id='f',boxes=[box()])]),
            {'A':{'f':[box()]},'B':{'f':[box()]}},context(0,{'A':'initialA','B':'initialB'}),
            lambda f:np.load(self.frames[0]['media']),self.policy)
        self.record(1,memory)
        root=self.store.path(self.pid);receipt=read(root/'rounds/M1/round.json')
        self.assertEqual(receipt['memory_sha256'],digest(root/'rounds/M1/memory.json'))
        with patch('annoloop.memory_store.update_frame',side_effect=AssertionError('Do not replay stored memory')):
            loaded=load(self.store,self.pid,self.frames,['car'],self.policy,lambda f:self.fail())
        self.assertEqual(loaded['round_index'],1)
        document=read(root/'rounds/M1/memory.json');document['frames'][0]['objects'][0]['box']['box'][0]+=1
        write(root/'rounds/M1/memory.json',document)
        with self.assertRaisesRegex(ValueError,'Memory'):self.store.select(self.pid,'M1')
        with self.assertRaisesRegex(ValueError,'Memory'):load(self.store,self.pid,self.frames,['car'],self.policy,lambda f:self.fail())


if __name__=='__main__':unittest.main()
