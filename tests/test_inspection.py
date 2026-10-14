"""Metric matching and checkpoint UI contracts; no GPU or model outputs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from annoloop.inspection import matched_counts,load_xyz,checkpoint_choices,start_inspection,inspection_status
from annoloop.projects import ProjectStore
from annoloop.settings import Settings
from annoloop.io import write,read
import yaml


class InspectionTests(unittest.TestCase):
    def test_one_gt_cannot_match_multiple_detections(self):
        self.assertEqual(matched_counts(np.array([[.8],[.9]]),np.array([.9,.8]),.5),dict(tp=1,fp=1,fn=0))

    def test_score_order_and_class_filter_input_support_empty_predictions(self):
        self.assertEqual(matched_counts(np.zeros((0,3)),[],.5),dict(tp=0,fp=0,fn=3))
        self.assertEqual(matched_counts(np.zeros((2,0)),[.9,.8],.5),dict(tp=0,fp=2,fn=0))

    def test_binary_layout_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'points.bin';np.zeros((3,4),dtype='<f4').tofile(path)
            with self.assertRaises(ValueError):load_xyz(path)
            self.assertEqual(load_xyz(path,4).shape,(3,3))

    def fixture(self,root):
        data=root/'data';target=root/'target';target.mkdir()
        np.save(target/'points.npy',np.zeros((4,3),dtype=np.float32))
        write(target/'class_schema.json',dict(data_type='3d',annotation_type='cuboid',classes=['car','pedestrian']))
        store=ProjectStore(data);pid=store.create('fixture',target)
        config=store.path(pid)/'generated/config.yaml';config.parent.mkdir()
        config.write_text(yaml.safe_dump(dict(CLASS_NAMES=['car','pedestrian'],MODEL=dict(NAME='SECONDNetIoU'))))
        job=data/'jobs'/'fixture';job.mkdir(parents=True)
        write(job/'job.json',dict(spec=dict(config=str(config),runtime='/runtime')))
        weights=job/'checkpoints/checkpoint_epoch_10.pth';weights.parent.mkdir();weights.write_bytes(b'fixture only')
        settings=Settings(data,Path('/python'),Path('/launcher'))
        return store,pid,settings,weights

    def test_intermediate_checkpoint_listed_without_promoting_training(self):
        with tempfile.TemporaryDirectory() as folder:
            store,pid,settings,weights=self.fixture(Path(folder))
            choices=checkpoint_choices(store,pid,settings)
            self.assertEqual(choices[0]['path'],str(weights));self.assertTrue(choices[0]['intermediate'])
            self.assertEqual(store.get(pid)['completed_rounds'],[])

    def test_target_inference_requires_coordinate_confirmation_before_job(self):
        with tempfile.TemporaryDirectory() as folder:
            store,pid,settings,weights=self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError,'Xác nhận'):start_inspection(store,pid,str(weights),'target_sample',settings)
            self.assertFalse((store.path(pid)/'inspections').exists())

    def test_dead_inspection_not_reported_running(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write(root/'inspection.json',dict(status='running',pid=123,process_start='old'))
            with patch('annoloop.inspection.process_identity',return_value=None):
                self.assertEqual(inspection_status(root)['status'],'interrupted')


if __name__=='__main__':unittest.main()
