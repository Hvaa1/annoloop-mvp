"""Preselection must not confuse taxonomy, input features and runtime readiness."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from annoloop.model_filter import point_features,dataset_profile,assess
from annoloop.model_hub import load_catalog


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.pcd=self.root/'one.pcd'
        self.pcd.write_bytes(b'VERSION 0.7\nFIELDS x y z rgb\nDATA binary\n\xff\xfe')
        self.frame=dict(frame_id='one',media=str(self.pcd))
        self.profile=dataset_profile([self.frame])

    def test_rgb_is_not_intensity_and_multisweep_is_not_a_sequence_id(self):
        catalog={m['id']:m for m in load_catalog()['models']}
        for mid in ['mmdet3d-fb8e603dd7af17a8','mmdet3d-33bd8edf79827397']:
            with patch('urllib.request.urlopen',side_effect=AssertionError('no download')):
                row=assess(catalog[mid],['car','pedestrian'],self.profile)
            self.assertFalse(row['eligible']);self.assertEqual(row['missing_classes'],[])
            self.assertIn('intensity',row['missing_features'])
            self.assertTrue(any('adapter adaptation' in r for r in row['reasons']))
        row=assess(catalog['mmdet3d-33bd8edf79827397'],['car'],dataset_profile([dict(self.frame,sequence_id='seq',timestamp=1.)]))
        self.assertTrue(any('10 sweep' in r for r in row['reasons']))

    def test_native_xyz_is_a_candidate_but_not_gpu_verified(self):
        config=self.root/'config.yaml'
        config.write_text('DATA_CONFIG:\n  POINT_FEATURE_ENCODING:\n    used_feature_list: [x, y, z]\n')
        native=dict(backend='native_st3d',config=str(config),classes=['car','pedestrian'])
        row=assess(native,['car'],self.profile)
        self.assertTrue(row['eligible']);self.assertFalse(row['gpu_verified'])
        row=assess(dict(native,classes=['Vehicle']),['car'],self.profile)
        self.assertFalse(row['eligible']);self.assertEqual(row['missing_classes'],['car'])

    def test_unknown_contract_is_not_accepted_even_if_flag_says_ready(self):
        row=assess(dict(classes=['car'],runtime_verified=True,adaptation_ready=True),['car'],self.profile)
        self.assertFalse(row['eligible'])
        self.assertTrue(any('Chưa xác minh' in r for r in row['reasons']))

    def test_source_class_availability_is_independent_of_checkpoint_coverage(self):
        cfg=self.root/'source.yaml';cfg.write_text('DATA_CONFIG:\n  POINT_FEATURE_ENCODING:\n    used_feature_list: [x, y, z]\n')
        native=dict(backend='native_st3d',config=str(cfg),classes=['car','pedestrian'])
        row=assess(native,['car','pedestrian'],self.profile,
                   source_counts={'train':{'car':5,'pedestrian':2},'validation':{'car':1}})
        self.assertFalse(row['eligible']);self.assertEqual(row['missing_classes'],[])
        self.assertIn('Source validation thiếu nhãn: pedestrian',row['reasons'])

    def test_preselection_samples_across_dataset_and_does_not_read_annotations(self):
        frames=[dict(frame_id=str(i),media='unused',annotation='never-read') for i in range(30)]
        with patch('annoloop.model_filter.point_features',return_value=['x','y','z']) as header:
            profile=dataset_profile(frames,limit=3)
        self.assertEqual([c.args[0]['frame_id'] for c in header.call_args_list],['0','14','29'])
        self.assertEqual((profile['checked'],profile['total']),(3,30))

    def test_mixed_features_use_intersection_and_failed_headers_block(self):
        other=self.root/'two.pcd';other.write_text('FIELDS x y z\nDATA ascii\n')
        profile=dataset_profile([self.frame,dict(frame_id='two',media=str(other))])
        self.assertTrue(profile['mixed']);self.assertEqual(profile['features'],['x','y','z'])
        profile=dataset_profile([self.frame,dict(frame_id='bad',media=str(self.root/'missing.pcd'))])
        self.assertFalse(profile['complete']);self.assertEqual(len(profile['errors']),1)

    def test_npy_headers_and_bin_unknown_features(self):
        np.save(self.root/'xyz.npy',np.ones((4,3),np.float32))
        np.save(self.root/'four.npy',np.ones((4,4),np.float32))
        self.assertEqual(point_features(dict(media=str(self.root/'xyz.npy'))),['x','y','z'])
        with self.assertRaises(ValueError):point_features(dict(media=str(self.root/'four.npy')))
        with self.assertRaises(ValueError):point_features(dict(media=str(self.root/'cloud.bin')))


if __name__=='__main__':unittest.main()
