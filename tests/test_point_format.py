"""KITTI target contracts use geometry-only evidence, never target gold."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from annoloop.io import digest, read, write
from annoloop.inventory import inspect_dataset
from annoloop.inspection import load_xyz
from annoloop.model_filter import point_features, dataset_profile
from annoloop.point_format import binary_contract,verification_contract
from annoloop.framework_worker import points_for
from annoloop.projects import ProjectStore
from annoloop.schema import ClassSchema
from annoloop.setup import inspect_target


class PointFormatTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.dataset=self.root/'kitti'
        self.path=self.dataset/'training/velodyne/000001.bin'
        self.path.parent.mkdir(parents=True)
        self.points=np.array([[1,2,3,.1],[4,5,6,.2],[7,8,9,.3]],dtype='<f4')
        self.points.tofile(self.path)
        self.frame=dict(frame_id='legacy-id',media=str(self.path),sha256=digest(self.path))
        self.format=self.dataset/'pointcloud_format.json'
        self.document=dict(schema_version='1.0',dtype='float32_le',format='kitti_velodyne',
            feature_names=['x','y','z','intensity'],
            files={'training/velodyne/000001.bin':dict(sha256=digest(self.path),points=3)})
        write(self.format,self.document)
        self.schema=ClassSchema('3d','cuboid',('car','pedestrian','motorcycle'))

    def test_legacy_snapshot_gets_xyzi_without_target_labels_or_rewriting(self):
        (self.dataset/'label_2').mkdir()
        (self.dataset/'label_2/000001.txt').write_text('MUST NOT OPEN')
        (self.dataset/'subset_manifest.json').write_text('MUST NOT OPEN')
        before=dict(self.frame);original=Path.open
        def guarded(path,*args,**kwargs):
            if path.parent.name in ('label_2','calib') or path.name=='subset_manifest.json':
                raise AssertionError('Target gold was opened')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            self.assertEqual(point_features(self.frame),['x','y','z','intensity'])
            profile=dataset_profile([self.frame]);self.assertTrue(profile['complete'])
            target=inspect_target(self.dataset);self.assertTrue(target['point_profile']['complete'])
            np.testing.assert_array_equal(points_for(self.frame,['x','y','z','intensity']),self.points)
        self.assertEqual(self.frame,before)

    def test_unknown_bin_is_not_assumed_to_be_kitti_by_shape_or_filename(self):
        self.format.unlink()
        self.assertIsNone(binary_contract(self.frame))
        with self.assertRaises(ValueError):point_features(self.frame)

    def test_wrong_three_column_choice_rejected_even_when_bytes_divisible(self):
        self.assertEqual(self.path.stat().st_size % 12,0)
        for callback in (lambda:load_xyz(self.path,3),lambda:points_for(self.frame,['x','y','z'],3)):
            with self.assertRaisesRegex(ValueError,'khớp'):callback()
        np.testing.assert_array_equal(load_xyz(self.path),self.points[:,:3])
        with self.assertRaisesRegex(ValueError,'khớp point format'):
            point_features(dict(self.frame,feature_names=['x','y','z']))

    def test_fresh_inventory_preserves_format_evidence_and_invalidates_change(self):
        store=ProjectStore(self.root/'state')
        pid=store.create('KITTI target',self.dataset,schema_override=self.schema)
        snapshot=read(store.path(pid)/'dataset.json');frame=snapshot['frames'][0]
        self.assertEqual(frame['feature_names'],['x','y','z','intensity'])
        self.assertEqual(frame['point_format_evidence']['sha256'],digest(self.format))
        store.verify_dataset(pid)
        self.document['format']='edited';write(self.format,self.document)
        with self.assertRaisesRegex(ValueError,'point format'):store.verify_dataset(pid)

    def test_modified_hash_missing_frame_shape_and_dtype_are_rejected(self):
        for field in ('hash','missing','shape','dtype'):
            with self.subTest(field=field):
                doc=json.loads(json.dumps(self.document));row=doc['files']['training/velodyne/000001.bin']
                if field=='hash':row['sha256']='0'*64
                elif field=='missing':doc['files']={'other.bin':row}
                elif field=='shape':row['points']=4
                else:doc['dtype']='float64_le'
                write(self.format,doc)
                with self.assertRaises(ValueError):point_features(self.frame)

    def test_manifest_declared_feature_names_are_not_lost_on_import(self):
        self.format.unlink()
        write(self.dataset/'manifest.json',dict(feature_names=['x','y','z','intensity'],
            frames=[dict(frame_id='one',points='training/velodyne/000001.bin')]))
        frame=inspect_dataset(self.dataset,self.schema)['frames'][0]
        self.assertEqual(point_features(frame),['x','y','z','intensity'])
        self.assertNotIn('point_format_evidence',frame)

    def test_verification_pins_legacy_format_and_blocks_wrong_columns(self):
        config=dict(bin_columns=4,coordinates_confirmed=True)
        first=verification_contract('dataset',config,[self.frame])
        self.assertEqual(first['input_formats'],{str(self.format):digest(self.format)})
        self.document['format']='same features, changed metadata';write(self.format,self.document)
        self.assertNotEqual(first,verification_contract('dataset',config,[self.frame]))
        with self.assertRaisesRegex(ValueError,'Số cột BIN'):
            verification_contract('dataset',dict(config,bin_columns=3),[self.frame])
        unknown=self.root/'unknown.bin';unknown.write_bytes(b'')
        self.assertNotIn('input_formats',verification_contract('dataset',config,[dict(media=str(unknown))]))


if __name__=='__main__':unittest.main()
