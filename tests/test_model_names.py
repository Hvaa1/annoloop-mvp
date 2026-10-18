"""Readable catalog names and the usable / download / unusable sort."""
import unittest
from annoloop.model_names import friendly, verdict


def report(**extra):
    return dict(dict(missing_classes=[], selectable=True, selection_blockers=[], eligible=False, reasons=[]), **extra)


class ModelNamesTest(unittest.TestCase):
    def test_friendly_names_decode_openmmlab_tokens(self):
        url = 'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/x/x_20220811_045458-808e69ad.pth'
        model = dict(family='CenterPoint', dataset='nuScenes', weights_url=url,
                     name='centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d')
        self.assertEqual(friendly(model), 'CenterPoint · pillar 0.2 m · SECFPN · DCN · nuScenes · 2022')
        model['name'] = 'centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d'
        self.assertIn('voxel 0.075 m', friendly(model))
        model.update(family='FreeAnchor', name='pointpillars_hv_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d')
        self.assertEqual(friendly(model).count('FreeAnchor'), 1)

    def test_verdict_keeps_runnable_and_downloadable_hides_known_failures(self):
        v1 = 'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/a.pth'
        v0 = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/a.pth'
        mm = dict(framework='mmdetection3d', family='PointPillars', weights_url=v1)
        self.assertEqual(verdict(mm, report(eligible=True), True)[0], 'usable')
        self.assertEqual(verdict(mm, report(), False)[0], 'download')
        self.assertEqual(verdict(dict(mm, weights_url=v0), report(), False)[0], 'unusable')
        self.assertEqual(verdict(dict(mm, family='FreeAnchor'), report(), False)[0], 'unusable')
        self.assertEqual(verdict(dict(mm, framework='OpenPCDet'), report(), False)[0], 'unusable')
        self.assertEqual(verdict(mm, report(missing_classes=['pedestrian']), False)[0], 'unusable')
        connected = dict(mm, mmdet_recipe=dict(blockers=['Chưa có loss/ignore recipe cho head X']))
        self.assertEqual(verdict(connected, report(), True), ('unusable', 'Chưa có loss/ignore recipe cho head X'))


if __name__ == '__main__':
    unittest.main()
