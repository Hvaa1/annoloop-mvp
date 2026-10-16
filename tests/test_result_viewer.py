"""Display filters must not promote ignore boxes or rewrite saved labels."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from annoloop.io import write
from result_viewer import filtered,figure_for,counts


def box(name,identity):
    return dict(id=identity,class_name=name,box=[0.,0.,0.,4.,2.,2.,0.],score=.4,
                confidence=.4,quality=.25)


class ResultViewerTests(unittest.TestCase):
    def test_pedestrian_ignore_is_visible_without_becoming_positive(self):
        ped=dict(box('pedestrian','p'),training_policy='ignore_region',reason='low_calibrated_quality')
        document=dict(boxes=[box('car','c')],ignored=[ped]);before=deepcopy(document)
        keep,review=filtered(document,['pedestrian'],'Được giữ + ignore')
        self.assertFalse(keep);self.assertEqual(review,[ped])
        figure=figure_for(keep,review,np.zeros((3,3)))
        self.assertEqual(figure.data[0].name,'Point cloud')
        self.assertEqual(figure.data[1].name,'pedestrian · ignore')
        self.assertEqual(figure.data[1].line.dash,'dash')
        self.assertIn('quality: 0.25',figure.data[1].hovertemplate)
        self.assertEqual(document,before)

    def test_accepted_and_ignore_filters_are_independent(self):
        document=dict(boxes=[box('car','a')],ignored=[box('car','b')])
        self.assertEqual(tuple(map(len,filtered(document,['car'],'Chỉ box được giữ'))),(1,0))
        self.assertEqual(tuple(map(len,filtered(document,['car'],'Chỉ ignore'))),(0,1))
        self.assertEqual(tuple(map(len,filtered(document,[],'Được giữ + ignore'))),(0,0))

    def test_counts_preserve_class_and_rejection_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            document=dict(frame_id='one',boxes=[box('car','a')],
                ignored=[dict(box('pedestrian','p'),reason='unmatched')])
            write(Path(temp)/'frame.json',document)
            accepted,ignored,reasons=counts(temp,[dict(frame_id='one',file='frame.json')],[(1,1)])
            self.assertEqual(accepted,{'car':1});self.assertEqual(ignored,{'pedestrian':1})
            self.assertEqual(reasons,[('pedestrian','unmatched',1)])

    def test_indexed_frame_mismatch_is_not_shown_as_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            write(Path(temp)/'frame.json',dict(frame_id='different',boxes=[]))
            with self.assertRaisesRegex(ValueError,'không khớp'):
                counts(temp,[dict(frame_id='expected',file='frame.json')],[(1,1)])
