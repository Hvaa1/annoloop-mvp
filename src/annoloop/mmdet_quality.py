"""Measured rotation consistency, never an invented native IoU-head score."""
from copy import deepcopy
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .reused.geometry import cuboid_iou

ANGLE=.05

def rotate_points(points,angle):
    out=points.copy();c,s=math.cos(angle),math.sin(angle)
    out[:,0]=points[:,0]*c-points[:,1]*s;out[:,1]=points[:,0]*s+points[:,1]*c
    return out


def consistency(first,transformed,angle=ANGLE):
    second=deepcopy(transformed)
    for item in second:
        b=item['box'];x,y=b[:2];c,s=math.cos(angle),math.sin(angle)
        b[0]=c*x+s*y;b[1]=-s*x+c*y;b[6]-=angle
    matrix=np.asarray([[cuboid_iou(a['box'],b['box']) if a['class_name']==b['class_name'] else 0.
                        for b in second] for a in first],dtype=float).reshape(len(first),len(second))
    matches={}
    if len(first) and len(second):
        rows,cols=linear_sum_assignment(1-matrix);matches={i:float(matrix[i,j]) for i,j in zip(rows,cols)}
    output=deepcopy(first)
    for i,box in enumerate(output):
        box.update(cls_score=box['score'],iou_score=matches.get(i,0.),
                   quality_semantics='rotation_consistency_iou',quality_is_native_iou_head=False)
    return output
