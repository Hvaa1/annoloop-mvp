"""Exact convex cuboid intersection, retaining CVAT/THREE Euler XYZ tilt."""
from itertools import combinations
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

TRIPLES = np.array(list(combinations(range(12),3)))

def matrix(rotation):
    return Rotation.from_euler('XYZ', rotation).as_matrix()

def point_support(points, box, rotation):
    local=(points[:,:3]-np.asarray(box[:3]))@matrix(rotation)
    return int(np.all(np.abs(local)<=np.asarray(box[3:6])/2+1e-8,axis=1).sum())

def planes(box, rotation):
    axes=matrix(rotation).T
    normal=np.concatenate([axes,-axes])
    limits=np.tile(np.asarray(box[3:6])/2,2)+normal@np.asarray(box[:3])
    return normal,limits

def cuboid_iou(first, second, first_rotation=None, second_rotation=None):
    first_rotation=[0.,0.,first[6]] if first_rotation is None else first_rotation
    second_rotation=[0.,0.,second[6]] if second_rotation is None else second_rotation
    if np.linalg.norm(np.asarray(first[:3])-second[:3])>(np.linalg.norm(first[3:6])+np.linalg.norm(second[3:6]))/2:
        return 0.
    a,b=planes(first,first_rotation);c,d=planes(second,second_rotation)
    normal=np.concatenate([a,c]);limits=np.concatenate([b,d])
    systems=normal[TRIPLES];rhs=limits[TRIPLES]
    valid=np.abs(np.linalg.det(systems))>1e-9
    vertices=np.linalg.solve(systems[valid],rhs[valid,...,None])[...,0]
    inside=np.all(vertices@normal.T<=limits+1e-7,axis=1)
    vertices=np.unique(np.round(vertices[inside],9),axis=0)
    if len(vertices)<4:return 0.
    if np.linalg.matrix_rank(vertices-vertices[0],tol=1e-8)<3:return 0.
    volume=ConvexHull(vertices).volume
    total=np.prod(first[3:6])+np.prod(second[3:6])-volume
    return float(np.clip(volume/total,0.,1.))
