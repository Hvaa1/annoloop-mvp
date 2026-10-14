"""Geometry helpers extracted from the demo; no fixed class catalogue."""
import numpy as np

def points_in_box(points, box):
    x, y, z, length, width, height, yaw = box
    d = points[:, :3] - [x, y, z]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.column_stack((c*d[:, 0]+s*d[:, 1], -s*d[:, 0]+c*d[:, 1], d[:, 2]))
    return np.all(np.abs(local) <= np.array([length, width, height])/2 + 1e-8, axis=1)

def corners(box):
    x, y, z, length, width, height, yaw = box
    local = np.array([[a*length/2, b*width/2, d*height/2]
                      for d in (-1, 1) for a, b in ((-1,-1),(1,-1),(1,1),(-1,1))])
    c, s = np.cos(yaw), np.sin(yaw)
    local[:, :2] = local[:, :2] @ np.array([[c, s], [-s, c]])
    return local + [x, y, z]
from ..predictions import validate_predictions
