#!/usr/bin/env python

import os
import numpy as np
import pytest

from mcvine.acc.components.optics import guide_anyshape
from mcvine.acc import test

thisdir = os.path.dirname(__file__)

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_propagate():
    in_neutron = np.array([
        0, 0.01, -20,
        0, -1, 2000,
        0, 0,
        0, 1,
    ])
    faces = np.array([[
        [-1, 0, -1],
        [-1, 0, 1],
        [1, 0, 1],
        [1, 0, -1],
    ]])
    centers = np.array([
        [0,0,0]
    ])
    unitvecs = np.array([[
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]])
    R0=0.99; Qc=0.0219; alpha=3; m=2; W=0.003
    faces2d = np.array([[
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1],
    ]])
    tmp1 = np.zeros(3)
    nfaces = len(faces)
    intersections = np.zeros(nfaces, dtype=np.float64)
    face_indexes = np.zeros(nfaces, dtype=np.int)
    guide_anyshape._propagate(
        in_neutron,
        faces, centers, unitvecs, faces2d,
        R0, Qc, alpha, m, W,
        tmp1, intersections, face_indexes
    )
    assert np.allclose(
        in_neutron,
        [
            0, 0, 0,
            0, 1, 2000,
            0, 0,
            0.01,
            0.99
        ]
    )
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_propagate2():
    geometry=os.path.join(thisdir, './data/guide_anyshape_straight_3.5cmX3.5cmX10m.off')
    faces, centers, unitvecs, faces2d = guide_anyshape.get_faces_data(geometry, xwidth=0, yheight=0, zdepth=0, center=False)
    R0=0.99; Qc=0.0219; alpha=3; m=2; W=0.003
    tmp1 = np.zeros(3)
    nfaces = len(faces)
    intersections = np.zeros(nfaces, dtype=np.float64)
    face_indexes = np.zeros(nfaces, dtype=np.int)
    neutron = np.array([
        0, 0, -1,
        0, 0.035/2*1000, 1*1000+1E-10,
        0, 0,
        0, 1,
    ])
    guide_anyshape._propagate(
        neutron,
        faces, centers, unitvecs, faces2d,
        R0, Qc, alpha, m, W,
        tmp1, intersections, face_indexes
    )
    # print(neutron)
    assert np.allclose(
        neutron[:-1],
        [
            0, 0.0175, 8,
            0, -17.5, 1000,
            0, 0,
            0.009,
        ]
    )
    assert neutron[-1] < 5E-18 # 4.78366558e-18]
    return
