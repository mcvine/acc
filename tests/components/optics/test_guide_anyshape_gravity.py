#!/usr/bin/env python

import os
import numpy as np
import pytest

from mcvine.acc.components.optics import guide_anyshape_gravity
from mcvine.acc import test

thisdir = os.path.dirname(__file__)

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_intersect():
    neutron = np.array([
        0, 0.01, -20,
        0, -1, 2000,
        0, 0,
        0, 1,
    ])
    gravity = np.array([0, -9.8, 0])
    center = np.array([0,0,0])
    normal = np.array([0,1,0])
    out = np.zeros(2)
    guide_anyshape_gravity.intersect_plane(
        neutron[:3], neutron[3:6], gravity,
        center, normal, out,
    )
    assert np.isclose(out[0], 0.009552841750957684)
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_propagate():
    gravity = np.array([0, -9.8, 0])
    in_neutron = np.array([
        0, -0.01, -20,
        0, 4, 8000,
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
    z_max = 1
    R0=0.99; Qc=0.0219; alpha=3; m=2; W=0.003
    faces2d = np.array([[
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1],
    ]])
    bounds2d = np.array([[
        [-1, -1],
        [1, 1],
    ]])
    tmp1 = np.zeros(3)
    tmp2 = np.zeros(3)
    tmp3 = np.zeros(3)
    tmp_neutron = np.zeros(10)
    nfaces = len(faces)
    intersections = np.zeros(nfaces, dtype=np.float64)
    face_indexes = np.zeros(nfaces, dtype=np.int)
    guide_anyshape_gravity._propagate(
        in_neutron,
        faces, centers, unitvecs, faces2d, bounds2d,
        z_max, R0, Qc, alpha, m, W,
        intersections, face_indexes,
        gravity,
        tmp1, tmp2, tmp3, tmp_neutron,
    )
    expected = np.array([
        0., -4.6637077E-4, 1,
        0., -3.97542451-9.8*(1-6.16280534E-2)/8000, 8000,
        0, 0,
        21/8000, 0.99
    ])
    assert np.allclose(in_neutron, expected)
    return
