#!/usr/bin/env python

import os
import numpy as np
import pytest

from mcvine.acc.components.optics import guide_anyshape
from mcvine.acc import test

thisdir = os.path.dirname(__file__)

def test_load_scaled_centered_faces():
    path = os.path.join(thisdir, './data/guide_anyshape_example1.off')
    faces = guide_anyshape.load_scaled_centered_faces(path, xwidth=0, yheight=0, zdepth=0, center=False)
    original = np.array([
        [[-0.01,-0.01, 1.  ],
         [-0.01, 0.01, 1.  ],
         [-0.01, 0.01, 2.  ],
         [-0.01,-0.01, 2.  ]],
        [[-0.01, 0.01, 1.  ],
         [ 0.01, 0.01, 1.  ],
         [ 0.01, 0.01, 2.  ],
         [-0.01, 0.01, 2.  ]],
        [[ 0.01,-0.01, 1.  ],
         [ 0.01,-0.01, 2.  ],
         [ 0.01, 0.01, 2.  ],
         [ 0.01, 0.01, 1.  ]],
        [[-0.01,-0.01, 1.  ],
         [-0.01,-0.01, 2.  ],
         [ 0.01,-0.01, 2.  ],
         [ 0.01,-0.01, 1.  ]]
    ])
    center = np.array([0,0,1.5])
    assert np.allclose(faces, original)
    faces = guide_anyshape.load_scaled_centered_faces(path, xwidth=0.04, yheight=0, zdepth=0, center=False)
    assert np.allclose(faces, (original-center)*2 + center)
    faces = guide_anyshape.load_scaled_centered_faces(path, xwidth=0, yheight=0, zdepth=3, center=False)
    assert np.allclose(faces, (original-center)*3 + center)
    with pytest.raises(ValueError) as excinfo:
        faces = guide_anyshape.load_scaled_centered_faces(path, xwidth=0.04, yheight=0, zdepth=3, center=False)
    faces = guide_anyshape.load_scaled_centered_faces(path, xwidth=0.04, yheight=0.06, zdepth=4, center=False)
    expected = np.array(original) - center
    expected[:, :, 0] *= 2
    expected[:, :, 1] *= 3
    expected[:, :, 2] *= 4
    expected += center
    assert np.allclose(faces, expected)
    faces = guide_anyshape.load_scaled_centered_faces(path, xwidth=0, yheight=0, zdepth=0, center=True)
    assert np.allclose(faces, original-center)
    faces = guide_anyshape.load_scaled_centered_faces(path, xwidth=0.04, yheight=0, zdepth=0, center=True)
    assert np.allclose(faces, (original-center)*2)
    faces = guide_anyshape.load_scaled_centered_faces(path, xwidth=0.04, yheight=0.06, zdepth=4, center=True)
    expected = np.array(original) - center
    expected[:, :, 0] *= 2
    expected[:, :, 1] *= 3
    expected[:, :, 2] *= 4
    assert np.allclose(faces, expected)
    return

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
