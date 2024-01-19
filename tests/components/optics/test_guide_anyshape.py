#!/usr/bin/env python

import os
import numpy as np
import pytest

from mcvine.acc.components.optics import guide_anyshape
from mcvine.acc import test

thisdir = os.path.dirname(__file__)

def test_load_scaled_centered_faces():
    path = './data/guide_anyshape_example1.off'
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
