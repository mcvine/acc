import os, pytest, numpy as np
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test
from mcvine.acc.components.optics import geometry2d as g2d

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_inside_convex_polygon():
    vertices = [
        [1, 0],
        [-1, 1],
        [-1, -1],
    ]
    assert g2d.inside_convex_polygon([0,0], vertices)
    assert not g2d.inside_convex_polygon([1,1], vertices)
    return
