#!/usr/bin/env python

import pytest, os, time
import math, numpy as np
from numba import cuda

from instrument.nixml import parse_file
from mcvine.acc import test
from mcvine.acc.geometry import locate, location, arrow_intersect
from mcvine.acc.geometry.composite import createMethods_3, make_find_1st_hit

thisdir = os.path.dirname(__file__)

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_forward_intersect_all():
    union = parse_file(os.path.join(thisdir, 'union_three_elements.xml'))[0]
    shapes = union.shapes
    methods = createMethods_3(shapes)
    forward_intersect_all = methods['forward_intersect_all']
    ts = np.zeros(6)
    assert forward_intersect_all(0.,0.,0., 1.,0.,0., ts) == 5
    expected = np.array([0.025,0.099,0.1, 0.199,0.2, 0])
    assert np.allclose(ts, expected)
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_find_1st_hit():
    union = parse_file(os.path.join(thisdir, 'union_three_elements.xml'))[0]
    shapes = union.shapes
    methods = createMethods_3(shapes)
    find_1st_hit = make_find_1st_hit(**methods)
    assert find_1st_hit(-0.25,0.,0., 1.,0.,0.) == 0
    assert find_1st_hit(-0.15,0.,0., 1.,0.,0.) == 1
    assert find_1st_hit(-0.05,0.,0., 1.,0.,0.) == 2
    assert find_1st_hit(0.05,0.,0., 1.,0.,0.) == 1
    assert find_1st_hit(0.15,0.,0., 1.,0.,0.) == 0
    assert find_1st_hit(0.25,0.,0., 1.,0.,0.) == -1
