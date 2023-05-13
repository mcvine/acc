#!/usr/bin/env python

import pytest, os, time
import math, numpy as np
from numba import cuda

from instrument.nixml import parse_file
from mcvine.acc import test
from mcvine.acc.geometry import locate, location, arrow_intersect
from mcvine.acc.geometry.composite_3 import (
    createRayTracingMethods_NonOverlappingShapes as createRTMethods,
    createUnionLocateMethod
)
from mcvine.acc.geometry.composite import _make_find_1st_hit

thisdir = os.path.dirname(__file__)

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_locate():
    union = parse_file(os.path.join(thisdir, 'union_three_elements.xml'))[0]
    shapes = union.shapes
    locate_u3 =  createUnionLocateMethod(shapes)
    assert locate_u3(0, 0, 0) == locate.inside
    assert locate_u3(0.025, 0, 0) == locate.onborder
    assert locate_u3(0.099, 0, 0) == locate.onborder
    assert locate_u3(0.1, 0, 0) == locate.onborder
    assert locate_u3(0.199, 0, 0) == locate.onborder
    assert locate_u3(0.2, 0, 0) == locate.onborder
    assert locate_u3(0.3, 0, 0) == locate.outside
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_intersect_all():
    union = parse_file(os.path.join(thisdir, 'union_three_elements.xml'))[0]
    shapes = union.shapes
    methods = createRTMethods(shapes)
    intersect_all = methods['intersect_all']
    ts = np.zeros(11)
    assert intersect_all(0.,0.,0., 1.,0.,0., ts) == 10
    expected = np.array([-0.2,-0.199, -0.1,-0.099, -0.025, 0.025, 0.099,0.1, 0.199,0.2, 0])
    assert np.allclose(ts, expected)
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_forward_intersect_all():
    union = parse_file(os.path.join(thisdir, 'union_three_elements.xml'))[0]
    shapes = union.shapes
    methods = createRTMethods(shapes)
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
    methods = createRTMethods(shapes)
    find_1st_hit = _make_find_1st_hit(**methods)
    assert find_1st_hit(-0.25,0.,0., 1.,0.,0.) == 0
    assert find_1st_hit(-0.15,0.,0., 1.,0.,0.) == 1
    assert find_1st_hit(-0.05,0.,0., 1.,0.,0.) == 2
    assert find_1st_hit(0.05,0.,0., 1.,0.,0.) == 1
    assert find_1st_hit(0.15,0.,0., 1.,0.,0.) == 0
    assert find_1st_hit(0.25,0.,0., 1.,0.,0.) == -1
