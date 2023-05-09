#!/usr/bin/env python

import pytest, os, time
import math, numpy as np
from numba import cuda

from instrument.nixml import parse_file
from mcvine.acc import test
from mcvine.acc.geometry import locate
from mcvine.acc.geometry.composite import makeModule, importModule, make_find_1st_hit

thisdir = os.path.dirname(__file__)

def test_makeModule():
    makeModule(4, overwrite=True)
    return

@pytest.fixture
def ray_tracing_methods():
    union = parse_file(os.path.join(thisdir, 'union_three_elements.xml'))[0]
    shapes = union.shapes
    mod = importModule(len(shapes))
    methods = mod.createMethods(shapes)
    methods['union_locate'] = mod.createUnionLocateMethod(shapes)
    return methods

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_locate(ray_tracing_methods):
    locate_u3 = ray_tracing_methods['union_locate']
    assert locate_u3(0, 0, 0) == locate.inside
    assert locate_u3(0.025, 0, 0) == locate.onborder
    assert locate_u3(0.099, 0, 0) == locate.onborder
    assert locate_u3(0.1, 0, 0) == locate.onborder
    assert locate_u3(0.199, 0, 0) == locate.onborder
    assert locate_u3(0.2, 0, 0) == locate.onborder
    assert locate_u3(0.3, 0, 0) == locate.outside
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_intersect_all(ray_tracing_methods):
    intersect_all = ray_tracing_methods['intersect_all']
    ts = np.zeros(11)
    assert intersect_all(0.,0.,0., 1.,0.,0., ts) == 10
    expected = np.array([-0.2,-0.199, -0.1,-0.099, -0.025, 0.025, 0.099,0.1, 0.199,0.2, 0])
    assert np.allclose(ts, expected)
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_forward_intersect_all(ray_tracing_methods):
    forward_intersect_all = ray_tracing_methods['forward_intersect_all']
    ts = np.zeros(6)
    assert forward_intersect_all(0.,0.,0., 1.,0.,0., ts) == 5
    expected = np.array([0.025,0.099,0.1, 0.199,0.2, 0])
    assert np.allclose(ts, expected)
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_find_1st_hit(ray_tracing_methods):
    find_1st_hit = make_find_1st_hit(**ray_tracing_methods)
    assert find_1st_hit(-0.25,0.,0., 1.,0.,0.) == 0
    assert find_1st_hit(-0.15,0.,0., 1.,0.,0.) == 1
    assert find_1st_hit(-0.05,0.,0., 1.,0.,0.) == 2
    assert find_1st_hit(0.05,0.,0., 1.,0.,0.) == 1
    assert find_1st_hit(0.15,0.,0., 1.,0.,0.) == 0
    assert find_1st_hit(0.25,0.,0., 1.,0.,0.) == -1
