#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, math, numpy as np, time
from numba import cuda
from mcvine.acc.geometry import locate, location, arrow_intersect

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_insert_into_sorted_list():
    l = np.arange(10.)
    d = 2.5
    N = arrow_intersect.insert_into_sorted_list(d, l, 0)
    assert N==1
    np.testing.assert_array_equal(l[:N], [2.5])

    l = np.arange(10.)
    d = 2.5
    N = arrow_intersect.insert_into_sorted_list(d, l, 1)
    assert N==2
    np.testing.assert_array_equal(l[:N], [0, 2.5])

    l = np.arange(10.)
    d = 2.5
    N = arrow_intersect.insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 1., 2., 2.5, 3., 4.])

    l = np.arange(10.)
    d = 2
    N = arrow_intersect.insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 1., 2., 2., 3., 4.])

    l = np.arange(10.)
    d = 0
    N = arrow_intersect.insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 0., 1., 2., 3., 4.])

    l = np.arange(10.)
    d = -1
    N = arrow_intersect.insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [-1, 0., 1., 2., 3., 4.])

    l = np.arange(10.)
    d = 4
    N = arrow_intersect.insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 1., 2., 3., 4., 4.])

    l = np.arange(10.)
    d = 8
    N = arrow_intersect.insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 1., 2., 3., 4., 8.])

    return

def main():
    test_insert_into_sorted_list()
    return

if __name__ == '__main__': main()
