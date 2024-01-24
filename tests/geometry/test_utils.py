#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, math, numpy as np, time
from numba import cuda
from mcvine.acc.geometry._utils import insert_into_sorted_list, insert_into_sorted_list_with_indexes

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_insert_into_sorted_list():
    l = np.arange(10.)
    d = 2.5
    N = insert_into_sorted_list(d, l, 0)
    assert N==1
    np.testing.assert_array_equal(l[:N], [2.5])

    l = np.arange(10.)
    d = 2.5
    N = insert_into_sorted_list(d, l, 1)
    assert N==2
    np.testing.assert_array_equal(l[:N], [0, 2.5])

    l = np.arange(10.)
    d = 2.5
    N = insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 1., 2., 2.5, 3., 4.])

    l = np.arange(10.)
    d = 2
    N = insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 1., 2., 2., 3., 4.])

    l = np.arange(10.)
    d = 0
    N = insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 0., 1., 2., 3., 4.])

    l = np.arange(10.)
    d = -1
    N = insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [-1, 0., 1., 2., 3., 4.])

    l = np.arange(10.)
    d = 4
    N = insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 1., 2., 3., 4., 4.])

    l = np.arange(10.)
    d = 8
    N = insert_into_sorted_list(d, l, 5)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0., 1., 2., 3., 4., 8.])

    return

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_insert_into_sorted_list_with_indexes():
    # 2.5 6 1 3 0 8
    l = np.zeros(6)
    index_list = np.zeros(6, dtype=int)
    #
    N = 0
    N = insert_into_sorted_list_with_indexes(0, 2.5, index_list, l, N)
    assert N==1
    np.testing.assert_array_equal(l[:N], [2.5])
    np.testing.assert_array_equal(index_list[:N], [0])
    #
    N = insert_into_sorted_list_with_indexes(1, 6, index_list, l, N)
    assert N==2
    np.testing.assert_array_equal(l[:N], [2.5, 6])
    np.testing.assert_array_equal(index_list[:N], [0, 1])
    #
    N = insert_into_sorted_list_with_indexes(2, 1, index_list, l, N)
    assert N==3
    np.testing.assert_array_equal(l[:N], [1, 2.5, 6])
    np.testing.assert_array_equal(index_list[:N], [2, 0, 1])
    #
    N = insert_into_sorted_list_with_indexes(3, 3, index_list, l, N)
    assert N==4
    np.testing.assert_array_equal(l[:N], [1, 2.5, 3, 6])
    np.testing.assert_array_equal(index_list[:N], [2, 0, 3, 1])
    #
    N = insert_into_sorted_list_with_indexes(4, 0, index_list, l, N)
    assert N==5
    np.testing.assert_array_equal(l[:N], [0, 1, 2.5, 3, 6])
    np.testing.assert_array_equal(index_list[:N], [4, 2, 0, 3, 1])
    #
    N = insert_into_sorted_list_with_indexes(5, 8, index_list, l, N)
    assert N==6
    np.testing.assert_array_equal(l[:N], [0, 1, 2.5, 3, 6, 8])
    np.testing.assert_array_equal(index_list[:N], [4, 2, 0, 3, 1, 5])
    return

def main():
    test_insert_into_sorted_list()
    return

if __name__ == '__main__': main()
