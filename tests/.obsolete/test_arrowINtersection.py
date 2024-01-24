#!/usr/bin/env python

import pytest, os
pytest.skip("obsolete", allow_module_level=True)
from mcvine.acc import test
if not test.USE_CUDA:
    pytest.skip("No CUDA", allow_module_level=True)

from mcvine.acc.arrowINtersection import ArrowIntersector
# from numpy_arrow_box import ArrowIntersector
import numpy as np
from collections import namedtuple

def test_arrowIntersector():
    # box = box_type(edge_X=1, edge_Y=2, edge_Z=3)
    # position = vector_3D(x=0, y=0, z=-5)
    # direction = vector_3D(x=0, y=0, z=1)
    # arrow = [arrow_type(position = position, direction = direction)]*10000

    arrow_array_position = np.array([[0.,0.,-5.]]*10000)
    # arrow_array_position = np.array([[-1, 0, 1.5]]) #given by andrei
    # arrow_array_position = np.array([[0, 0, -5]])
    arrow_array_direction = np.array([[0.,0.,1.]]*10000)
    # arrow_array_direction = np.array([[1, 0, 0]]) #given by andrei
    # arrow_array_direction = np.array([[0, 0, 1]])
    # box = np.array([1,1,3]) #given by andrei
    box = np.array([1., 2., 3.])
    # print (arrow_array_position[0][0])


    time = ArrowIntersector(arrow_array_position,arrow_array_direction,box)
    assert(len(time) == 10000)
    assert(len(time[20])==2)
    assert (time[0][0] == 3.5)
    assert (time[0][1] == 6.5)

def main():
    test_arrowIntersector()
    return

if __name__ == '__main__': main()
