#!/usr/bin/env python

import pytest, os, math
from mcvine.acc import test

import os, numpy as np, time
from numba import cuda
from mcvine.acc.geometry.propagation import makePropagateMethods
from mcvine.acc.geometry import arrow_intersect
from instrument.nixml import parse_file

thisdir = os.path.dirname(__file__)

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_propagate_out():
    parsed = parse_file(os.path.join(thisdir, 'hollowcylinder_example1.xml'))
    shape = parsed[0]
    print(shape)
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)
    forward_intersect, propagate_out, propagate_to_next_incident_surface, propagate_to_next_exiting_surface = \
        makePropagateMethods(intersect, locate)
    neutron = np.array([0.,0.,0., 1.,0.,0., 0.,0., 0., 1.])
    propagate_out(neutron)
    print(neutron)
    return

def main():
    test_propagate_out()
    return

if __name__ == '__main__': main()
