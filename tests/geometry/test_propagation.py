#!/usr/bin/env python

import pytest, os, math
from mcvine.acc import test

import os, numpy as np, time
from numba import cuda
from mcvine.acc.geometry.propagation import makePropagateMethods
from mcvine.acc.geometry import arrow_intersect
from instrument.nixml import parse_file

thisdir = os.path.dirname(__file__)

def getPropagateMethods():
    parsed = parse_file(os.path.join(thisdir, 'hollowcylinder_example1.xml'))
    shape = parsed[0]
    # print(shape)
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)
    return makePropagateMethods(intersect, locate)

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_propagate_out():
    propagate_out = getPropagateMethods()['propagate_out']
    neutrons = np.array([
        [-0.025,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.015,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.015,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.025,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.,0.,0., 0.,0.,1., 0.,0., 0., 1.],
    ])
    expected = np.array([
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.045, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.04, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.035, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.03, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.02, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.01, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.005, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.025,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.,0.,0., 0.,0.,1., 0.,0., 0., 1.],
    ])
    for neutron, out in zip(neutrons, expected):
        propagate_out(neutron)
        assert np.allclose(neutron, out), f"{neutron} != {out}"
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_tof_before_exit():
    tof_before_exit = getPropagateMethods()['tof_before_exit']
    neutrons = np.array([
        [-0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.015,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.015,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
    ])
    expected = (0.01, 0.005, 0., 0.01, 0.005, 0.)
    for neutron, out in zip(neutrons, expected):
        t = tof_before_exit(neutron)
        assert np.allclose(t, out), f"{t} != {out}"
    neutrons = np.array([
        [-0.025,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.025,0.,0., 1.,0.,0., 0.,0., 0., 1.],
    ])
    for neutron in neutrons:
        with pytest.raises(RuntimeError):
            tof_before_exit(neutron)
    return


@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_propagate_to_next_incident_surface():
    propagate_to_next_incident_surface = getPropagateMethods()[
        'propagate_to_next_incident_surface']
    neutrons = np.array([
        [-1.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [1.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
    ])
    expected = np.array([
        [-0.02,0.,0., 1.,0.,0., 0.,0., 0.98, 1.],
        [-0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.01,0.,0., 1.,0.,0., 0.,0., 0.02, 1.],
        [0.01,0.,0., 1.,0.,0., 0.,0., 0.01, 1.],
        [0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [1.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
    ])
    for neutron, out in zip(neutrons, expected):
        propagate_to_next_incident_surface(neutron)
        assert np.allclose(neutron, out), f"{neutron} != {out}"
    neutrons = np.array([
        [-0.015,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.015,0.,0., 1.,0.,0., 0.,0., 0., 1.],
    ])
    for neutron in neutrons:
        with pytest.raises(RuntimeError):
            propagate_to_next_incident_surface(neutron)
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_propagate_to_next_exiting_surface():
    propagate_to_next_exiting_surface = getPropagateMethods()[
        'propagate_to_next_exiting_surface']
    neutrons = np.array([
        [-1.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.025,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.015,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.01,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.015,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [0.025,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [1.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
     ])
    expected = np.array([
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0.99, 1.],
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0.015, 1.],
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0.01, 1.],
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0.005, 1.],
        [-0.01,0.,0., 1.,0.,0., 0.,0., 0.01, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.02, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.01, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.005, 1.],
        [0.02,0.,0., 1.,0.,0., 0.,0., 0.0, 1.],
        [0.025,0.,0., 1.,0.,0., 0.,0., 0., 1.],
        [1.,0.,0., 1.,0.,0., 0.,0., 0., 1.],
    ])
    return

def main():
    test_propagate_out()
    test_tof_before_exit()
    test_propagate_to_next_incident_surface()
    test_propagate_to_next_exiting_surface()
    return

if __name__ == '__main__': main()
