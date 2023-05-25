#!/usr/bin/env python

import os, numpy as np, pytest
thisdir = os.path.abspath(os.path.dirname(__file__))
import mcvine
from mcvine.acc import test
from mcvine.acc.components.optics.arm import Arm
from mcvine.acc.run_script import calcTransformations
from mcvine.acc.neutron import abs2rel

def createInstrument():
    instrument = mcvine.instrument()
    comp1 = Arm('comp1')
    instrument.append(comp1, position=(0,0,0.))
    comp2 = Arm('comp2')
    instrument.append(comp2, position=(0,0,1.))
    comp3 = Arm('comp3')
    instrument.append(comp3, position=(0,0,0.), orientation=(0, 90, 0), relativeTo=comp2)
    comp4 = Arm('comp4')
    instrument.append(comp4, position=(0,0,1.), orientation=(0, 0, 90), relativeTo=comp2)
    return instrument

@pytest.mark.skipif(not test.USE_CUDASIM, reason='No CUDASIM')
def test():
    instrument = createInstrument()
    offsets, rotmats = calcTransformations(instrument)
    tmp_position = np.array([0., 0., 0.])
    tmp_velocity = np.array([0., 0., 0.])
    # translation only
    position = np.array([0., 0., 0.])
    velocity = np.array([0., 0., 1.])
    abs2rel(position, velocity, rotmats[0], offsets[0], tmp_position, tmp_velocity)
    assert np.allclose(position, [0,0,-1])
    assert np.allclose(velocity, [0,0,1])
    # rotation only
    position = np.array([0., 0., -1.])
    velocity = np.array([0., 0., 1.])
    abs2rel(position, velocity, rotmats[1], offsets[1], tmp_position, tmp_velocity)
    assert np.allclose(position, [1,0,0])
    assert np.allclose(velocity, [-1,0,0])
    # translation and rotation
    position = np.array([0., 0., 0])
    velocity = np.array([0., 0., 1.])
    abs2rel(position, velocity, rotmats[2], offsets[2], tmp_position, tmp_velocity)
    assert np.allclose(position, [0,0,-1])
    assert np.allclose(velocity, [0,-1,0])
    return
