#!/usr/bin/env python

import os, numpy as np, pytest
thisdir = os.path.abspath(os.path.dirname(__file__))
import mcvine
from mcvine.acc import test
from mcvine.acc.components.optics.arm import Arm
from mcvine.acc.run_script import calcTransformations
from mcvine.acc.neutron import applyTransformation

def createTestInstrument():
    instrument = mcvine.instrument()
    comp1 = Arm('comp1')
    instrument.append(comp1, position=(0,0,0.))
    comp2 = Arm('comp2')
    instrument.append(comp2, position=(0,0,1.))
    comp3 = Arm('comp3')
    instrument.append(comp3, position=(0,0,0.), orientation=(0, 90, 0), relativeTo=comp2)
    comp4 = Arm('comp4')
    instrument.append(comp4, position=(0,0,1.), orientation=(0, 0, 90), relativeTo=comp2)
    comp5 = Arm('comp5')
    instrument.append(comp5, position=(1,0,0.), orientation=(90, 0, 0), relativeTo=comp4)
    return instrument

@pytest.mark.skipif(not test.USE_CUDASIM, reason='No CUDASIM')
def test():
    instrument = createTestInstrument()
    offsets, rotmats = calcTransformations(instrument)
    assert len(rotmats) == len(offsets) == len(instrument.components)-1
    # tmp arrays required by applyTransformation
    tmp_position = np.array([0., 0., 0.])
    tmp_velocity = np.array([0., 0., 0.])
    # translation only between comp1 and comp2
    position = np.array([0., 0., 0.])
    velocity = np.array([0., 0., 1.])
    applyTransformation(position, velocity, rotmats[0], offsets[0], tmp_position, tmp_velocity)
    assert np.allclose(position, [0,0,-1])
    assert np.allclose(velocity, [0,0,1])
    # rotation only between comp2 and comp3
    position = np.array([0., 0., -1.])
    velocity = np.array([0., 0., 1.])
    applyTransformation(position, velocity, rotmats[1], offsets[1], tmp_position, tmp_velocity)
    assert np.allclose(position, [1,0,0])
    assert np.allclose(velocity, [-1,0,0])
    # translation and rotation between comp3 and comp4 (indirectly thru comp2)
    position = np.array([0., 0., 0])
    velocity = np.array([0., 0., 1.])
    applyTransformation(position, velocity, rotmats[2], offsets[2], tmp_position, tmp_velocity)
    assert np.allclose(position, [0,0,-1])
    assert np.allclose(velocity, [0,-1,0])
    # translation and rotation between comp4 and comp5
    position = np.array([0., 1., 0])
    velocity = np.array([0., 0., 1.])
    applyTransformation(position, velocity, rotmats[3], offsets[3], tmp_position, tmp_velocity)
    assert np.allclose(position, [-1,0,-1])
    assert np.allclose(velocity, [0,1,0])
    return
