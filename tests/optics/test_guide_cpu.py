#!/usr/bin/env python

import os, shutil
import histogram.hdf as hh
import numpy as np
import pytest
thisdir = os.path.dirname(__file__)
from mcvine import run_script
interactive = False

def test():
    '''
    Tests the acc cpu implementation of a straight guide against mcvine
    '''
    num_neutrons = 100000
    # Run the mcvine instrument first
    mcvine_instr = os.path.join(thisdir, "mcvine_guide_cpu_instrument.py")
    mcvine_outdir = 'out.debug-mcvine_guide_cpu_instrument'
    if os.path.exists(mcvine_outdir): shutil.rmtree(mcvine_outdir)
    run_script.run1(mcvine_instr, mcvine_outdir, ncount=num_neutrons, overwrite_datafiles=True)

    # Run our guide implementation
    instr = os.path.join(thisdir, "guide_cpu_instrument.py")
    outdir = 'out.debug-guide_cpu_instrument'
    if os.path.exists(outdir): shutil.rmtree(outdir)
    run_script.run1(instr, outdir, ncount=num_neutrons, overwrite_datafiles=True)

    # Compare output files
    mcvine_Ixy = hh.load(os.path.join(mcvine_outdir, "Ixy.h5"))
    mcvine_Ixdivx = hh.load(os.path.join(mcvine_outdir, "Ixdivx.h5"))
    Ixy = hh.load(os.path.join(outdir, "Ixy.h5"))
    Ixdivx = hh.load(os.path.join(outdir, "Ixdivx.h5"))

    global interactive
    if interactive:
        from histogram import plot as plotHist
        plotHist(mcvine_Ixy)
        plotHist(mcvine_Ixdivx)
        plotHist(Ixy)
        plotHist(Ixdivx)
    assert mcvine_Ixy.shape() == Ixy.shape()
    assert mcvine_Ixdivx.shape() == Ixdivx.shape()
    assert np.allclose(mcvine_Ixy.data().storage(), Ixy.data().storage())
    assert np.allclose(mcvine_Ixdivx.data().storage(), Ixdivx.data().storage())
    return


def assert_approximately_equal(expected, actual):
    """
    Assert that the given arguments are numerically very close.
    Arguments may be scalar or iterable.
    Throws AssertionError if the arguments are dissimilar.

    Parameters:
    expected: a number or numbers
    actual: a number or numbers
    """

    from collections.abc import Iterable

    if isinstance(expected, Iterable):
        assert np.allclose(expected, actual)
    else:
        assert_approximately_equal([expected], [actual])


@pytest.mark.parametrize("position_x", np.arange(-0.5, 1, 0.5))
@pytest.mark.parametrize("velocity_z", np.arange(3, 4.5, 0.5))
def test_expected_exits(position_x, velocity_z):
    """
    Check that neutrons exit the guide at the expected angle, etc.
    """
    import math
    from mcni import neutron
    from mcvine.acc.components.guide import do_process, Guide

    # set up simple guide
    guide_length = 16
    guide = Guide('test guide', 3, 3, 2, 2, guide_length)
    guide_angle = math.atan(((3 - 2) / 2) / guide_length)

    # set up particle
    position = np.array([position_x, -1, 0], dtype=float)
    velocity = np.array([0, 1, velocity_z], dtype=float)

    # determine expected angle from z-axis of exit assuming two reflections
    angle_expected = math.atan(velocity[1] / velocity[2])
    for i in range(0, 2):
        offset_from_normal = math.pi/2 - angle_expected - guide_angle
        angle_expected = math.pi - angle_expected - 2 * offset_from_normal

    # propagate particle through guide
    result = do_process(guide, [neutron(position, velocity)])
    (position, velocity, duration) = (result[0:3], result[3:6], result[8])
    angle_actual = math.atan(velocity[1] / velocity[2])
    path_length = duration * np.linalg.norm(velocity)

    # check outcome
    assert_approximately_equal(position_x, position[0])
    assert_approximately_equal(guide_length, position[2])
    assert_approximately_equal(angle_expected, angle_actual)

    assert path_length > guide_length
    assert path_length < guide_length * 1.1


def test_miss_guide():
    """
    Checks several cases where neutrons should miss the guide
    """
    from mcni import neutron
    from mcvine.acc.components.guide import do_process, Guide

    guide = Guide('test guide', 3, 3, 2, 2, 16)

    # neutron moving away from guide
    r = do_process(guide, neutron(r=(1.0, 1.0, 0.0), v=(-1.0, -1.0, -5.0)))
    assert len(r) == 0


def main():
    global interactive
    interactive = True
    test()
    return

if __name__ == '__main__': main()
