#!/usr/bin/env python

import os, shutil
import histogram.hdf as hh
import numpy as np
import pytest
thisdir = os.path.dirname(__file__)
from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcvine import run_script
from mcvine.acc.components.guide import Guide
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

    def helper(xy):
        (x, y) = xy
        assert abs(x - y) < 1e-10

    from collections.abc import Iterable

    if isinstance(expected, Iterable):
        map(helper, zip(expected, actual))
    else:
        helper((expected, actual))


def do_process(guide, neutrons):
    """
    Testing helper function to run a neutron through the guide and
    return the result as a numpy array

    Parameters
    ----------
    guide : instance of a Guide
    neutrons : a NeutronEvent or a list of NeutronEvents

    Returns
    -------
    Numpy array containing: [x, y, z, vx, vy, vz, s1, s2, t, p] for each
    input in neutrons
    """
    from mcni.mcnibp import NeutronEvent

    assert isinstance(guide, Guide)
    buffer = neutron_buffer(1)
    if isinstance(neutrons, list):
        buffer.resize(len(neutrons), neutron())
        for i in range(len(neutrons)):
            buffer[i] = neutrons[i]
    elif isinstance(neutrons, NeutronEvent):
        buffer[0] = neutrons
    else:
        raise RuntimeError(
            "Expected a NeutronEvent or a list of NeutronEvents")

    guide.process(buffer)
    result = neutrons_as_npyarr(buffer)
    result.shape = -1, ndblsperneutron
    if result.shape[0] == 1:
        # return only a single dim array to make test comparisons easier
        result = result[0]
    return result


@pytest.mark.parametrize("position_x", np.arange(-0.5, 1, 0.5))
@pytest.mark.parametrize("velocity_z", np.arange(3, 4.5, 0.5))
def test_expected_exits(position_x, velocity_z):
    """
    Check that neutrons exit the guide at the expected angle, etc.
    """
    import math

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
    guide = Guide('test guide', 3, 3, 2, 2, 16)

    # neutron moving away from guide should miss entirely
    r = do_process(guide, neutron(r=(1.0, 1.0, 0.0), v=(-1.0, -1.0, -5.0)))
    assert len(r) == 0

    # neutron moving toward guide but just misses entrance
    angle = np.arctan((0.5 * 3.0) / 5.0)
    r = do_process(guide,
                   neutron(r=(0.0, 0.0, -5.0), v=(
                       0.0, 0.5 * np.sin(angle) + 0.05, 0.5 * np.cos(angle))))
    assert len(r) == 0


def test_pass_through_guide():
    """
    Test several cases where neutrons pass through the guide
    """
    guide_length = 16
    guide_exit_height = 2.0
    guide = Guide('test guide', 3, 3, 2, guide_exit_height, guide_length)

    # neutron straight through at origin
    result = do_process(guide,
                        neutron(r=(0., 0., 0.), v=(0.0, 0.0, 0.5)))
    # the neutron should be at the same place propagated along the +Z axis
    np.testing.assert_equal([0., 0., guide_length], result[0:3])
    # make sure the velocity does not change
    np.testing.assert_equal([0., 0., 0.5], result[3:6])
    # the time should be the same as t = d/v
    np.testing.assert_equal(guide_length / 0.5, result[8])
    # probability shouldn't change since there are no reflections
    np.testing.assert_equal(1.0, result[9])

    # neutron angled at the upper back guide exit from the origin
    vmag = 0.5
    angle = np.arctan((0.5 * guide_exit_height) / guide_length)
    result = do_process(guide,
                        neutron(r=(0., 0., 0.), v=(
                            0.0, vmag * np.sin(angle), vmag * np.cos(angle))))
    np.testing.assert_equal([0.0, 0.5 * guide_exit_height, guide_length],
                            result[0:3])
    np.testing.assert_equal(1.0, result[9])


def main():
    global interactive
    interactive = True
    test()
    return

if __name__ == '__main__': main()
