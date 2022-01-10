#!/usr/bin/env python

import os
import histogram.hdf as hh
import numpy as np
import pytest
import shutil
from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcvine import run_script
from mcvine.acc import test
from mcvine.acc.components.optics.guide import Guide
from mcvine.acc.geometry.plane import Plane


thisdir = os.path.dirname(__file__)
interactive = False


class TestReflectivity:

    @pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
    def test_velocity(self):
        """
        Check that the reflectivity is calculated correctly for a reflection.
        """
        from mcni.utils.conversion import v2k

        R0 = 0.99
        Qc = 0.0219  # Å-1
        alpha = 6.07  # Å
        m = 2
        W = 0.003  # Å-1

        guide = Guide('test guide', 3, 3, 2, 2, 16)
        side = guide.sides[2]
        plane = Plane(side[0], side[1])

        speed = 400  # m/s
        arbitrary_vector = np.array([1, 1, 1], dtype=float)
        v_i = np.cross(plane.state[1], arbitrary_vector)
        v_i += plane.state[1] * speed / 1e4
        v_i *= speed / np.linalg.norm(v_i)
        v_f = plane.reflect(v_i)
        assert np.isclose(speed, np.linalg.norm(v_f))

        # check that reflection is at a shallow angle
        (v_i_hat, v_f_hat) = map(lambda v: v / np.linalg.norm(v), (v_i, v_f))
        v_dot = np.dot(v_i_hat, v_f_hat)
        assert 0.99 < v_dot < 1

        (k_i, k_f) = map(v2k, (v_i, v_f))  # Å-1
        Q = np.linalg.norm(k_i - k_f)  # Å-1

        actual = guide.reflectivity(v_i, v_f)

        if Q > Qc:
            p_l = 1 - np.tanh((Q - m * Qc) / W)
            p_r = 1 - alpha * (Q - Qc)
            assert np.isclose(actual, R0 * p_l * p_r / 2)
        else:
            assert np.isclose(actual, R0)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine():
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    num_neutrons = 100000
    # Run the mcvine instrument first
    mcvine_instr = os.path.join(thisdir, "mcvine_guide_cpu_instrument.py")
    mcvine_outdir = 'out.debug-mcvine_guide_cpu_instrument'
    if os.path.exists(mcvine_outdir):
        shutil.rmtree(mcvine_outdir)
    run_script.run1(mcvine_instr, mcvine_outdir, ncount=num_neutrons,
                    overwrite_datafiles=True)

    # Run our guide implementation
    instr = os.path.join(thisdir, "guide_gpu_instrument.py")
    outdir = 'out.debug-guide_gpu_instrument'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    run_script.run1(instr, outdir, ncount=num_neutrons,
                    overwrite_datafiles=True)

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


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("position_x", np.arange(-0.5, 1, 0.5))
@pytest.mark.parametrize("velocity_z", np.arange(3, 4.5, 0.5))
@pytest.mark.parametrize("use_propagate", [False, True])
def test_expected_exits(position_x, velocity_z, use_propagate):
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
    if use_propagate:
        (position, velocity, duration, weight) = guide.propagate(
            tuple(position), tuple(velocity), 0, 1)
    else:
        result = do_process(guide, [neutron(position, velocity)])
        (position, velocity, duration) = (result[0:3], result[3:6], result[8])
    angle_actual = math.atan(velocity[1] / velocity[2])
    path_length = duration * np.linalg.norm(velocity)

    # check outcome
    assert np.isclose(position_x, position[0])
    assert np.isclose(guide_length, position[2])
    assert np.isclose(angle_expected, angle_actual)

    assert guide_length < path_length < guide_length * 1.1


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_last_position_velocity():
    """
    Check that neutrons' position and velocity is as they exit.
    Checks: misses guide, enters and does not reflect, enters and reflects.
    """
    import math

    # set up simple guide
    # narrower exit to guarantee some reflections from level paths
    guide_length = 10
    guide_entry_x = 5
    guide_exit_x = 4
    guide = Guide(
        'test guide', guide_entry_x * 2, 3, guide_exit_x * 2, 3, guide_length)

    slope = (guide_entry_x - guide_exit_x) / guide_length
    deflection_angle = 2 * math.atan(slope)

    cases = set()

    velocity_initial = np.array([0, 0, 5], dtype=float)
    speed = np.linalg.norm(velocity_initial)

    neutrons = []
    expected = []
    for position_x in np.arange(3.125, 6, 0.25):

        # check that edge cases are avoided
        assert not np.isclose(guide_entry_x, position_x)
        assert not np.isclose(guide_exit_x, position_x)

        # set up particle
        position_initial = np.array([position_x, -1, 0], dtype=float)

        neutrons.append(neutron(position_initial, velocity_initial))

        # predict outcome
        if position_x > guide_entry_x:
            # misses altogether
            cases.add('misses')
        else:
            if position_x < guide_exit_x:
                # passes straight through
                cases.add('direct')

                position_final = position_initial.copy()
                position_final[2] = guide_length
                velocity_final = velocity_initial
                duration = guide_length / speed
            else:
                # reflects off the side
                cases.add('indirect')

                # calculate geometry of reflection
                unimpeded_length = (guide_entry_x - position_x) / slope
                remaining_length = guide_length - unimpeded_length
                sideways_deflection = \
                    remaining_length * math.tan(deflection_angle)
                distance_traveled = unimpeded_length + math.sqrt(
                    remaining_length ** 2 + sideways_deflection ** 2)

                # set up expected outcome
                position_final = np.array([
                    position_x - sideways_deflection,
                    position_initial[1],
                    guide_length
                ], dtype=float)
                velocity_final = np.array([
                    speed * math.sin(-deflection_angle),
                    0,
                    speed * math.cos(deflection_angle)
                ], dtype=float)
                duration = distance_traveled / speed

            assert abs(position_final[0]) < guide_exit_x
            expected.append((position_final, velocity_final, duration))

    # check that all kinds of cases are covered
    assert len(cases) == 3

    # propagate particles through guide
    result = do_process(guide, neutrons[0:9])

    expected = np.array(
        [list(position) + list(velocity) + [duration]
            for (position, velocity, duration) in expected]
    )
    actual = result.T[[0, 1, 2, 3, 4, 5, 8]].T

    assert expected.shape == actual.shape
    assert np.allclose(expected, actual)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_miss_guide():
    """
    Checks several cases where neutrons should miss the guide
    """
    guide = Guide('test guide', 3, 2, 2, 2, 16)

    # neutron moving away from guide should miss entirely
    r = do_process(guide, neutron(r=(1.0, 1.0, 0.0), v=(-1.0, -1.0, -5.0)))
    assert len(r) == 0

    # neutron moving toward guide but just misses entrance
    angle = np.arctan((0.5 * 3.0) / 5.0)
    r = do_process(guide,
                   neutron(r=(0.0, 0.0, -5.0), v=(
                       0.0, 0.5 * np.sin(angle) + 0.05, 0.5 * np.cos(angle))))
    assert len(r) == 0

    # check for guide height and width being switched
    r = do_process(guide, neutron(r=(1.25, 0.75, -1), v=(0.0, 0.0, 1.0)))
    assert len(r) > 0
    r = do_process(guide, neutron(r=(0.75, 1.25, -1), v=(0.0, 0.0, 1.0)))
    assert len(r) == 0


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
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


if __name__ == '__main__':
    main()
