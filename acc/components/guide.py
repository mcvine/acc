#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

import math
from mcni.AbstractComponent import AbstractComponent
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcni import neutron_buffer, neutron
# import mcvine
# import mcvine.components as mc
import numpy
from mcvine.acc.geometry.plane import Plane

category = 'optics'


class Guide(AbstractComponent):

    @classmethod
    def __new_plane(cls, point_p, point_q, point_r):
        # Represent everything as numpy arrays in preparation for
        # vectorized operations across multi-dimensional arrays.
        return Plane.construct(numpy.array(point_p, dtype=float),
                               numpy.array(point_q, dtype=float),
                               numpy.array(point_r, dtype=float))

    def __init__(
            self, name,
            w1, h1, w2, h2, l,
            # To do: take any notice of these arguments.
            R0=0.99, Qc=0.0219, alpha=6.07, m=2, W=0.003):

        """Initialize this Guide component.
        The guide is centered on the z-axis
        with the entrance at z=0.

        Parameters:
        name (str): the name of this component
        w1 (m): width at the guide entry
        h1 (m): height at the guide entry
        w2 (m): width at the guide exit
        h2 (m): height at the guide exit
        l (m): length of guide
        R0: low-angle reflectivity
        Qc: critical scattering vector
        alpha: slope of reflectivity
        m: m-value of material (0 is complete absorption)
        W: width of supermirror cutoff
        """

        self.name = name
        self.sides = [
            Guide.__new_plane([+w2/2, -h2/2, l],
                              [-w2/2, +h2/2, l],
                              [-w2/2, -h2/2, l]),
            Guide.__new_plane([+w1/2, +h1/2, 0],
                              [+w1/2, -h1/2, 0],
                              [+w2/2, +h2/2, l]),
            Guide.__new_plane([-w1/2, +h1/2, 0],
                              [-w1/2, -h1/2, 0],
                              [-w2/2, +h2/2, l]),
            Guide.__new_plane([+w1/2, +h1/2, 0],
                              [-w1/2, +h1/2, 0],
                              [+w2/2, +h2/2, l]),
            Guide.__new_plane([+w1/2, -h1/2, 0],
                              [-w1/2, -h1/2, 0],
                              [+w2/2, -h2/2, l])
            ]
        self.R0 = R0
        self.Qc = Qc
        self.alpha = alpha
        self.m = m
        self.W = W
        self.m_neutron = 1.67492e-27  # mass of neutron in kg (from mcstas manual)
        self.hbar = 1.05459e-34       # planck constant in Js (from mcstas manual)
        self.m_over_h = 1e-10 * self.m_neutron / self.hbar  # includes A^-1 to m conversion

    def propagate(self, position, velocity):

        """Propagate a particle through this guide.

        Parameters:
        position (vector): x,y,z of the particle's initial position
        velocity (vector): x,y,z of the particle's initial velocity

        Returns a tuple (or None if particle does not exit):
        vector: x,y,z of the particle's exit position
        vector: x,y,z of the particle's exit velocity
        float: the total time that the particle takes to pass through the guide
        """
        (pos_curr, vel_curr) = (position, velocity)
        duration = 0
        ind_prev = None
        while True:
            # Find which side is hit next.
            dur_min = math.inf
            for index in range(0, len(self.sides)):
                if index == ind_prev:
                    continue
                side = self.sides[index]
                pos_curr_wrapped = numpy.array([pos_curr])
                vel_curr_wrapped = numpy.array([vel_curr])
                pos_dur_next_wrapped = side.intersection_duration(pos_curr_wrapped, vel_curr_wrapped)
                if pos_dur_next_wrapped:
                    (pos_next_wrapped, dur_next_wrapped) = pos_dur_next_wrapped
                    (pos_next, dur_next) = (pos_next_wrapped[0], dur_next_wrapped[0])
                    if dur_next > 0 and dur_next < dur_min:
                        ind_min = index
                        (pos_min, dur_min) = (pos_next, dur_next)
            if dur_min == math.inf:
                # No new side was hit.
                return None
            duration += dur_min
            if ind_min == 0:
                # The first side is the exit from the guide.
                return (pos_min, vel_curr, duration)
            # Update particle to be reflecting from the new side.
            pos_curr = pos_min
            vel_curr = self.sides[ind_min].reflect(vel_curr)
            ind_prev = ind_min

    def calc_reflectivity(self, velocity_i, velocity_f):
        """
        Calculates the mirror reflectivity

        Parameters:
        velocity_i: initial velocities before reflecting
        velocity_f: final velocities after reflecting

        Returns
        Reflectivity for each neutron
        """
        # see eq 5.2 in mcstas manual, sec 5.1.1
        Q = self.m_over_h * numpy.linalg.norm(
            velocity_i - velocity_f, axis=1)  # length of scattering vector
        R = numpy.where(Q > self.Qc, 0.5 * self.R0 * (
                1.0 - numpy.tanh((Q - self.m * self.Qc) / self.W) * (
                    1.0 - self.alpha * (Q - self.Qc))),
                        self.R0)
        return R

    # To do: use propagate method to implement process method
    def process(self, neutrons):
        if not len(neutrons):
            return
        arr = neutrons_as_npyarr(neutrons)
        arr.shape = -1, ndblsperneutron

        position = arr[:, 0:3]  # x, y, z
        velocity = arr[:, 3:6]  # vx, vy, vz
        time = arr[:, 8].reshape((arr.shape[0], 1))
        prob = arr[:, 9].reshape((arr.shape[0], 1))

        # initialize arrays containing neutron duration and side index
        side = numpy.full((arr.shape[0], 1), -2, dtype=int)
        old_side = side.copy()  # might not be necessary?
        new_duration = numpy.full((arr.shape[0], 1), numpy.inf)

        # Iterate until all neutrons hit end of guide or are absorbed
        iter = 0
        while numpy.count_nonzero(side == 0) + numpy.count_nonzero(
                side == -1) != len(neutrons):
            duration = numpy.full((arr.shape[0], 1), numpy.inf)
            # Calculate minimum intersection time with all sides of the guide
            for s in range(0, len(self.sides)):
                intersection = self.sides[s].intersection_duration(position, velocity)
                new_duration = numpy.minimum(intersection[1], duration,
                                             where=((intersection[1] > 1e-10) & (s != old_side) & (old_side != 0)),
                                             out=new_duration)

                # Update the index of which side was hit based on new minimum
                side = numpy.where(new_duration != duration, s, side)
                duration = new_duration.copy()

                # If duration is inf, mark the side as invalid
                side = numpy.where(duration == numpy.inf, -1, side)

            # Propagate the neutrons based on the minimum times
            position += numpy.multiply(velocity, duration, where=((duration != numpy.inf) | (old_side != 0)))
            time = numpy.add(time, duration, where=((duration != numpy.inf) | (old_side != 0)))
            old_side = side.copy()

            velocity_before = velocity.copy()

            # Update the velocity due to reflection
            # TODO: vectorize this
            for ind in range(len(neutrons)):
                # Only update the velocity if reflecting on one of the guide sides
                if side[ind] != 0 and side[ind] != -1:
                    velocity[ind] = self.sides[side.item(ind)].reflect(velocity[ind])

            # Calculate reflectivity
            reflectivity = self.calc_reflectivity(velocity_before, velocity)
            prob *= numpy.where(side != 0, reflectivity.reshape(arr.shape[0], 1), prob)
            iter += 1

        print("process took {} iterations".format(iter))
        # Update neutron positions, velocities and times and select those that hit the guide exit
        arr[:, 0:3] = position
        arr[:, 3:6] = velocity
        arr[:, 8] = time.reshape((arr.shape[0],))
        arr[:, 9] = prob.reshape((arr.shape[0],))
        good = arr[(side == 0).flatten(), :]

        neutrons.resize(good.shape[0], neutrons[0])
        neutrons.from_npyarr(good)


def test_process():
    from mcni import neutron_buffer, neutron
    neutrons = neutron_buffer(10)

    # TODO: generate a more random / better set of neutrons for input?
    for i in range(10):
        neutrons[i] = neutron(r=(i * 0.1, 1.0 - i * 0.1, 0.5),
                              v=(0.05 * i, 0.2 * i, 0.5 * i))

    print("Neutrons: (N = {})".format(len(neutrons)))
    # convert to numpy arr for better debug print
    neutrons_arr = neutrons_as_npyarr(neutrons)
    neutrons_arr.shape = -1, ndblsperneutron
    print(neutrons_arr)

    # TODO: use a more complex guide and also test each neutron's outcome
    guide = Guide('guide', 1.0, 1.0, 0.8, 0.8, 10.0)
    guide.process(neutrons)

    print("Neutrons AFTER: (N = {})".format(len(neutrons)))
    result = neutrons_as_npyarr(neutrons)
    result.shape = -1, ndblsperneutron
    print(result)


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
    #print(result)
    return result


def test_guide():
    length = 1.0

    # a square guide to test very simple reflection cases
    sq_guide = Guide('guide', 1.0, 1.0, 1.0, 1.0, length)

    # TODO: test cases for: hitting no mirror, hitting one mirror, hitting two mirrors, etc
    result = do_process(sq_guide,
                        neutron(r=(0.0, 0., 0.), v=(0.5, 0.0, 0.5)))
    numpy.testing.assert_equal([0, 0, length], result[0:3])

    result = do_process(sq_guide,
                        [neutron(r=(0.1, 0, 0), v=(0.1, 0.0, 0.5)),
                         neutron(r=(0.1, 0, 0), v=(0.1, 0.0, 0.5))]
                        )


if __name__ == '__main__':
    # test()
    # test_process()
    test_guide()
