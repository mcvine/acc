#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

import math
from mcni.AbstractComponent import AbstractComponent
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
import numpy
from mcvine.acc.geometry.numpy_plane import Plane

category = 'optics'


class Guide(AbstractComponent):

    @staticmethod
    def __new_plane(point_p, point_q, point_r):
        # Represent everything as numpy arrays in preparation for
        # vectorized operations across multi-dimensional arrays.
        return Plane.construct(numpy.array(point_p, dtype=float),
                               numpy.array(point_q, dtype=float),
                               numpy.array(point_r, dtype=float))

    def __init__(
            self, name,
            w1, h1, w2, h2, l,
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
        self.entrance = Guide.__new_plane([+w1/2, +h1/2, 0],
                                          [-w1/2, +h1/2, 0],
                                          [-w1/2, -h1/2, 0])
        self.entrance_width = w1
        self.entrance_height = h1
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
                    if 0 < dur_next < dur_min:
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
                    1.0 - numpy.tanh((Q - self.m * self.Qc) / self.W)) * (
                                    1.0 - self.alpha * (Q - self.Qc)),
                        self.R0)
        return R

    # To do: use propagate method to implement process method
    def process(self, neutrons):
        if not len(neutrons):
            return
        arr = neutrons_as_npyarr(neutrons)
        arr.shape = -1, ndblsperneutron

        # Filter out neutrons that do not hit guide entrance
        entrance_intersection, entrance_dur = \
            self.entrance.intersection_duration(arr[:, 0:3], arr[:, 3:6])
        mask = ((entrance_intersection[:, 0] < self.entrance_width / 2) &
                (entrance_intersection[:, 0] > -self.entrance_width / 2) &
                (entrance_intersection[:, 1] < self.entrance_height / 2) &
                (entrance_intersection[:, 1] > -self.entrance_height / 2) &
                ((entrance_dur.flatten() > 1e-10) |
                 numpy.isclose(arr[:, 2], 0.0))).flatten()
        arr = arr[mask]
        entrance_dur = entrance_dur[mask]
        neutrons.from_npyarr(arr)
        if not len(neutrons):
            return

        position = arr[:, 0:3]  # x, y, z
        velocity = arr[:, 3:6]  # vx, vy, vz
        time = arr[:, 8].reshape((arr.shape[0], 1))
        prob = arr[:, 9].reshape((arr.shape[0], 1))

        # initialize arrays containing neutron duration and side index
        side = numpy.full((arr.shape[0], 1), -2, dtype=int)
        old_side = side.copy()
        new_duration = numpy.full((arr.shape[0], 1), numpy.inf)

        # propagate to the guide entrance plane
        position += numpy.multiply(velocity, entrance_dur)
        time += entrance_dur

        # Iterate until all neutrons hit end of guide or are absorbed
        iter = 0
        while numpy.count_nonzero(side == 0) + numpy.count_nonzero(
                side == -1) != len(neutrons):
            duration = numpy.full((arr.shape[0], 1), numpy.inf)
            # Calculate minimum intersection time with all sides of the guide
            for s in range(0, len(self.sides)):
                intersection = self.sides[s].intersection_duration(position, velocity)
                new_duration = numpy.minimum(intersection[1], duration,
                                             where=((intersection[1] > 1e-10) &
                                                    (s != old_side) &
                                                    (old_side != 0) &
                                                    (old_side != -1)),
                                             out=new_duration)

                # Update the index of which side was hit based on new minimum
                side = numpy.where(new_duration != duration, s, side)
                duration = new_duration.copy()

                # If duration is inf, mark the side as invalid
                side = numpy.where(duration == numpy.inf, -1, side)

            # Propagate the neutrons based on the minimum times
            mask = (old_side != side).flatten()
            position[mask] += velocity[mask, :] * duration[mask]
            time[mask] += duration[mask]

            velocity_before = velocity.copy()

            # Update the velocity due to reflection
            mask = numpy.logical_and(old_side != side, side > 0)
            for s in range(1, len(self.sides)):
                # Only update the velocity if reflecting on one of the guide sides
                side_mask = numpy.logical_and(mask, side == s).flatten()
                velocity[side_mask] = self.sides[s].reflect(velocity[side_mask])

            # Calculate reflectivity
            mask = mask.flatten()
            reflectivity = self.calc_reflectivity(velocity_before[mask, :],
                                                  velocity[mask, :])
            prob[mask] = prob[mask] * reflectivity.reshape(
                reflectivity.shape[0], 1)
            side[prob <= 0] = -1

            old_side = side.copy()
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
