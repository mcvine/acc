#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

import math
from mcni.AbstractComponent import AbstractComponent
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
# import mcvine
# import mcvine.components as mc
import numpy

category = 'optics'


class Plane:

    def __init__(self, point, normal):

        """Construct a plane.

        Parameters:
        point (vector): x,y,z position of a point on the plane
        normal (vector): x,y,z direction of a normal to the plane

        Returns:
        Plane: a plane specified by the given point and normal
        """
        if len(point) != 3 or len(normal) != 3:
            raise ValueError('expecting three dimensions')
        elif not normal.any():
            raise ValueError('normal has no magnitude')

        self.point = point
        self.normal = numpy.array(normal / numpy.linalg.norm(normal)).reshape(1, 3)

    @classmethod
    def construct(cls, point_p, point_q, point_r):

        """Construct a plane that includes the three given points.

        Parameters:
        point_p (vector): x,y,z position of a first point
        point_q (vector): x,y,z position of a second point
        point_r (vector): x,y,z position of a third point

        Returns:
        Plane: a plane that includes the three given points
        """
        return Plane(
            point_p,
            numpy.cross(point_q - point_p, point_r - point_p))

    # To do: add unit tests.
    def intersection_duration(self, position, velocity):

        """Find a particle's next intersection of this plane.

        Parameters:
        position (vector): x,y,z of the particle's position
        velocity (vector): x,y,z of the particle's velocity

        Returns a tuple:
        vector: x,y,z of where the particle intersects this plane
        float: the time taken to reach this plane, may be negative
        """
        if len(position.shape) != 2 or len(velocity.shape) != 2:
            raise ValueError('expecting 2D array')
        if position.shape[1] != 3 or velocity.shape[1] != 3:
            raise ValueError('expecting three dimensions')
        elif not velocity.any():
            raise ValueError('particle is stationary')

        dot_p_l = numpy.dot(self.normal, (self.point - position).T)
        dot_n_v = numpy.dot(self.normal, velocity.T)

        duration = numpy.full((1, position.shape[0]), numpy.inf)
        numpy.divide(dot_p_l, dot_n_v, where=dot_n_v != 0, out=duration)
        duration = duration.T
        intersection = position + velocity * duration
        return (intersection, duration)

    # To do: add unit tests.
    # Are both sides mirrored or does the direction of the normal matter?
    def reflect(self, velocity):

        """Calculate the velocity of a particle reflected off this plane.

        Parameters:
        velocity (vector): x,y,z of the particle's velocity

        Returns:
        vector: x,y,z of the outgoing reflection
        """
        dot_d_n = numpy.dot(velocity, self.normal.flatten())
        reflection = velocity - 2 * dot_d_n * self.normal.flatten()
        return reflection


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
        self.m_over_h = self.m_neutron / self.hbar

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
                pos_dur_next = side.intersection_duration(pos_curr, vel_curr)
                if pos_dur_next:
                    (pos_next, dur_next) = pos_dur_next
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
            velocity_i - velocity_f, axis=1)  # length of scattering vector (in A^-1)
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
        time = arr[:, 8]
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
                                             where=((intersection[1] > 1e-5) & (s != old_side) & (old_side != 0)),
                                             out=new_duration)

                # Update the index of which side was hit based on new minimum
                side = numpy.where(new_duration != duration, s, side)
                duration = new_duration.copy()

                # If duration is inf, mark the side as invalid
                side = numpy.where(duration == numpy.inf, -1, side)

            # Propagate the neutrons based on the minimum times
            position += numpy.multiply(velocity, duration, where=((duration != numpy.inf) | (old_side != 0)))
            old_side = side.copy()

            velocity_before = velocity.copy()

            # Update the velocity due to reflection
            # TODO: vectorize this
            for ind in range(len(neutrons)):
                # Only update the velocity if reflecting on one of the guide sides
                if side[ind] != 0 and side[ind] != -1:
                    velocity[ind] = self.sides[side.item(ind)].reflect(velocity[ind])

            # Calculate reflectivity
            # TODO: Fix - this is giving large numbers, check this and its units
            reflectivity = self.calc_reflectivity(velocity_before, velocity)
            prob *= numpy.where(side != 0, reflectivity.reshape(arr.shape[0], 1), prob)
            iter += 1

        print("process took {} iterations".format(iter))
        # Update neutron positions, velocities and times and select those that hit the guide exit
        arr[:, 0:3] = position
        arr[:, 3:6] = velocity
        arr[:, 8] = duration.reshape((arr.shape[0],))
        arr[:, 9] = prob.reshape((arr.shape[0],))
        good = arr[(side == 0).flatten(), :]

        neutrons.resize(good.shape[0], neutrons[0])
        neutrons.from_npyarr(good)
        return


# To do: turn this into a proper test
def test():
    import math

    # set up simple guide
    length = 12
    guide = Guide('test guide', 3, 3, 2, 2, length)
    guide_angle = math.atan(((3 - 2) / 2) / length)

    # set up particle
    position = numpy.array([0.8, -1, 0], dtype=float)
    velocity = numpy.array([0, 1, 3], dtype=float)

    # determine expected angle of exit assuming two reflections
    angle_from_z = math.atan(velocity[1] / velocity[2])
    for i in range(0, 2):
        offset_from_normal = math.pi/2 - angle_from_z - guide_angle
        angle_from_z = math.pi - angle_from_z - 2 * offset_from_normal

    # propagate particle through guide
    (position, velocity, duration) = guide.propagate(position, velocity)

    # report outcome
    print('should be at x=0.8, got to x={}'.format(position[0]))
    print('should have reached z={}, got to z={}'.format(length, position[2]))
    print('guide length is {}, distance traveled by particle is {}'.format(
        length, duration * numpy.linalg.norm(velocity)))
    print('final angle is {} radians, expected {} radians'.format(
        math.atan(velocity[1] / velocity[2]),
        angle_from_z))


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
    neutrons_arr = neutrons_as_npyarr(neutrons)
    neutrons_arr.shape = -1, ndblsperneutron
    print(neutrons_arr)


if __name__ == '__main__':
    # test()
    test_process()
