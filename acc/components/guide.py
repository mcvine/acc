#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

import math
from mcni.AbstractComponent import AbstractComponent
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
        self.normal = normal / numpy.linalg.norm(normal)

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

        Returns a tuple (or None if particle does not intersect):
        vector: x,y,z of where the particle intersects this plane
        float: the time taken to reach this plane
        """
        if len(position) != 3 or len(velocity) != 3:
            raise ValueError('expecting three dimensions')
        elif not velocity.any():
            raise ValueError('particle is stationary')

        dot_p_l = numpy.dot(self.normal, self.point - position)
        dot_n_v = numpy.dot(self.normal, velocity)
        if dot_n_v == 0:
            # velocity is along plane
            return None

        duration = dot_p_l / dot_n_v
        if duration <= 0:
            # intersection is not ahead of particle
            return None

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
        dot_d_n = numpy.dot(velocity, self.normal)
        reflection = velocity - 2 * dot_d_n * self.normal
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
            # To do: no need to find intersection of most recently hit side.
            pos_dur_nexts = [side.intersection_duration(pos_curr, vel_curr)
                             for side in self.sides]
            for index in range(0, len(pos_dur_nexts)):
                if pos_dur_nexts[index]:
                    (pos_next, dur_next) = pos_dur_nexts[index]
                    if index != ind_prev and dur_next < dur_min:
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

    # To do: use propagate method to implement process method


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
