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
    def intersection_distance(self, position, direction):

        """Find a particle's next intersection of this plane.

        Parameters:
        position (vector): x,y,z of the particle's position
        direction (vector): x,y,z of the particle's velocity

        Returns a tuple (or None if particle does not intersect):
        vector: x,y,z of where the particle intersects this plane
        float: the distance traveled to reach this plane
        """
        if len(position) != 3 or len(direction) != 3:
            raise ValueError('expecting three dimensions')
        elif not direction.any():
            raise ValueError('direction has no magnitude')

        dot_p_l = numpy.dot(self.normal, self.point - position)
        dot_n_d = numpy.dot(self.normal, direction)
        if dot_n_d == 0:
            # direction is along plane
            return None

        progress = dot_p_l / dot_n_d
        if progress <= 0:
            # intersection is not ahead of particle
            return None

        intersection = position + direction * progress
        distance = numpy.linalg.norm(intersection - position)
        return (intersection, distance)

    # To do: add unit tests.
    # Are both sides mirrored or does the direction of the normal matter?
    def reflect(self, direction):

        """Calculate the direction of a particle reflected off this plane.

        Parameters:
        direction (vector): x,y,z of the incident particle

        Returns:
        vector: x,y,z of the outgoing reflection
        """
        dot_d_n = numpy.dot(direction, self.normal)
        reflection = direction - 2 * dot_d_n * self.normal
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

    def propagate(self, position, direction):

        """Propagate a particle through this guide.

        Parameters:
        position (vector): x,y,z of the particle's initial position
        direction (vector): x,y,z of the particle's initial velocity

        Returns a tuple (or None if particle does not exit):
        vector: x,y,z of the particle's exit position
        vector: x,y,z of the particle's exit velocity
        float: the total distance traveled by the particle within the guide
        """
        (pos_curr, dir_curr) = (position, direction)
        distance = 0
        ind_prev = None
        while True:
            # Find which side is hit next, i.e., in shortest distance.
            dis_min = math.inf
            # To do: no need to find intersection of most recently hit side.
            pos_dis_nexts = [side.intersection_distance(pos_curr, dir_curr)
                             for side in self.sides]
            for index in range(0, len(pos_dis_nexts)):
                if pos_dis_nexts[index]:
                    (pos_next, dis_next) = pos_dis_nexts[index]
                    if index != ind_prev and dis_next < dis_min:
                        ind_min = index
                        (pos_min, dis_min) = (pos_next, dis_next)
            if dis_min == math.inf:
                # No new side was hit.
                return None
            distance += dis_min
            if ind_min == 0:
                # The first side is the exit from the guide.
                return (pos_min, dir_curr, distance)
            # Update particle to be reflecting from the new side.
            pos_curr = pos_min
            dir_curr = self.sides[ind_min].reflect(dir_curr)
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
    position = numpy.array([1, -1, 0], dtype=float)
    direction = numpy.array([0, 1, 3], dtype=float)

    # determine expected angle of exit assuming two reflections
    angle_from_z = math.atan(direction[1] / direction[2])
    for i in range(0, 2):
        offset_from_normal = math.pi/2 - angle_from_z - guide_angle
        angle_from_z = math.pi - angle_from_z - 2 * offset_from_normal

    # propagate particle through guide
    (position, direction, distance) = guide.propagate(position, direction)

    # report outcome
    print('should be at x=1, got to x={}'.format(position[0]))
    print('should have reached z={}, got to z={}'.format(length, position[2]))
    print('guide length is {}, distance traveled by particle is {}'.format(
        length, distance))
    print('final angle is {} radians, expected {} radians'.format(
        math.atan(direction[1] / direction[2]),
        angle_from_z))
