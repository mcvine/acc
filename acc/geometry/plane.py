#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

import numpy


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

        normal_unit_vector = normal / numpy.linalg.norm(normal)

        self.point = point
        self.normal = numpy.array(normal_unit_vector).reshape(1, 3)

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
        return intersection, duration

    def reflect(self, velocity):

        """Calculate the velocity of a particle reflected off this plane.

        Parameters:
        velocity (vector): x,y,z of the particle's velocity

        Returns:
        vector: x,y,z of the outgoing reflection
        """
        dot_d_n = numpy.dot(velocity, self.normal.flatten())
        reflection = velocity - 2 * dot_d_n[:, numpy.newaxis] * \
                     self.normal.flatten()
        return reflection
