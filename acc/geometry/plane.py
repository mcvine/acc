#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

from math import isnan, nan, sqrt
from numba import cuda
from numpy import empty


@cuda.jit(device=True, inline=True)
def dot_product(p, q):
    """
    Calculate the dot product of two vectors.

    Parameters:
    tuple: x,y,z of first vector
    tuple: x,y,z of second vector

    Returns:
    float: the vectors' dot product
    """
    (px, py, pz) = p
    (qx, qy, qz) = q
    return px * qx + py * qy + pz * qz


@cuda.jit(device=True, inline=True)
def cross_product(p, q):
    """
    Calculate the cross product of two vectors.

    Parameters:
    tuple: x,y,z of first vector
    tuple: x,y,z of second vector

    Returns:
    tuple: x,y,z of the vectors' cross product
    """
    (px, py, pz) = p
    (qx, qy, qz) = q
    return (py * qz - pz * qy,
            pz * qx - px * qz,
            px * qy - py * qx)


@cuda.jit(device=True, inline=True)
def size_of_vector(p):
    """
    Calculate the scalar magnitude of a vector.

    Parameters:
    tuple: a vector

    Returns:
    float: the vector's magnitude
    """
    magnitude = 0
    for p_n in p:
        magnitude += p_n * p_n
    return sqrt(magnitude)


@cuda.jit(device=True, inline=True)
def plane_from_normal(point, normal):
    """
    Construct a plane.

    Parameters:
    point (tuple): x,y,z position of a point on the plane
    normal (tuple): x,y,z direction of a normal to the plane

    Returns a tuple:
    tuple: x,y,z of a point on the plane
    tuple: x,y,z of a unit normal to the plane
    """
    magnitude = size_of_vector(normal)
    if magnitude == 0:
        return point, (nan, nan, nan)
    else:
        (nx, ny, nz) = normal
        return point, (nx / magnitude, ny / magnitude, nz / magnitude)


@cuda.jit(device=True, inline=True)
def plane_from_points(point_p, point_q, point_r):
    """
    Construct a plane that includes the three given points.

    Parameters:
    point_p (tuple): x,y,z position of a first point
    point_q (tuple): x,y,z position of a second point
    point_r (tuple): x,y,z position of a third point

    Returns a tuple:
    tuple: x,y,z of a point on the plane
    tuple: x,y,z of a unit normal to the plane
    """
    (px, py, pz) = point_p
    (qx, qy, qz) = point_q
    (rx, ry, rz) = point_r
    q_sub_p = (qx - px, qy - py, qz - pz)
    r_sub_p = (rx - px, ry - py, rz - pz)
    normal = cross_product(q_sub_p, r_sub_p)
    return plane_from_normal(point_p, normal)


@cuda.jit(device=True, inline=True)
def plane_intersection_duration(plane_point, plane_normal, position, velocity):
    """
    Find a particle's next intersection of a plane.

    Parameters:
    plane_point (tuple): x,y,z position of a point on the plane
    plane_normal (tuple): x,y,z direction of a unit normal to the plane
    position (tuple): x,y,z of the particle's position
    velocity (tuple): x,y,z of the particle's velocity

    Returns a tuple:
    tuple: x,y,z of where the particle intersects this plane
    float: the time taken to reach this plane, may be negative
    """
    (ppx, ppy, ppz) = plane_point
    (pox, poy, poz) = position
    (vex, vey, vez) = velocity
    dot_p_l = dot_product(plane_normal, (ppx - pox, ppy - poy, ppz - poz))
    dot_n_v = dot_product(plane_normal, velocity)

    if dot_n_v == 0:
        # velocity is along the plane
        return (nan, nan, nan), nan

    duration = dot_p_l / dot_n_v
    intersection = (pox + vex * duration,
                    poy + vey * duration,
                    poz + vez * duration)
    return intersection, duration


@cuda.jit(device=True, inline=True)
def plane_reflect(plane_normal, velocity):
    """
    Calculate the velocity of a particle reflected off a plane.

    Parameters:
    plane_normal (tuple): x,y,z direction of a unit normal to the plane
    velocity (tuple): x,y,z of the particle's velocity

    Returns:
    tuple: x,y,z of the outgoing reflection
    """
    (pnx, pny, pnz) = plane_normal
    (vex, vey, vez) = velocity
    dot_d_n = dot_product(plane_normal, velocity)
    reflection = (vex - 2 * dot_d_n * pnx,
                  vey - 2 * dot_d_n * pny,
                  vez - 2 * dot_d_n * pnz)
    return reflection


@cuda.jit
def plane_from_normal_kernel(point, normal, plane):
    """
    GPU kernel wrapping plane_from_normal.

    Parameters:
    point (vector): a point on the plane
    normal (vector): a normal to the plane
    plane (array): write target,
        returns plane state for passing to other kernels
    """
    (plane_point, plane_normal) = plane_from_normal(point, normal)
    (plane[0][0], plane[0][1], plane[0][2]) = plane_point
    (plane[1][0], plane[1][1], plane[1][2]) = plane_normal


@cuda.jit
def plane_from_points_kernel(point_p, point_q, point_r, plane):
    """
    GPU kernel wrapping plane_from_points.

    Parameters:
    point_p (vector): a first point on the plane
    point_q (vector): a second point on the plane
    point_r (vector): a third point on the plane
    plane (array): write target,
        returns plane state for passing to other kernels
    """
    (plane_point, plane_normal) = plane_from_points(point_p, point_q, point_r)
    (plane[0][0], plane[0][1], plane[0][2]) = plane_point
    (plane[1][0], plane[1][1], plane[1][2]) = plane_normal


@cuda.jit
def plane_intersection_duration_kernel(plane, position, velocity,
                                       intersection, duration):
    """
    GPU kernel wrapping plane_intersection_duration.

    Parameters:
    position (vector): the particle's position
    velocity (vector): the particle's velocity
    intersection (vector): write target,
        returns where the particle intersects this plane
    duration (float): write target,
        returns the time taken to reach this plane, may be negative
    """
    (plane_point, plane_normal) = (plane[0], plane[1])
    (intersection_rv, duration_rv) = plane_intersection_duration(
        plane_point, plane_normal, position, velocity)
    (intersection[0], intersection[1], intersection[2]) = \
        (intersection_rv[0], intersection_rv[1], intersection_rv[2])
    duration[0] = duration_rv


@cuda.jit
def plane_reflect_kernel(plane, velocity, reflection):
    """
    GPU kernel wrapping plane_reflect.

    Parameters:
    plane (array): plane state as from kernel method that returns a plane
    velocity (vector): the particle's velocity
    reflection (vector): write target, returns the outgoing reflection
    """
    plane_normal = plane[1]
    reflection_rv = plane_reflect(plane_normal, velocity)
    (reflection[0], reflection[1], reflection[2]) = \
        (reflection_rv[0], reflection_rv[1], reflection_rv[2])


class Plane:

    def __init__(self, point, normal):
        """
        Construct a plane.

        Parameters:
        point (vector): x,y,z position of a point on the plane
        normal (vector): x,y,z direction of a normal to the plane

        Returns:
        Plane: a plane specified by the given point and normal
        """
        self.state = empty((2, 3), dtype=float)
        plane_from_normal_kernel[1, 1](point, normal, self.state)
        if any(map(isnan, self.state.flatten())):
            raise ValueError

    @staticmethod
    def construct(point_p, point_q, point_r):
        """
        Construct a plane that includes the three given points.

        Parameters:
        point_p (vector): x,y,z position of a first point
        point_q (vector): x,y,z position of a second point
        point_r (vector): x,y,z position of a third point

        Returns:
        Plane: a plane that includes the three given points
        """
        state = empty((2, 3), dtype=float)
        plane_from_points_kernel[1, 1](point_p, point_q, point_r, state)
        if any(map(isnan, state.flatten())):
            raise ValueError
        return Plane(state[0], state[1])

    def intersection_duration(self, position, velocity):
        """
        Find a particle's next intersection of this plane.

        Parameters:
        position (vector): x,y,z of the particle's position
        velocity (vector): x,y,z of the particle's velocity

        Returns a tuple:
        vector: x,y,z of where the particle intersects this plane
        float: the time taken to reach this plane, may be negative
        """
        intersection = empty(3, dtype=float)
        duration = empty(1, dtype=float)
        plane_intersection_duration_kernel[1, 1](
            self.state, position, velocity, intersection, duration)
        return None if isnan(duration[0]) else (intersection, duration[0])

    def reflect(self, velocity):
        """
        Calculate the velocity of a particle reflected off this plane.

        Parameters:
        velocity (vector): x,y,z of the particle's velocity

        Returns:
        vector: x,y,z of the outgoing reflection
        """
        reflection = empty(3, dtype=float)
        plane_reflect_kernel[1, 1](self.state, velocity, reflection)
        return reflection
