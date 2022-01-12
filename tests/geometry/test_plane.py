#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

import numpy
import pytest
from mcvine.acc import test
from mcvine.acc.geometry.plane import Plane


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_construct_no_normal():
    """
    Check that one cannot construct a plane with a zero-magnitude normal.
    """
    with pytest.raises(ValueError):
        Plane(numpy.array([1, 2, 3], dtype=float),
              numpy.array([0, 0, 0], dtype=float))


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_construct_collinear_points():
    """
    Check that one cannot construct a plane from points along the same line.
    """
    with pytest.raises(ValueError):
        Plane.construct(numpy.array([1, 2, 3], dtype=float),
                        numpy.array([2, 4, 6], dtype=float),
                        numpy.array([3, 6, 9], dtype=float))


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("x1", [2, 3])
@pytest.mark.parametrize("y1", [5, 7])
@pytest.mark.parametrize("z1", [11, 13])
@pytest.mark.parametrize("x2", [17, 19])
@pytest.mark.parametrize("y2", [23, 29])
@pytest.mark.parametrize("z2", [31, 37])
def test_construct_correct_normal(x1, y1, z1, x2, y2, z2):
    """
    Check that a plane constructed from points has a corresponding normal.
    """
    point_1 = numpy.array([x1, y1, z1], dtype=float)
    point_2 = numpy.array([x2, y2, z2], dtype=float)
    point_3 = numpy.array([-1, -2, -3], dtype=float)
    plane = Plane.construct(point_1, point_2, point_3)
    (plane_point, plane_normal) = plane.state

    normal_magnitude = numpy.linalg.norm(plane_normal)

    # The normal should be a unit vector.
    assert numpy.isclose(1, normal_magnitude)

    # The normal should be perpendicular to the vectors defined by the points.
    assert numpy.isclose(0, numpy.dot(plane_normal, point_1 - point_2))
    assert numpy.isclose(0, numpy.dot(plane_normal, point_1 - point_3))


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("x1", [2, 3])
@pytest.mark.parametrize("y1", [5, 7])
@pytest.mark.parametrize("z1", [11, 13])
@pytest.mark.parametrize("x2", [17, 19])
@pytest.mark.parametrize("y2", [23, 29])
@pytest.mark.parametrize("z2", [31, 37])
def test_construct_points_on_plane(x1, y1, z1, x2, y2, z2):
    """
    Check that a plane constructed from points has those points on the plane.
    """
    point_1 = numpy.array([x1, y1, z1], dtype=float)
    point_2 = numpy.array([x2, y2, z2], dtype=float)
    point_3 = numpy.array([-1, -2, -3], dtype=float)
    plane = Plane.construct(point_1, point_2, point_3)
    (plane_point, plane_normal) = plane.state

    for point in [point_1, point_2, point_3]:
        # A vector from a point on the plane to a point used to construct
        # the plane should be perpendicular to the normal.
        assert numpy.isclose(0, numpy.dot(point - plane_point,
                                          plane_normal))


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("xp", [2, 3])
@pytest.mark.parametrize("yp", [5, 7])
@pytest.mark.parametrize("zp", [11, 13])
@pytest.mark.parametrize("xv", [17, 19])
@pytest.mark.parametrize("yv", [23, 29])
@pytest.mark.parametrize("zv", [31, 37])
def test_reflection(xp, yp, zp, xv, yv, zv):
    """
    Check the properties of reflections off the plane.
    """
    velocity = numpy.array([xv, yv, zv], dtype=float)

    plane_1 = Plane.construct(numpy.array([1, 2, 3], dtype=float),
                              numpy.array([2, 3, 4], dtype=float),
                              numpy.array([xp, yp, zp], dtype=float))
    (plane_1_point, plane_1_normal) = plane_1.state

    # Plane 2 mirrors the normal of plane 1.
    plane_2 = Plane(plane_1_point, -plane_1_normal)

    reflection_1 = plane_1.reflect(velocity)
    reflection_2 = plane_2.reflect(velocity)

    # Reflection should not change the speed of a particle.
    magnitudes = map(numpy.linalg.norm, [velocity, reflection_1, reflection_2])
    assert numpy.isclose(0, numpy.var(list(magnitudes)))

    # Mirroring the normal should not change the reflection.
    assert numpy.allclose(reflection_1, reflection_2)

    cross_n_v = numpy.cross(plane_1_normal, velocity)
    cross_n_r = numpy.cross(plane_1_normal, reflection_1)

    # For these cases, there should be a non-zero cross product of the normal
    # with the particle's velocity both before and after reflection.
    assert not numpy.isclose(0, numpy.linalg.norm(cross_n_v))
    assert not numpy.isclose(0, numpy.linalg.norm(cross_n_r))

    # The direction of reflection should make those cross products correspond.
    assert numpy.allclose(cross_n_v, cross_n_r)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_reflection_along_normal():
    """
    Check reflection exactly along the normal.
    """
    plane = Plane.construct(numpy.array([11, 13, 17], dtype=float),
                            numpy.array([19, 23, 29], dtype=float),
                            numpy.array([31, 37, 41], dtype=float))
    (plane_point, plane_normal) = plane.state

    velocity = 5 * plane_normal

    reflection = plane.reflect(velocity)

    # The reflection should be the opposite of the velocity.
    assert numpy.allclose(velocity, -reflection)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_intersection_duration_stationary():
    """
    Check that there is no intersection with a stationary particle.
    """
    plane = Plane.construct(numpy.array([11, 13, 17], dtype=float),
                            numpy.array([19, 23, 29], dtype=float),
                            numpy.array([31, 37, 41], dtype=float))

    position = numpy.array([1, 1, 1], dtype=float)
    velocity = numpy.array([0, 0, 0], dtype=float)

    assert None is plane.intersection_duration(position, velocity)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("xp", [2, 3])
@pytest.mark.parametrize("yp", [5, 7])
@pytest.mark.parametrize("zp", [11, 13])
@pytest.mark.parametrize("xv", [17, 19])
@pytest.mark.parametrize("yv", [23, 29])
@pytest.mark.parametrize("zv", [31, 37])
def test_intersection_duration_moving(xp, yp, zp, xv, yv, zv):
    """
    Check the properties of intersections with the plane.
    """
    plane = Plane.construct(numpy.array([11, 13, 17], dtype=float),
                            numpy.array([19, 23, 29], dtype=float),
                            numpy.array([31, 37, 41], dtype=float))
    (plane_point, plane_normal) = plane.state

    position = numpy.array([xp, yp, zp], dtype=float)
    velocity = numpy.array([xv, yv, zv], dtype=float)

    (intersection, duration) = plane.intersection_duration(position, velocity)

    # A vector from a point on the plane to the intersection should be
    # perpendicular to the normal.
    assert numpy.isclose(0, numpy.dot(intersection - plane_point,
                                      plane_normal))

    # The intersection should be reached after the duration.
    assert numpy.allclose(position + velocity * duration, intersection)
