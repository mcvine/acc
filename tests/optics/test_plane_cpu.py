#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

import numpy
import pytest
from acc.geometry.plane import Plane


def test_construct_no_normal():
    """
    Check that one cannot construct a plane with a zero-magnitude normal.
    """
    with pytest.raises(ValueError):
        Plane(numpy.array([1, 2, 3]), numpy.array([0, 0, 0]))


def test_construct_collinear_points():
    """
    Check that one cannot construct a plane from points along the same line.
    """
    with pytest.raises(ValueError):
        Plane.construct(numpy.array([1, 2, 3]),
                        numpy.array([2, 4, 6]),
                        numpy.array([3, 6, 9]))


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

    normal_magnitude = numpy.linalg.norm(plane.normal[0])

    # The normal should be a unit vector.
    assert abs(normal_magnitude - 1) < 1e-10

    # The normal should be perpendicular to the vectors defined by the points.
    assert numpy.dot(plane.normal[0], point_1 - point_2) < 1e-10
    assert numpy.dot(plane.normal[0], point_1 - point_3) < 1e-10


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

    for point in [point_1, point_2, point_3]:
        # A vector from a point on the plane to a point used to construct
        # the plane should be perpendicular to the normal.
        assert numpy.dot(point - plane.point, plane.normal[0]) < 1e-10


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

    # Plane 2 mirrors the normal of plane 1.
    plane_2 = Plane(plane_1.point, -plane_1.normal[0])

    reflection_1 = plane_1.reflect(velocity)
    reflection_2 = plane_2.reflect(velocity)

    # Reflection should not change the speed of a particle.
    magnitudes = map(numpy.linalg.norm, [velocity, reflection_1, reflection_2])
    assert numpy.var(list(magnitudes)) < 1e-10

    # Mirroring the normal should not change the reflection.
    assert numpy.linalg.norm(reflection_1 - reflection_2) < 1e-10

    # For these cases, the velocity should not match the normal.
    # This is crucial for the following cross products.
    assert numpy.linalg.norm(velocity - plane_1.normal) > 1e-10
    assert numpy.linalg.norm(reflection_1 - plane_1.normal) > 1e-10

    cross_n_v = numpy.cross(plane_1.normal, velocity)
    cross_n_r = numpy.cross(plane_1.normal, reflection_1)

    # There should be a non-zero cross product of the normal with the
    # particle's velocity both before and after reflection.
    assert numpy.linalg.norm(cross_n_v) > 1e-10
    assert numpy.linalg.norm(cross_n_r) > 1e-10

    # The direction of reflection should make those cross products correspond.
    assert numpy.linalg.norm(cross_n_v - cross_n_r) < 1e-10


def test_reflection_along_normal():
    """
    Check reflection exactly along the normal.
    """
    plane = Plane.construct(numpy.array([11, 13, 17], dtype=float),
                            numpy.array([19, 23, 29], dtype=float),
                            numpy.array([31, 37, 41], dtype=float))

    velocity = 5 * plane.normal[0]

    reflection = plane.reflect(velocity)

    # Reflection should not change the speed of a particle.
    assert numpy.linalg.norm(velocity) - numpy.linalg.norm(reflection) < 1e-10

    # The reflection should be the opposite of the velocity.
    assert numpy.linalg.norm(velocity + reflection) < 1e-10


def test_intersection_duration_stationary():
    """
    Check that there is no intersection with a stationary particle.
    """
    plane = Plane.construct(numpy.array([11, 13, 17], dtype=float),
                            numpy.array([19, 23, 29], dtype=float),
                            numpy.array([31, 37, 41], dtype=float))

    position = numpy.array([1, 1, 1], dtype=float)
    velocity = numpy.array([0, 0, 0], dtype=float)

    with pytest.raises(ValueError):
        plane.intersection_duration(position, velocity)


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

    position = numpy.array([[xp, yp, zp]], dtype=float)
    velocity = numpy.array([[xv, yv, zv]], dtype=float)

    (intersection, duration) = plane.intersection_duration(position, velocity)

    # A vector from a point on the plane to the intersection should be
    # perpendicular to the normal.
    assert numpy.dot(intersection[0] - plane.point, plane.normal[0]) < 1e-10

    path = intersection - position

    velocity_magnitude = numpy.linalg.norm(velocity)
    path_magnitude = numpy.linalg.norm(path)

    # The duration should match what would be expected from the distance to the
    # intersection and the particle's speed.
    assert duration - path_magnitude / velocity_magnitude < 1e-10

    # The intersection should be in the direction of the velocity.
    assert numpy.linalg.norm(numpy.cross(velocity, path)) < 1e-10
