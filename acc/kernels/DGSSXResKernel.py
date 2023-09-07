import numpy as np
from mcvine.acc.neutron import v2e
from numba import cuda, float64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from typing import List
import math

from .. import vec3
from ..config import get_numba_floattype

NB_FLOAT = get_numba_floattype()

epsilon = 1e-7


@cuda.jit(device=True, inline=True)
def choose_direction(threadindex: int, rng_states: xoroshiro128p_type, direction: List[float], target_direction: List[float], target_radius: float):
    """
    Randomly choose a direction inside the solid angle bound by target_radius in the direction of target_direction
    Returns the size of solid angle
    """

    # calculate square of distance
    l2 = vec3.length2(target_direction)

    costheta_max = math.sqrt(l2 / (target_radius*target_radius+l2))
    if target_radius < 0.0:
        costheta_max *= -1.0

    solidangle = 2.0*math.pi*(1.0 - costheta_max)

    # choose theta and phi
    r = xoroshiro128p_uniform_float32(rng_states, threadindex)
    r = r * (costheta_max - 1.0) + 1.0

    theta = math.acos(r)
    phi = xoroshiro128p_uniform_float32(rng_states, threadindex) * 2.0*math.pi

    # choose normal vector
    norm = cuda.local.array(3, dtype=float64)
    if target_direction[0] == 0.0 and target_direction[2] == 0.0:
        # TODO: change to tolerance or ensure these can be exactly 0.0
        norm[0] = 1.0
        norm[1] = 0.0
        norm[2] = 0.0
    else:
        norm[0] = -target_direction[2]
        norm[1] = 0.0
        norm[2] = target_direction[0]

    vec3.cross(norm, target_direction, norm)

    vec3.copy(target_direction, direction)
    vec3.rotate(direction, norm, theta, epsilon)
    vec3.rotate(direction, target_direction, phi, epsilon)

    return solidangle


@cuda.jit(device=True)
def scatter(threadindex, rng_states, neutron, target, target_radius, tof_target, delta_tof):
    """
    Scatters to a specific target location (within circle of given radius) and at a specific TOF (within +/- delta_tof / 2)
    """

    # vector from neutron position to target
    displacement = cuda.local.array(3, dtype=float64)
    vec3.subtract(target, neutron[0:3], displacement)

    # pick direction of scattered neutron
    vf_dir = cuda.local.array(3, dtype=float64)
    solid_angle = choose_direction(
        threadindex, rng_states, vf_dir, displacement, target_radius)
    vec3.normalize(vf_dir)

    # randomly pick tof
    r = xoroshiro128p_uniform_float32(rng_states, threadindex)
    min_tof = tof_target - 0.5 * delta_tof
    max_tof = tof_target + 0.5 * delta_tof
    tof = r * (max_tof - min_tof) + min_tof

    # compute vf length
    vf = vec3.length(displacement) / (tof - neutron[-2])
    # compute vf vector
    vec3.scale(vf_dir, vf)
    # compute final energy
    Ef = v2e(vf)

    # compute probability factor
    dE_over_dt = 2.0 * Ef / tof
    prob = solid_angle / 4.0 / math.pi * delta_tof * dE_over_dt
    vf_over_vi = vf / vec3.length(neutron[3:6])

    # set vf
    neutron[3] = vf_dir[0]
    neutron[4] = vf_dir[1]
    neutron[5] = vf_dir[2]
    neutron[-1] *= prob * vf_over_vi
