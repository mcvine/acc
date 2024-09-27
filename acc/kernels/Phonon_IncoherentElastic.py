import numpy as np
from mcni.utils import conversion
from mcvine.acc.neutron import v2e
from numba import cuda, float64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
import math

from .. import vec3
from ..config import get_numba_floattype

NB_FLOAT = get_numba_floattype()

epsilon = 1e-7


@cuda.jit(device=True)
def scatter(threadindex, rng_states, neutron, dw_core):
    """
    Scatters to a specific target location (within circle of given radius) and at a specific TOF (within +/- delta_tof / 2)
    """

    v = neutron[3:6]

    # incident neutron velocity
    vi = vec3.length(v)

    # randomly pick a theta between [0, PI]
    r = xoroshiro128p_uniform_float32(rng_states, threadindex)
    theta = r * math.pi

    # randomly pick a phi between [0, 2*PI]
    r = xoroshiro128p_uniform_float32(rng_states, threadindex)
    phi = r * 2.0 * math.pi

    Q = conversion.V2K * vi * 2.0 * math.sin(0.5 * theta)

    # Debye Waller factor
    dw = math.exp(-dw_core * Q * Q)

    e1 = cuda.local.array(3, dtype=float64)
    e2 = cuda.local.array(3, dtype=float64)
    norm = cuda.local.array(3, dtype=float64)

    # Scattered neutron velocity vector
    vec3.copy(v, e1)
    vec3.normalize(e1)

    # if e1 is not in the z-direction we set e2 to be the cross product of e1 and (0, 0, 1)
    # if e1 is right on the z-direction, that means e1 = (0, 0, 1) and we set e2 = (1, 0, 0)
    if cuda.libdevice.fabs(e1[0]) > epsilon or cuda.libdevice.fabs(e1[1]) > epsilon:
        #norm = [0.0, 0.0, 1.0]
        norm[0] = 0.0
        norm[1] = 0.0
        norm[2] = 1.0
        vec3.cross(norm, e1, e2)
        vec3.normalize(e2)
    else:
        #e2 = [1.0, 0.0, 0.0]
        e2[0] = 1.0
        e2[1] = 0.0
        e2[2] = 0.0
 
    # calculate e3 = e1 * e2
    vec3.cross(e1, e2, norm) # norm = e3

    sint = cuda.libdevice.sin(theta)

    # V_f = sin(theta) * cos(phi) * e2 +
    #       sin(theta) * sin(phi) * e3 + 
    #       cos(theta) * e1

    vec3.scale(e2, sint * cuda.libdevice.cos(phi))     # sin(theta) * cos(phi) * e2
    vec3.scale(norm, sint * cuda.libdevice.sin(phi))   # sin(theta) * sin(phi) * e3
    vec3.scale(e1, cuda.libdevice.cos(theta))          # cos(theta) * e1
    
    # elastic scattering
    neutron[3] = vi * (e2[0] + norm[0] + e1[0])
    neutron[4] = vi * (e2[1] + norm[1] + e1[1])
    neutron[5] = vi * (e2[2] + norm[2] + e1[2])

    # adjust probability of neutron event
    # normalization factor is 2 * PI^2 / 4PI = PI / 2
    neutron[-1] *= sint * (math.pi * 0.5) * dw
