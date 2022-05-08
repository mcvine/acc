from numba import cuda, void, int64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from math import sqrt, pi, sin, cos

@cuda.jit(device=True)
def scatter(threadindex, rng_states, neutron):
    # randomly pick direction
    cos_t = xoroshiro128p_uniform_float32(rng_states, threadindex)*2-1
    phi = xoroshiro128p_uniform_float32(rng_states, threadindex)*(2*pi)
    if cos_t>1: cos_t = 1
    sin_t = sqrt(1-cos_t*cos_t)
    sin_p, cos_p = sin(phi), cos(phi)
    # compute velocity
    vx, vy, vz = neutron[3:6]
    vi = sqrt(vx*vx+vy*vy+vz*vz)
    vx = vi*sin_t*cos_p
    vy = vi*sin_t*sin_p
    vz = vi*cos_t
    neutron[3:6] = vx,vy,vz
    return


@cuda.jit(device=True)
def scatter(threadindex, rng_states, neutron):
    # randomly pick direction
    theta = xoroshiro128p_uniform_float32(rng_states, threadindex)*pi
    phi = xoroshiro128p_uniform_float32(rng_states, threadindex)*(2*pi)
    cos_t, sin_t = cos(theta), sin(theta)
    sin_p, cos_p = sin(phi), cos(phi)
    # compute velocity
    vx, vy, vz = neutron[3:6]
    vi = sqrt(vx*vx+vy*vy+vz*vz)
    vx = vi*sin_t*cos_p
    vy = vi*sin_t*sin_p
    vz = vi*cos_t
    neutron[3:6] = vx,vy,vz
    neutron[-1] *= sin_t
    return

