# -*- python -*-
#

import math
from numba import cuda
import numba as nb
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states
import time

from mcni.utils.conversion import V2K, SE2V, K2V
from .SourceBase import SourceBase

category = 'sources'

FLOAT = nb.float64

class Source_simple(SourceBase):

    def __init__(
            self, name,
            radius=0.05, height=0, width=0, dist=10.0,
            xw=0.1, yh=0.1,
            E0=60, dE=10, Lambda0=0, dLambda=0,
            flux=1, gauss=0, N=1
    ):
        """
        Initialize this Source_simple component.

        Parameters
        ----------

        radius : float
            Radius in meter of circle in (x,y,0) plane
        height : float
            Height in meter of rectangle in (x,y,0) plane
        width : float
            Width in meter of rectangle in (x,y,0) plane
        dist : float
            Distance in meter to target along z axis.
        xw : float
            Width(x) in meter of target
        yh : float
            Height(y) in meter of target
        E0 : float
            Mean energy of neutrons in meV.
        dE : float
            Energy spread of neutrons (flat or gaussian sigma) in meV.
        Lambda0 : float
            Mean wavelength of neutrons in AA
        dLambda : float
            Wavelength spread of neutrons in AA
        flux : float
            Energy integrated flux in 1/(s*cm^2*sr)
        gauss : bool
            Gaussian (True) or Flat (False) energy/wavelength distribution
        """
        self.name = name
        # Determine source area:
        if (radius != 0 and height == 0 and width == 0) :
            square = False
            srcArea = math.pi*radius*radius
        elif (radius == 0 and height !=0 and width!=0) :
            square = True
            srcArea = width * height
        else :
            msg = f"Source_simple: confused! Both radius ({radius}) and width({width})/height({height}) are specified"
            raise RuntimeError(msg)
        pmul=flux*1e4*srcArea/N
        if (srcArea <= 0 or dist < 0 or xw < 0 or yh < 0):
            raise RuntimeError("bad source geometry")
        if (Lambda0==0 and dLambda==0 and (E0 <= 0 or dE < 0 or E0-dE <= 0)):
            raise RuntimeError("bad energy distribution spec")
        if (E0==0 and dE==0 and (Lambda0 <= 0 or dLambda < 0 or Lambda0-dLambda <= 0)) :
            raise RuntimeError("bad wavelength distribution spec")
        wl_distr = Lambda0!=0
        self.propagate_params = (
            square, width, height, radius,
            wl_distr, Lambda0, dLambda, E0, dE,
            xw, yh, dist, pmul
        )
        import mcni
        neutrons = mcni.neutron_buffer(1)
        self.process(neutrons)


@cuda.jit
def process_kernel_no_buffer(
        rng_states, N, n_neutrons_per_thread,
        square, width, height, radius,
        wl_distr, Lambda0, dLambda, E0, dE,
        xw, yh, dist, pmul,
):
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    neutron = cuda.local.array(shape=10, dtype=FLOAT)
    for i in range(start_index, end_index):
        propagate(
            thread_index, rng_states,
            neutron,
            square, width, height, radius,
            wl_distr, Lambda0, dLambda, E0, dE,
            xw, yh, dist, pmul
        )
    return
Source_simple.process_kernel_no_buffer = process_kernel_no_buffer

@cuda.jit
def process_kernel(
        rng_states, neutrons, n_neutrons_per_thread,
        square, width, height, radius,
        wl_distr, Lambda0, dLambda, E0, dE,
        xw, yh, dist, pmul,
):
    N = len(neutrons)
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    for i in range(start_index, end_index):
        propagate(
            thread_index, rng_states, neutrons[i],
            square, width, height, radius,
            wl_distr, Lambda0, dLambda, E0, dE,
            xw, yh, dist, pmul,
        )
    return
Source_simple.process_kernel = process_kernel

@cuda.jit(device=True)
def propagate(
        threadindex, rng_states,
        in_neutron,
        square, width, height, radius,
        wl_distr, Lambda0, dLambda, E0, dE,
        xw, yh, dist, pmul
):
    r1 = xoroshiro128p_uniform_float32(rng_states, threadindex)
    r2 = xoroshiro128p_uniform_float32(rng_states, threadindex)
    r3 = xoroshiro128p_uniform_float32(rng_states, threadindex)
    r4 = xoroshiro128p_uniform_float32(rng_states, threadindex)
    r5 = xoroshiro128p_uniform_float32(rng_states, threadindex)
    if square:
        x = width * (r1 - 0.5)
        y = height * (r2 - 0.5)
    else:
        chi=2*math.pi*r1
        r=math.sqrt(r2)*radius
        x=r*math.cos(chi)
        y=r*math.sin(chi)
    in_neutron[:3] = x, y, 0.
    # choose final vector
    target = cuda.local.array(shape=3, dtype=FLOAT)
    target[0] = target[1] = 0.0
    target[2] = dist
    vec_f = cuda.local.array(shape=3, dtype=FLOAT)
    solidangle = randvec_target_rect(target, xw, yh, r3, r4, vec_f)
    # vector from moderator to final position is
    # (vec_f[0]-x, vec_f[1]-y, dist)
    dx = vec_f[0]-x; dy = vec_f[1]-y
    dist1 = math.sqrt(dx*dx+dy*dy+dist*dist)
    # velocity scalar
    if wl_distr:
        L = Lambda0+dLambda*(r5*2-1)
        v = K2V*(2*math.pi/L)
    else:
        E = E0+dE*(r5*2-1)
        v = SE2V*math.sqrt(E)
    in_neutron[3:6] = v*dx/dist1, v*dy/dist1, v*dist/dist1
    in_neutron[-2] = 0
    in_neutron[-1] = pmul*solidangle
    return


@cuda.jit(device=True)
def randvec_target_rect(
        target, width, height, rand1, rand2,
        vecout
):
    dx = width*(rand1*2-1)/2.0
    dy = height*(rand2*2-1)/2.0
    dist = len_vec3(target)
    # horiz direction perp to target
    p1 = cuda.local.array(shape=3, dtype=FLOAT)
    vertical = cuda.local.array(shape=3, dtype=FLOAT)
    vertical[0] = vertical[2] = 0.0
    vertical[1] = 1.0
    cross_vec3(target, vertical, p1)
    normalize_vec3(p1)
    # another perp unit vec
    p2 = cuda.local.array(shape=3, dtype=FLOAT)
    cross_vec3(target, p1, p2)
    normalize_vec3(p2)
    scale_vec3(p1, dx)
    scale_vec3(p2, dy)
    tmp = cuda.local.array(shape=3, dtype=FLOAT)
    add_vec3(p1, target, tmp)
    add_vec3(p2, tmp, vecout)
    dist2 = math.sqrt(dx*dx + dy*dy + dist*dist)
    return (width*height*dist)/(dist2*dist2*dist2)


@cuda.jit(device=True, inline=True)
def cross_vec3(v1, v2, vout):
    vout[0] = v1[1]*v2[2]-v1[2]*v2[1]
    vout[1] = v1[2]*v2[0]-v1[0]*v2[2]
    vout[2] = v1[0]*v2[1]-v1[1]*v2[0]
    return

@cuda.jit(device=True, inline=True)
def len_vec3(v):
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

@cuda.jit(device=True, inline=True)
def normalize_vec3(v):
    l = len_vec3(v)
    scale_vec3(v, 1.0/l)
    return

@cuda.jit(device=True, inline=True)
def scale_vec3(v, s):
    v[0]*=s
    v[1]*=s
    v[2]*=s
    return

@cuda.jit(device=True, inline=True)
def add_vec3(v1, v2, v3):
    v3[0]=v1[0]+v2[0]
    v3[1]=v1[1]+v2[1]
    v3[2]=v1[2]+v2[2]
    return

