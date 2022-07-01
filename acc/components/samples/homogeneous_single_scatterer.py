# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

import numpy as np, numba
from numba import cuda, void, int64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from math import sqrt, exp

from ... import test
from .SampleBase import SampleBase
from ...neutron import absorb, prop_dt_inplace
from ...geometry.arrow_intersect import max_intersections

from numba.core import config
if not config.ENABLE_CUDASIM:
    from numba.cuda.compiler import Dispatcher, DeviceFunction

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()


category = 'samples'

def factory(shape, kernel):
    """
    Usage:
    * create a new python module with code to define a new scatterer class
      that inherits from a base class created using this factory method.
      see tests/components/samples/HSS_isotropic_sphere.py for an example
      - ...load shape and kernel...
      - HSSbase = factory(shape = shape, kernel = None)
      - class HSS(HSSbase): pass
    """
    from ...geometry import arrow_intersect
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)

    class HomogeneousSingleScatterer(SampleBase):

        def __init__(self, name):
            """
            Initialize the isotropicbox component.

            Parameters:
            name (str): the name of this component
            """
            self.name = name
            self.propagate_params = ()

            # Aim neutrons toward the sample to cause JIT compilation.
            import mcni
            neutrons = mcni.neutron_buffer(1)
            neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
            self.process(neutrons)

        @cuda.jit(
            void(int64, xoroshiro128p_type[:], NB_FLOAT[:],
        ) , device=True)
        def propagate(threadindex, rng_states, neutron):
            x, y, z, vx, vy, vz = neutron[:6]
            # loc = locate(x,y,z)
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            ninter = intersect(x,y,z, vx,vy,vz, ts, 0)
            if ninter < 2: return
            if ts[ninter-1] <= 0: return
            rand = xoroshiro128p_uniform_float32(rng_states, threadindex)
            dt = calc_time_to_point_of_scattering(ts, ninter, rand)
            if dt<=0: return
            # propagate to scattering point
            prop_dt_inplace(neutron, dt)
            return

    return HomogeneousSingleScatterer

if test.USE_CUDASIM:
    @cuda.jit(device=True, inline=True)
    def calc_time_to_point_of_scattering(ts, ninter, rand):
        "compute the time to travel to the point for scattering"
        tpartialsums = np.zeros(max_intersections, dtype=np.float64)
        tpartialsums_include_gaps = np.zeros(max_intersections, dtype=np.float64)
        return _calc_time_to_point_of_scattering_impl(
            ts, ninter, rand, tpartialsums, tpartialsums_include_gaps)
else:
    @cuda.jit(device=True, inline=True)
    def calc_time_to_point_of_scattering(ts, ninter, rand):
        "compute the time to travel to the point for scattering"
        tpartialsums = cuda.local.array(max_intersections, dtype=numba.float64)
        tpartialsums_include_gaps = cuda.local.array(max_intersections, dtype=numba.float64)
        return _calc_time_to_point_of_scattering_impl(
            ts, ninter, rand, tpartialsums, tpartialsums_include_gaps)

@cuda.jit(device=True)
def _calc_time_to_point_of_scattering_impl(
        ts, ninter, rand, tpartialsums, tpartialsums_include_gaps):
    assert ninter%2 == 0
    # compute total time in shape and prepare partial sum
    T = 0.0; time_to_enter = -1.0 # is this needed?
    sign = 0
    i_partialsum = 0
    for i in range(ninter):
        t = ts[i]
        if t<0: continue
        if i%2==0 and time_to_enter < 0.0: time_to_enter = t
        if sign == 0:
            # the ts[0] is the first intersection.
            # so for t in ts[2n] and ts[2n+1], particle is inside the shape
            # that means when t just turn positive,
            # it is going to be entering the shape if i is even
            sign = -1 if i%2==0 else 1
        else: sign *= -1
        T += sign * t
        if sign > 0:
            tpartialsums[i_partialsum] = T
            tpartialsums_include_gaps[i_partialsum] = t
            i_partialsum += 1
    total_time_in_shape = T
    if time_to_enter < 0.0: time_to_enter = 0.0
    ttistsp = time_travelled_in_shape_to_scattering_point = rand*total_time_in_shape
    for i in range(i_partialsum):
        if ttistsp < tpartialsums[i]:
            break
    # print(ttistsp, i, tpartialsums, tpartialsums_include_gaps)
    total_time_travel = ttistsp - tpartialsums[i] + tpartialsums_include_gaps[i]
    return total_time_travel

@cuda.jit(device=True)
def total_time_in_shape(ts, ninter):
    "calculate total time a particle inside a shape given time array of intersections"
    assert ninter%2 == 0
    T = 0
    sign = 0
    for i in range(ninter):
        t = ts[i]
        if t<0: continue
        if sign == 0:
            # the ts[0] is the first intersection.
            # so for t in ts[2n] and ts[2n+1], particle is inside the shape
            # that means when t just turn positive,
            # it is going to be entering the shape if i is even
            sign = -1 if i%2==0 else 1
        else: sign *= -1
        T += sign * t
    return T

@cuda.jit(device=True)
def time_to_enter(ts, ninter):
    "calculate time to enter a shape"
    for i in range(ninter):
        t = ts[i]
        if t<0: continue
        # the ts[0] is the first intersection.
        # so for t in ts[2n] and ts[2n+1], particle is inside the shape
        # that means when t just turn positive,
        # it is going to be entering the shape if i is even
        # it is already in the shape if i is odd
        if i%2==0:
            return t
        else:
            return 0
    return 0
