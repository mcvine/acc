#!/usr/bin/env python

from numba import cuda, int64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type

from mcvine.acc.config import get_numba_floattype
NB_FLOAT = get_numba_floattype()


@cuda.jit(device=True, inline=True)
def randrange(rngstates, threadindex, min, max):
    r = xoroshiro128p_uniform_float32(rngstates, threadindex)
    return r * (max - min) + min

@cuda.jit(NB_FLOAT(xoroshiro128p_type[:], int64), device=True, inline=True)
def rand01(rngstates, threadindex):
    return xoroshiro128p_uniform_float32(rngstates, threadindex)

@cuda.jit(NB_FLOAT(xoroshiro128p_type[:], int64), device=True, inline=True)
def randpm1(rngstates, threadindex):
    return 2.0 * xoroshiro128p_uniform_float32(rngstates, threadindex) - 1.0

@cuda.jit(NB_FLOAT(xoroshiro128p_type[:], int64), device=True, inline=True)
def randnorm(rngstates, threadindex):
    return 2.0 * xoroshiro128p_uniform_float32(rngstates, threadindex) - 1.0
