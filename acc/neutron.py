#!/usr/bin/env python

from math import isnan
from numba import boolean, cuda, void

from . import vec3

from mcvine.acc.config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

@cuda.jit(device=True, inline=True)
def abs2rel(r, v, rotmat, offset, rtmp, vtmp):
    vec3.copy(r, rtmp); vec3.copy(v, vtmp)
    vec3.abs2rel(rtmp, rotmat, offset, r)
    vec3.mXv(rotmat, vtmp, v)


@cuda.jit(void(NB_FLOAT[:]),
          device=True, inline=True)
def absorb(neutron):
    neutron[-1] = -1


@cuda.jit(boolean(NB_FLOAT[:]),
          device=True, inline=True)
def is_absorbed(neutron):
    prob = neutron[-1]
    return prob <= 0 and not isnan(prob)


@cuda.jit(device=True, inline=True)
def prop_dt_inplace(neutron, dt):
    "propagate neutron by dt"
    x,y,z,vx,vy,vz = neutron[:6]
    t = neutron[-2]
    neutron[:3] = x+vx*dt, y+vy*dt, z+vz*dt
    neutron[-2] = t+dt
    return

@cuda.jit(device=True, inline=True)
def prop_dt(neutron, dt):
    "propagate neutron by dt; return x,y,z,t"
    x,y,z,vx,vy,vz = neutron[:6]
    t = neutron[-2]
    return x+vx*dt, y+vy*dt, z+vz*dt, t+dt

@cuda.jit(device=True, inline=True)
def prop_z0(neutron):
    "propagate neutron to z=0; return x,y,t"
    x,y,z,vx,vy,vz = neutron[:6]
    t = neutron[-2]
    dt = -z/vz
    return x+vx*dt, y+vy*dt, 0.0, t+dt
