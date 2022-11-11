#!/usr/bin/env python

from math import sqrt, isnan
from mcni.utils import conversion
from numba import boolean, cuda, void

from . import vec3

from mcvine.acc.config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

@cuda.jit(device=True, inline=True)
def clone(in_neutron, out_neutron):
    for i in range(10):
        out_neutron[i] = in_neutron[i]
    return

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

MIN_VELOCITY = 1.e-6      # typical thermal neutron are 10^3 m/s
@cuda.jit(boolean(NB_FLOAT[:]), device=True, inline=True)
def is_moving(neutron):
    vx,vy,vz = neutron[3:6]
    v = sqrt(vx*vx+vy*vy+vz*vz)
    return v>MIN_VELOCITY

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


@cuda.jit(NB_FLOAT(NB_FLOAT), device=True, inline=True)
def v2e(v):
    return v * v * conversion.VS2E


@cuda.jit(NB_FLOAT(NB_FLOAT), device=True, inline=True)
def e2v(e):
    return sqrt(e) * conversion.SE2V
