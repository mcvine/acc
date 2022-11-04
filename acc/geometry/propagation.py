import numpy as np
from numba import cuda
from .arrow_intersect import max_intersections
from . import location
from ..neutron import prop_dt_inplace
from .. import test

def makePropagateMethods(intersect, locate):

    @cuda.jit(device=True)
    def forward_intersect(x,y,z, vx,vy,vz, ts):
        N = intersect(x,y,z, vx,vy,vz, ts)
        Nm = N
        for i in range(N):
            if ts[i] >= 0:
                Nm = i
                break
        Np = N - Nm
        for i in range(Np):
            ts[i] = ts[i+Nm]
        return Np

    @cuda.jit(device=True)
    def _propagate_out(neutron, ts):
        x,y,z,vx,vy,vz = neutron[:6]
        N = forward_intersect(x,y,z, vx,vy,vz, ts)
        if N==0: return
        prop_dt_inplace(neutron, ts[N-1])
        return

    @cuda.jit(device=True)
    def _propagate_to_next_incident_surface(neutron, ts):
        # this method should only be called if neutron is not inside the shape
        # check it before calling this method
        x,y,z,vx,vy,vz = neutron[:6]
        if locate(x,y,z)==location.inside:
            raise RuntimeError("_propagate_to_next_incident_surface only valid for neutrons outside the shape")
        N = forward_intersect(x,y,z, vx,vy,vz, ts)
        if N==0: return
        loc = locate(x,y,z)
        if loc == location.outside:
            t  = ts[0]
        elif loc == location.onborder:
            # find the first intersection that is not the same point as the starting point
            found = False
            for i in range(N):
                midx = x+ts[i]/2*vx
                midy = y+ts[i]/2*vy
                midz = z+ts[i]/2*vz
                loc1 = locate(midx,midy,midz)
                if loc1 == location.inside:
                    # the starting point is already the incident surface
                    return
                elif loc1 == location.outside:
                    t = ts[i]
                    found = True
                    break
            if not found: return
        else:
            return # should not reach here
        prop_dt_inplace(neutron, t)
        return

    @cuda.jit(device=True)
    def _propagate_to_next_exiting_surface(neutron, ts):
        x,y,z,vx,vy,vz = neutron[:6]
        loc = locate(x,y,z)
        if loc == location.inside:
            # the next intersection should be the one
            N = forward_intersect(x,y,z, vx,vy,vz, ts)
            # N must be positive
            t = ts[0]
        else:
            if loc == location.outside:
                propagate_to_next_incident_surface(neutron)
                x,y,z,vx,vy,vz = neutron[:6]
            # now we are at the surface
            # find the first intersection that is not the same point as the current point
            N = forward_intersect(x,y,z, vx,vy,vz, ts)
            found = False
            for i in range(N):
                midx = x+ts[i]/2*vx
                midy = y+ts[i]/2*vy
                midz = z+ts[i]/2*vz
                loc1 = locate(midx, midy, midz)
                if loc1 == location.outside:
                    # starting point is exiting surface
                    return
                elif loc1 == location.inside:
                    t = ts[i]
                    found = True
                    break
            if not found: return
        prop_dt_inplace(neutron, t)
        return

    if test.USE_CUDASIM:
        @cuda.jit(device=True, inline=True)
        def propagate_out(neutron):
            ts = np.zeros(max_intersections, dtype=float)
            return _propagate_out(neutron, ts)
        def propagate_to_next_incident_surface(neutron):
            ts = np.zeros(max_intersections, dtype=float)
            return _propagate_to_next_incident_surface(neutron, ts)
        def propagate_to_next_exiting_surface(neutron):
            ts = np.zeros(max_intersections, dtype=float)
            return _propagate_to_next_exiting_surface(neutron, ts)
    else:
        @cuda.jit(device=True, inline=True)
        def propagate_out(neutron):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _propagate_out(neutron, ts)
        def propagate_to_next_incident_surface(neutron):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _propagate_to_next_incident_surface(neutron, ts)
        def propagate_to_next_exiting_surface(neutron):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _propagate_to_next_exiting_surface(neutron, ts)

    return forward_intersect, propagate_out, propagate_to_next_incident_surface, propagate_to_next_exiting_surface
