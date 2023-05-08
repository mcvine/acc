import numpy as np, numba
from numba import cuda
from numba.core.config import ENABLE_CUDASIM
from .arrow_intersect import max_intersections
from .arrow_intersect import inside, outside, onborder
from ..neutron import prop_dt_inplace
from ..vec3 import distance as v3_dist, length as v3_length
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
    def _is_exiting(neutron, ts):
        """is the neutron moving out of the shape?"""
        x,y,z, vx,vy,vz = neutron[:6]
        loc = locate(x,y,z)
        # if event is still inside the shape, it is not exiting
        if loc == inside: return False
        # if no intersections, it is already exiting
        N = forward_intersect(x,y,z, vx,vy,vz, ts)
        if N==0: return True
        # if event is outside the shape, but will hit the shape again, it
        # does not count as exiting
        if loc == outside: return False
        # if we reach here, the neutron is on surface, need to check the intersections.
        # if any intersection is not negligible, then it is not yet exiting
        previous = 0.
        for it in range(N):
            now = ts[it]
            middle = (previous+now)/2.
            midx = x + vx*middle
            midy = y + vy*middle
            midz = z + vz*middle
            if locate(midx, midy, midz) != onborder:
                return False
            previous = now
            continue
        return True

    @cuda.jit(device=True)
    def _propagate_out(neutron, ts):
        """propagate a neutron out of a shape"""
        x,y,z,vx,vy,vz = neutron[:6]
        N = forward_intersect(x,y,z, vx,vy,vz, ts)
        if N==0: return
        prop_dt_inplace(neutron, ts[N-1])
        return

    @cuda.jit(device=True)
    def _propagate_to_next_incident_surface(neutron, ts):
        """
        propagate a neutron to the next incident-surface of a shape
        please notice that a neutorn could go through a shape in/out
        several times. For example, a neutron can go through a
        hollow cylinder by entering/exiting it twice (one at the
        front surface, and another at the back surface.
        note: shape cannot be infinitely large.
        note: point must be out of shape, or it may be at an exiting
              surface.
        note: if the starting point is already on an
              incident surface, nothing will be done.
        """
        x,y,z,vx,vy,vz = neutron[:6]
        loc = locate(x,y,z)
        if loc==inside:
            if ENABLE_CUDASIM:
                e = "_propagate_to_next_incident_surface only valid for neutrons outside the shape"
                raise RuntimeError(e)
            return
        N = forward_intersect(x,y,z, vx,vy,vz, ts)
        if N==0: return
        if loc == outside:
            t  = ts[0]
        elif loc == onborder:
            # find the first intersection that is not the same point as the starting point
            found = False
            for i in range(N):
                midx = x+ts[i]/2*vx
                midy = y+ts[i]/2*vy
                midz = z+ts[i]/2*vz
                loc1 = locate(midx,midy,midz)
                if loc1 == inside:
                    # the starting point is already the incident surface
                    return
                elif loc1 == outside:
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
        """
        propagate a neutron to the next out-surface of a shape
        please notice that a neutorn could go through a shape in/out
        several times. For example, a neutron can go through a
        hollow cylinder by entering/exiting it twice (one at the
        front surface, and another at the back surface.
        note: shape cannot be infinitely large.
        note: the starting point must be either
          1. inside the shape
          2. outside the shape
          3. on the input surface of the shape
        If the starting point is already on the exiting surface of the shape,
        nothing will be done.
        """
        x,y,z,vx,vy,vz = neutron[:6]
        loc = locate(x,y,z)
        if loc == inside:
            # the next intersection should be the one
            N = forward_intersect(x,y,z, vx,vy,vz, ts)
            # N must be positive
            t = ts[0]
        else:
            if loc == outside:
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
                if loc1 == outside:
                    # starting point is exiting surface
                    return
                elif loc1 == inside:
                    t = ts[i]
                    found = True
                    break
            if not found: return
        prop_dt_inplace(neutron, t)
        return

    @cuda.jit(device=True)
    def _tof_before_first_exit(neutron, ts):
        """
        calcualte the total tof of neutron for it to exit
        the given shape for the first time.
        please notice that a neutorn could go through a shape in/out
        several times. For example, a neutron can go through a 
        hollow cylinder by entering/exiting it twice (one at the
        front surface, and another at the back surface.
        note: shape cannot be infinitely large.
        note: point must be inside the shape, or it may be at an incident
        surface.
        """
        x,y,z,vx,vy,vz = neutron[:6]
        loc = locate(x,y,z)
        if loc == outside:
            if ENABLE_CUDASIM:
                e = '({},{},{}) is out of shape'.format(x,y,z)
                raise RuntimeError(e)
            return 0.
        N = forward_intersect(x,y,z, vx,vy,vz, ts)
        if N == 0:
            return 0.
        if loc == inside:
            return ts[0]
        t = 0.
        for i in range(N):
            midx = x+ts[i]/2*vx
            midy = y+ts[i]/2*vy
            midz = z+ts[i]/2*vz
            loc1 = locate(midx,midy,midz)
            if loc1 == outside:
                return 0.
            elif loc1 == inside:
                t = ts[i]
                break
            continue
        return t

    @cuda.jit(device=True)
    def _forward_distance_in_shape(neutron, end, ts):
        "distance to travel in shape to get to `end`. `end` must be on the path of the neutron"
        start = neutron[:3]
        vv = neutron[3:6]
        v = v3_length(vv)
        nintersect = forward_intersect(start[0], start[1], start[2], vv[0], vv[1], vv[2], ts)
        length = v3_dist(start, end)
        tofmax = length/v;
        prev = 0; length = 0
        for i in range(nintersect):
            tof = ts[i]
            if (tof > tofmax):
                tof = tofmax
            middle = (tof+prev)/2.
            # middle point p = start + vv*middle;
            x = start[0] + vv[0]*middle
            y = start[1] + vv[1]*middle
            z = start[2] + vv[2]*middle
            # if middle point is inside, count this segment
            if locate(x,y,z) == inside:
                length += (tof-prev) * v;
            if tof > tofmax:
                break
            prev = tof
            continue
        return length


    if test.USE_CUDASIM:
        @cuda.jit(device=True, inline=True)
        def is_exiting(neutron):
            ts = np.zeros(max_intersections, dtype=float)
            return _is_exiting(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def propagate_out(neutron):
            ts = np.zeros(max_intersections, dtype=float)
            return _propagate_out(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def tof_before_first_exit(neutron):
            ts = np.zeros(max_intersections, dtype=float)
            return _tof_before_first_exit(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def propagate_to_next_incident_surface(neutron):
            ts = np.zeros(max_intersections, dtype=float)
            return _propagate_to_next_incident_surface(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def propagate_to_next_exiting_surface(neutron):
            ts = np.zeros(max_intersections, dtype=float)
            return _propagate_to_next_exiting_surface(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def forward_distance_in_shape(neutron, end):
            ts = np.zeros(max_intersections, dtype=float)
            return _forward_distance_in_shape(neutron, end, ts)
    else:
        @cuda.jit(device=True, inline=True)
        def is_exiting(neutron):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _is_exiting(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def propagate_out(neutron):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _propagate_out(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def tof_before_first_exit(neutron):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _tof_before_first_exit(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def propagate_to_next_incident_surface(neutron):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _propagate_to_next_incident_surface(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def propagate_to_next_exiting_surface(neutron):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _propagate_to_next_exiting_surface(neutron, ts)
        @cuda.jit(device=True, inline=True)
        def forward_distance_in_shape(neutron, end):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _forward_distance_in_shape(neutron, end, ts)

    return dict(
        is_exiting = is_exiting,
        forward_intersect = forward_intersect,
        tof_before_first_exit = tof_before_first_exit,
        propagate_out = propagate_out,
        propagate_to_next_incident_surface = propagate_to_next_incident_surface,
        propagate_to_next_exiting_surface = propagate_to_next_exiting_surface,
        forward_distance_in_shape = forward_distance_in_shape,
    )
