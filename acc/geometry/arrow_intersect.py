import numpy as np, math
from numba import cuda
from mcni import units
from math import sqrt
from ..vec3 import dot

from . import epsilon, location
inside = location.inside
outside = location.outside
onborder = location.onborder

class ArrowIntersectFuncFactory:

    def render(self, shape):
        return shape.identify(self)

    def onUnion(self, u):
        s1, s2 = u.shapes
        f1 = s1.identify(self)
        f2 = s2.identify(self)
        @cuda.jit(device=True, inline=True)
        def intersectUnion(x, y, z):
            return
        return intersectUnion

    def onSphere(self, s):
        R = s.radius/units.length.meter
        @cuda.jit(device=True, inline=True)
        def intersectSphere(x, y, z):
            return cu_device_intersect_sphere(x,y,z, R)
        return intersectSphere

    def onCylinder(self, cyl):
        R = cyl.radius/units.length.meter
        H = cyl.height/units.length.meter
        @cuda.jit(device=True, inline=True)
        def intersectCylinder(x, y, z):
            return cu_device_intersect_cylinder(x,y,z, R, H)
        return intersectCylinder

# device functions for solid shapes
@cuda.jit(device=True, inline=True)
def cu_device_intersect_sphere(x,y,z, vx,vy,vz, R):
    r0dotv = x*vx+y*vy+z*vz
    r0Xv_x = y*vz - z*vy
    r0Xv_y = z*vx - x*vz
    r0Xv_z = x*vy - y*vx
    v2 = vx*vx+vy*vy+vz*vz
    r0Xv2 = r0Xv_x*r0Xv_x + r0Xv_y*r0Xv_y + r0Xv_z*r0Xv_z
    b2m4ac = v2*R*R - r0Xv2
    t1 = np.nan; t2 = np.nan
    if b2m4ac <0: return t1, t2
    sqrt_b2m4ac = sqrt( b2m4ac )
    t1 = - (r0dotv + sqrt_b2m4ac )/v2
    t2 = - (r0dotv - sqrt_b2m4ac )/v2
    return t1, t2


"""
calculate the time an arrow intersecting 
the side of a cylinder.
the cylinder is decribed by equation
  x^2 + y^2 = r^2
and z is limited in (-h/2, h/2)
"""
@cuda.jit(device=True)
def intersectCylinderSide(x,y,z, vx,vy,vz, R, H):
    a = vx*vx + vy*vy
    b = x*vx + y*vy
    c = x*x+y*y - R*R
    k = b*b-a*c
    hh = H/2.

    if k<0 or a==0: return 0, np.nan, np.nan
    if k==0 :
      t = -b/a
      if abs(z+vz*t)<hh:
          return 1, t, np.nan
    k = math.sqrt(k)
    t1 = (-b+k)/a
    t2 = (-b-k)/a
    goodt1 = abs(z+vz*t1)<hh
    goodt2 = abs(z+vz*t2)<hh
    if goodt1:
        if goodt2:
            return 2, t2, t1
        else:
            return 1, t1, np.nan
    else:
        if goodt2:
            return 1, t2, np.nan
        return 0, np.nan, np.nan

"""
calculate the time an arrow intersecting 
 the top/bottom of a cylinder.
 the cylinder is decribed by equation
   x^2 + y^2 = r^2
 and z is limited in (-h/2, h/2)
"""
@cuda.jit(device=True)
def intersectCylinderTopBottom(x,y,z, vx,vy,vz, R, H):
    if vz == 0: return 0, np.nan, np.nan
    hh = H/2
    r2 = R*R

    t1 = (hh-z)/vz
    x1 = x + vx*t1
    y1 = y + vy*t1
    goodt1 = x1*x1 + y1*y1 <= r2
    t2 = (-hh-z)/vz
    x1 = x + vx*t2
    y1 = y + vy*t2
    goodt2 = x1*x1 + y1*y1 <= r2

    if goodt1:
        if goodt2:
            if t1 < t2:
                return 2, t1, t2
            else:
                return 2, t2, t1
        else:
            return 1, t1, np.nan
    else:
        if goodt2:
            return 1, t2, np.nan
        return 0, np.nan, np.nan

@cuda.jit(device=True, inline=True)
def cu_device_intersect_cylinder(x,y,z, vx,vy,vz, R, H):
    n1, t1, t2 = intersectCylinderSide(x,y,z, vx,vy,vz, R, H)
    n2, t3, t4 = intersectCylinderTopBottom(x,y,z, vx,vy,vz, R, H)
    n = n1 + n2
    if n==0: np.nan, np.nan
    if n==1:
        if n1==1:
            return t1, t1
        else:
            return t3, t3
    elif n==2:
        if n1==1:
            if t1>t3: return t3,t1
            return t1, t3
        elif n1==0:
            return t3, t4
        else:
            return t1, t2
    elif n==3:
        if n1==1:
            return min(t1, t3), max(t1, t4)
        else:
            return min(t3, t1), max(t3, t2)
    else: # n=4
        return min(t1, t3), max(t2, t4)
    return
