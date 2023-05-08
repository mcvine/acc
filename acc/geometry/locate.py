import numpy as np
from numba import cuda
from mcni import units
from . import epsilon, location
inside = location.inside
outside = location.outside
onborder = location.onborder

class LocateFuncFactory:

    def render(self, shape):
        return shape.identify(self)

    def onUnion(self, u):
        nelements = len(u.shapes)
        if nelements == 2: return self.onUnion2(u)
        if nelements == 3:
            from .composite_3 import createUnionLocateMethod_3
            return createUnionLocateMethod_3(u.shapes)
        raise NotImplementedError(f"locate for union of {nelements} elements")

    def onUnion2(self, u):
        s1, s2 = u.shapes
        f1 = s1.identify(self)
        f2 = s2.identify(self)
        @cuda.jit(device=True, inline=True)
        def locateWrtUnion(x, y, z):
            return cu_device_locate_wrt_union(x,y,z, f1,f2)
        return locateWrtUnion

    def onSphere(self, s):
        R = s.radius/units.length.meter
        @cuda.jit(device=True, inline=True)
        def locateWrtSphere(x, y, z):
            return cu_device_locate_wrt_sphere(x,y,z, R)
        return locateWrtSphere

    def onCylinder(self, cyl):
        R = cyl.radius/units.length.meter
        H = cyl.height/units.length.meter
        @cuda.jit(device=True, inline=True)
        def locateWrtCylinder(x, y, z):
            return cu_device_locate_wrt_cylinder(x,y,z, R, H)
        return locateWrtCylinder

    def onBlock(self, b):
        W = b.width / units.length.meter
        H = b.height / units.length.meter
        D = b.thickness / units.length.meter

        @cuda.jit(device=True, inline=True)
        def locateWrtBlock(x, y, z):
            return cu_device_locate_wrt_box(x, y, z, W, H, D)
        return locateWrtBlock

    def onTranslation(self, t):
        x0,y0,z0 = [_/units.length.meter for _ in t.vector]
        body = t.body
        locate_in_body = self.render(body)
        @cuda.jit(device=True, inline=True)
        def locateWrtTranslation(x,y,z):
            return locate_in_body(x-x0, y-y0, z-z0)
        return locateWrtTranslation

    def onRotation(self, r):
        euler_angles = [_/units.angle.rad for _ in r.euler_angles]
        # hack
        if not np.allclose(euler_angles, [0,0,0]):
            raise NotImplementedError
        return self.render(r.body)

    def onDifference(self, s):
        f1 = s.op1.identify(self)
        f2 = s.op2.identify(self)

        @cuda.jit(device=True, inline=True)
        def locateWrtDifference(x, y, z):
            return cu_device_locate_wrt_difference(x, y, z, f1, f2)
        return locateWrtDifference

    def onIntersection(self, u):
        s1, s2 = u.shapes
        f1 = s1.identify(self)
        f2 = s2.identify(self)
        @cuda.jit(device=True, inline=True)
        def locateWrtIntersection(x, y, z):
            return cu_device_locate_wrt_intersection(x,y,z, f1,f2)
        return locateWrtIntersection

# device functions for operations
@cuda.jit(device=True)
def cu_device_locate_wrt_union(x, y, z, f1, f2):
    "f1 and f2 are locate methods of elements in the union"
    isoutside = True
    l = f1(x,y,z)
    if l == inside:
        return inside
    isoutside = isoutside and l == outside
    l = f2(x,y,z)
    if l == inside:
        return inside
    isoutside = isoutside and l == outside
    if isoutside:
        return outside
    return onborder

# device functions for solid shapes
@cuda.jit(device=True, inline=True)
def cu_device_locate_wrt_sphere(x,y,z, R):
    dist2 = x*x+y*y+z*z
    if dist2>(R+epsilon)*(R+epsilon): return outside
    elif dist2<(R-epsilon)*(R-epsilon): return inside
    return onborder

@cuda.jit(device=True, inline=True)
def cu_device_locate_wrt_cylinder(x,y,z, R, H):
    if ( abs(z)-H/2. > epsilon ) :
        return outside

    r2 = x*x+y*y
    if r2 > (R+epsilon)*(R+epsilon):
        return outside

    if H/2.-abs(z) > epsilon and r2 < (R-epsilon)*(R-epsilon):
        return inside
    return onborder

@cuda.jit(device=True, inline=True)
def cu_device_locate_wrt_box(x, y, z, W, H, D):
    if (abs(x) - 0.5 * W > epsilon):
        return outside

    if (abs(y) - 0.5 * H > epsilon):
        return outside

    if (abs(z) - 0.5 * D > epsilon):
        return outside

    # check whether point is within bounds
    if ((-0.5 * W < x < 0.5 * W) and
            (-0.5 * H < y < 0.5 * H) and
            (-0.5 * D < z < 0.5 * D)):
        return inside

    return onborder


@cuda.jit(device=True, inline=True)
def cu_device_locate_wrt_difference(x, y, z, f1, f2):
    p1 = f1(x, y, z)
    p2 = f2(x, y, z)
    if p1 == outside or p2 == inside:
        return outside
    elif p1 == onborder or p2 == onborder:
        return onborder
    else:
        return inside


@cuda.jit(device=True)
def cu_device_locate_wrt_intersection(x, y, z, f1, f2):
    "f1 and f2 are locate methods of elements in the intersection"
    isinside = True
    l = f1(x, y, z)
    if l == outside:
        return outside
    isinside = isinside and l == inside
    l = f2(x, y, z)
    if l == outside:
        return outside
    isinside = isinside and l == inside
    if isinside:
        return inside
    return onborder
