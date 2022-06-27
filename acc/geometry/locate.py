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
        s1, s2 = u.shapes
        f1 = s1.identify(self)
        f2 = s2.identify(self)
        @cuda.jit(device=True, inline=True)
        def locateWrtUnion(x, y, z):
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

