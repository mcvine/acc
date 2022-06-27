from numba import cuda
from mcni import units
from . import location

class LocateFuncFactory:

    def render(self, shape):
        return shape.identify(self)

    def onUnion(self, u):
        s1, s2 = u.shapes
        f1 = s1.identify(self)
        f2 = s2.identify(self)
        @cuda.jit(device=True)
        def locateWrtUnion(x, y, z):
            isoutside = True
            l = f1(x,y,z)
            if l == location.inside:
                return location.inside
            isoutside = isoutside and l == location.outside
            l = f2(x,y,z)
            if l == location.inside:
                return location.inside
            isoutside = isoutside and l == location.outside
            if isoutside:
                return location.outside
            return location.onborder
        return locateWrtUnion

    def onSphere(self, s):
        from . import onsphere
        R = s.radius/units.length.meter
        @cuda.jit(device=True)
        def locateWrtSphere(x, y, z):
            return onsphere.cu_device_locate_wrt_sphere(x,y,z, R)
        return locateWrtSphere

    def onCylinder(self, cyl):
        from . import oncylinder
        R = cyl.radius/units.length.meter
        H = cyl.height/units.length.meter
        @cuda.jit(device=True)
        def locateWrtCylinder(x, y, z):
            return oncylinder.cu_device_locate_wrt_cylinder(x,y,z, R, H)
        return locateWrtCylinder
