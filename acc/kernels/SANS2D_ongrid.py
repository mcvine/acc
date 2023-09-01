import numpy as np
from mcni.utils import conversion
from mcvine.acc.neutron import v2e, e2v, v2k
from numba import cuda, float64
from numba.core import config
from .._numba import xoroshiro128p_uniform_float32

from .. import vec3
from ..config import get_numba_floattype

NB_FLOAT = get_numba_floattype()

epsilon = 1e-1

@cuda.jit(device=True, inline=True)
def bilinear_interp(x, y, f00, f01, f10, f11):
    a00 = f00
    a10 = f10 - f00
    a01 = f01 - f00
    a11 = f11 - f10 - f01 + f00
    return a00+a10*x+a01*y+a11*x*y

@cuda.jit(device=True)
def S(threadindex, rng_states, neutron, S_QxQy, nx, ny, Qx_min, Qx_max, Qy_min, Qy_max):
    # assume neutron velocity is mostly along z
    v = neutron[3:6]
    # incident neutron velocity
    vi = vec3.length(v)
    # incident neutron momentum
    ki = v2k(vi)
    Qx = xoroshiro128p_uniform_float32(rng_states, threadindex) * (Qx_max-Qx_min)
    Qx += Qx_min
    Qy = xoroshiro128p_uniform_float32(rng_states, threadindex) * (Qy_max-Qy_min)
    Qy += Qy_min

    # final velocity
    v[0] += Qx/ki * vi
    v[1] += Qy/ki * vi
    # weight adjustment
    dQx = (Qx_max-Qx_min)/nx
    dQy = (Qy_max-Qy_min)/ny
    x_index = max(0, int((Qx-Qx_min)/dQx))
    y_index = max(0, int((Qy-Qy_min)/dQy))
    x_index = min(x_index, nx-2)
    y_index = min(y_index, ny-2)
    f00 = S_QxQy[x_index, y_index]
    f10 = S_QxQy[x_index+1, y_index]
    f01 = S_QxQy[x_index, y_index+1]
    f11 = S_QxQy[x_index+1, y_index+1]
    x_fractional = (Qx-Qx_min)/dQx-x_index
    y_fractional = (Qy-Qy_min)/dQy-y_index
    prob = bilinear_interp(x_fractional, y_fractional, f00, f01, f10, f11)
    neutron[-1] *= prob
    return

def makeS(S_QxQy_in, Qx_min, Qx_max, Qy_min, Qy_max):
    # Assume that the input image (y,x). Hence the transpose
    S_QxQy = np.ascontiguousarray(S_QxQy_in.T)

    device_S_QxQy = cuda.to_device(S_QxQy)
    nx, ny = S_QxQy.shape
    @cuda.jit(device=True, inline=True)
    def _S(threadindex, rng_states, neutron):
        S(threadindex, rng_states, neutron, device_S_QxQy, nx, ny, Qx_min, Qx_max, Qy_min, Qy_max)
    return _S

from mccomponents.homogeneous_scatterer.Kernel import Kernel
class SANS2D_ongrid_Kernel(Kernel):

    def __init__(self, S_QxQy, Qx_min, Qx_max, Qy_min, Qy_max):
        self.S_QxQy = S_QxQy
        self.Qx_min, self.Qx_max = Qx_min, Qx_max
        self.Qy_min, self.Qy_max = Qy_min, Qy_max

    def identify(self, visitor):
        return visitor.onSANS2D_ongrid_Kernel(self)
