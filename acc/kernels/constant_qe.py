import numpy as np
from mcni.utils import conversion
from mcvine.acc.neutron import v2e, e2v
from numba import cuda, float64
from numba.cuda.random import xoroshiro128p_uniform_float32

from .. import vec3
from ..config import get_numba_floattype

NB_FLOAT = get_numba_floattype()

epsilon = 1e-1


@cuda.jit(device=True)
def S(threadindex, rng_states, neutron, Q, E):
    """
    Scatters to a pre-determined (Q, E)
    """

    v = neutron[3:6]
    # incident neutron velocity
    vi = vec3.length(v)
    # incident neutron energy
    Ei = v2e(vi)
    # final energy
    Ef = Ei - E
    # final velocity
    vf = e2v(Ef)

    ki = conversion.V2K * vi
    kf = conversion.V2K * vf

    cost = (ki * ki + kf * kf - Q * Q) / (2 * ki * kf)
    sint = cuda.libdevice.sqrt(1.0 - cost * cost)
    phi = xoroshiro128p_uniform_float32(rng_states, threadindex) * 2.0 * np.pi

    # scattered neutron velocity
    vx = vf * sint * cuda.libdevice.cos(phi)
    vy = vf * sint * cuda.libdevice.sin(phi)
    vz = vf * cost

    vtmp = cuda.local.array(3, dtype=float64)
    norm = cuda.local.array(3, dtype=float64)

    vec3.copy(v, vtmp)
    vec3.normalize(vtmp)

    if cuda.libdevice.fabs(vtmp[0]) > epsilon or cuda.libdevice.fabs(vtmp[1]) > epsilon:
        norm[0] = 0
        norm[1] = 0
        norm[2] = 1
    else:
        norm[0] = 1
        norm[1] = 0
        norm[2] = 0

    vec3.cross(norm, vtmp, v)
    vec3.normalize(v)  # ex

    vec3.cross(vtmp, v, norm)  # ey

    vec3.scale(v, vx)
    vec3.scale(norm, vy)
    vec3.scale(vtmp, vz)

    # final velocity
    neutron[3] = v[0] + norm[0] + vtmp[0]
    neutron[4] = v[1] + norm[1] + vtmp[1]
    neutron[5] = v[2] + norm[2] + vtmp[2]
