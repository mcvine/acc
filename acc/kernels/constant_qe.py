import numpy as np
from numba import cuda, float64
from numba.cuda.random import xoroshiro128p_uniform_float32

from .. import vec3
from ..config import get_numba_floattype

NB_FLOAT = get_numba_floattype()

epsilon = 1e-1

neutron_mass = 1.6749286e-27
electron_charge = 1.60217733e-19
hbar = 1.054571628e-34

vsq2e = neutron_mass / (2.0e-3 * electron_charge)
sqrte2v = np.sqrt((2.0e-3 * electron_charge) / neutron_mass)
# neutron wave vector k (AA^-1) to velocity (m/s)
k2v = hbar / neutron_mass * 1.0e10
# neutron velocity (m/s) to wave vector k (AA^-1)
v2k = 1.0 / k2v


@cuda.jit(NB_FLOAT(NB_FLOAT), device=True, inline=True)
def v2E(v):
    return v * v * vsq2e


@cuda.jit(NB_FLOAT(NB_FLOAT), device=True, inline=True)
def E2v(e):
    return cuda.libdevice.sqrt(e) * sqrte2v


@cuda.jit(device=True)
def S(threadindex, rng_states, neutron, Q, E):
    """
    Scatters to a pre-determined (Q, E)
    """

    v = neutron[3:6]
    # incident neutron velocity
    vi = vec3.length(v)
    # incident neutron energy
    Ei = v2E(vi)
    # final energy
    Ef = Ei - E
    # final velocity
    vf = E2v(Ef)

    ki = v2k * vi
    kf = v2k * vf

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
