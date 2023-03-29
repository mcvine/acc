import numpy as np
from mcvine.acc.neutron import v2e, v2k, e2k
from numba import cuda, float64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from typing import List
from mcni.utils.conversion import K2V
import math

from .. import vec3
from ..config import get_numba_floattype

NB_FLOAT = get_numba_floattype()

epsilon = 1e-7


@cuda.jit(device=True)
def scatter(threadindex, rng_states, neutron, qmin, qmax, emin, emax, sqe, nq, ne):
    """
    S(Q, E) scattering kernel where Q is scalar
    """

    vi = vec3.length(neutron[3:6])
    Ei = v2e(vi)
    ki = v2k * vi

    if emin > Ei:
        return

    e_max = min(Ei, emax)
    # randomly pick energy transfer
    Er = xoroshiro128p_uniform_float32(threadindex, rng_states)
    E = Er * (e_max - emin) + emin
    # use the same random number [0,1] and map to histogram size of [0, len(sqehist.energy) - 1]. See comment below for Q.
    Eind = int(Er * int(ne))

    # final energy, wave vector
    Ef = Ei - E
    kf = e2k(Ef)

    # randomly pick momentum transfer
    # Q must satisfy abs(ki - kf) < Q < ki + kf and Qmin < Q < Qmax
    q_min = max(qmin, abs(ki - kf))
    q_max = min(qmax, ki + kf)

    if q_max < q_min:
        return

    Qr = xoroshiro128p_uniform_float32(threadindex, rng_states)
    Q = Qr * (q_max - q_min) + q_min
    # TODO: check this - the CPU mcvine looks up sqehist[Q,E] using float values. Since the histogram data seems to have Q,E values ordered,
    # this is converting the same random number (Qr) used in Q from [Qmin,Qmax] and mapping it an integer index from [0,len(SQE.Q_axis)-1] to try and find the closest Q value for that index.
    # Linear interpolation might be more appropriate here?
    # Refer to the mcvine implementation of https://github.com/mcvine/mcvine/blob/682e72543472e09a10f3df5d4e5aca895e31abde/packages/mccomponents/lib/kernels/sample/SQE/fxy.h#L26-L27
    Qind = int(Qr * int(nq))

    # adjust neutron probability
    neutron[-1] *= sqe[Qind, Eind] * Q * \
        (q_max - q_min) * (e_max - emin) / (2 * ki * ki)

    costheta = (kf * kf + ki * ki - Q*Q) / 2.0 / kf / ki
    sintheta = math.sqrt(1.0 - costheta * costheta)

    phi = xoroshiro128p_uniform_float32(threadindex, rng_states) * 2 * math.pi

    cosphi = math.cos(phi)
    sinphi = math.sin(phi)

    e1 = cuda.local.array(3, dtype=float64)
    e2 = cuda.local.array(3, dtype=float64)
    e3 = cuda.local.array(3, dtype=float64)

    e1 = neutron[3:6]
    vec3.normalize(e1)
    if math.fabs(e1[0]) > epsilon or math.fabs(e1[1]) > epsilon:
        e2[0] = 0.0
        e2[1] = 0.0
        e2[2] = 1.0
        vec3.cross(e2, e1, e2)
        vec3.normalize(e2)
    else:
        e2[0] = 1.0
        e2[1] = 0.0
        e2[2] = 0.0

    vec3.cross(e1, e2, e3)

    vec3.scale(e1, costheta)
    vec3.scale(e2, sintheta*cosphi)
    vec3.scale(e3, sintheta*sinphi)

    # ekf = e3
    vec3.add(e3, e2, e3)
    vec3.add(e3, e1, e3)

    vec3.scale(e3, kf*K2V)

    # set vf
    neutron[3] = e3[0]
    neutron[4] = e3[1]
    neutron[5] = e3[2]
