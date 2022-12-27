import numpy as np
from mcni.utils import conversion
from mcvine.acc.neutron import v2e, e2v
from numba import cuda, float64
from numba.core import config
from .._numba import xoroshiro128p_uniform_float32

from .. import vec3
from ..config import get_numba_floattype

NB_FLOAT = get_numba_floattype()

epsilon = 1e-1


def makeS(E_Q, S_Q, Qmin, Qmax, max_iter=100):
    from math import (
        exp, ceil, fabs, factorial, floor, log, log10, pow, sqrt,
        sin, cos, tan, asin, acos, atan2,
        sinh, cosh, tanh, asinh, acosh, atanh,
        erf, erfc,
        pi, e
    )
    E_Q = E_Q.replace('^', '**')
    S_Q = S_Q.replace('^', '**')
    context = locals().copy()
    code = 'def E_Q_func(Q): return ' + E_Q
    exec(code, context, context)
    E_Q_func = context['E_Q_func']
    code = 'def S_Q_func(Q): return ' + S_Q
    exec(code, context, context)
    S_Q_func = context['S_Q_func']
    # print(E_Q_func(10), S_Q_func(10))
    S_Q_df = cuda.jit(device=True)(S_Q_func)
    E_Q_df = cuda.jit(device=True)(E_Q_func)

    @cuda.jit(device=True)
    def _S(threadindex, rng_states, neutron, e1,e2,e3,norm):
        v = neutron[3:6]
        # incident neutron velocity
        vi = vec3.length(v)
        # incident neutron energy
        Ei = v2e(vi)
        found = False
        for itr in range(max_iter):
            Q = xoroshiro128p_uniform_float32(rng_states, threadindex) * (Qmax-Qmin)
            Q += Qmin
            E = E_Q_df(Q)
            # final energy
            Ef = Ei - E
            if Ef < 0: continue
            # final velocity
            vf = e2v(Ef)
            ki = conversion.V2K * vi
            kf = conversion.V2K * vf
            cost = (ki * ki + kf * kf - Q * Q) / (2 * ki * kf)
            cost2 = cost*cost
            if cost2>1: continue
            found = True
            break
        if not found:
            return
        sint = sqrt(1.0 - cost2)
        phi = xoroshiro128p_uniform_float32(rng_states, threadindex) * 2.0 * np.pi

        # scattered neutron velocity
        v2 = vf * sint * cos(phi)
        v3 = vf * sint * sin(phi)
        v1 = vf * cost

        # e1
        vec3.copy(v, e1); vec3.normalize(e1)
        # e2
        if fabs(e1[0]) > epsilon or fabs(e1[1]) > epsilon:
            norm[0] = 0
            norm[1] = 0
            norm[2] = 1
        else:
            norm[0] = 1
            norm[1] = 0
            norm[2] = 0
        vec3.cross(norm, e1, e2)
        vec3.normalize(e2)
        # e3
        vec3.cross(e1, e2, e3)  # ey

        vec3.scale(e1, v1)
        vec3.scale(e2, v2)
        vec3.scale(e3, v3)

        # final velocity
        neutron[3] = e1[0] + e2[0] + e3[0]
        neutron[4] = e1[1] + e2[1] + e3[1]
        neutron[5] = e1[2] + e2[2] + e3[2]

        prob_f = S_Q_df(Q) * (vf/vi) * Q*(Qmax-Qmin) / (kf*ki) /2
        neutron[-1] *= prob_f# /(itr+1)
        return

    if config.ENABLE_CUDASIM:
        @cuda.jit(device=True)
        def S(threadindex, rng_states, neutron):
            e1 = np.zeros(3, dtype=np.float64)
            e2 = np.zeros(3, dtype=np.float64)
            e3 = np.zeros(3, dtype=np.float64)
            norm = np.zeros(3, dtype=np.float64)
            return _S(threadindex, rng_states, neutron, e1,e2,e3,norm)
    else:
        @cuda.jit(device=True)
        def S(threadindex, rng_states, neutron):
            e1 = cuda.local.array(3, dtype=float64)
            e2 = cuda.local.array(3, dtype=float64)
            e3 = cuda.local.array(3, dtype=float64)
            norm = cuda.local.array(3, dtype=float64)
            return _S(threadindex, rng_states, neutron, e1,e2,e3,norm)
    return S
