from numba import boolean, cuda, void
from math import isnan, tanh
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()

@cuda.jit(NB_FLOAT(NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
                   NB_FLOAT),
          device=True, inline=True)
def calc_reflectivity(Q, R0, Qc, alpha, m, W):
    """
    Calculate the mirror reflectivity for a neutron.

    Returns:
    float: the reflectivity for the neutron's given momentum change
    """
    R = R0
    if Q > Qc:
        tmp = (Q - m * Qc) / W
        if tmp < 10:
            R *= (1 - tanh(tmp)) * (1 - alpha * (Q - Qc)) / 2
        else:
            R = 0
    return R


@cuda.jit(void(NB_FLOAT[:]),
          device=True, inline=True)
def absorb(neutron):
    neutron[-1] = -1


@cuda.jit(boolean(NB_FLOAT[:]),
          device=True, inline=True)
def is_absorbed(neutron):
    prob = neutron[-1]
    return prob <= 0 and not isnan(prob)
