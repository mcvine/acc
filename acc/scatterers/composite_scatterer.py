# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

import numpy as np, numba
from numba import cuda, void, int64
from mcvine.acc._numba import xoroshiro128p_uniform_float32
from math import sqrt, exp

from .interaction_types import absorption, scattering, none
from .. import test
from ..neutron import absorb, prop_dt_inplace, clone
from ..geometry.arrow_intersect import max_intersections
from ..geometry.locate import inside, outside, onborder
from ..geometry.propagation import makePropagateMethods


from numba.core import config
if not config.ENABLE_CUDASIM:
    from numba.cuda.compiler import Dispatcher, DeviceFunction

from ..config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

