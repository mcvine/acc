import numpy as np
from numba.core import config
if config.ENABLE_CUDASIM:
    def xoroshiro128p_uniform_float32(rng_states, threadindex):
        return np.random.random()
    from numba.cuda.random import xoroshiro128p_type
else:
    from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
