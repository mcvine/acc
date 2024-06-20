ntotalthreads = int(1e6)
threads_per_block = 512
floattype = "float64"
rng_seed = 1

from numba.core import config
ENABLE_CUDASIM = config.ENABLE_CUDASIM

def get_numpy_floattype():
    import numpy as np
    return getattr(np, floattype)

def get_numba_floattype():
    import numba
    return getattr(numba, floattype)


def get_max_registers(threads=threads_per_block):
    # returns the maximum registers a kernel can use based on the device and block size
    if ENABLE_CUDASIM:
        return 128
    from numba.cuda import get_current_device
    max_regs_per_block = getattr(
        get_current_device(), "MAX_REGISTERS_PER_BLOCK")
    return int(max_regs_per_block / threads)
