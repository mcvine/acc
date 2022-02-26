ntotalthreads = int(1e6)
threads_per_block = 512
floattype = "float64"

def get_numpy_floattype():
    import numpy as np
    return getattr(np, floattype)

def get_numba_floattype():
    import numba
    return getattr(numba, floattype)
