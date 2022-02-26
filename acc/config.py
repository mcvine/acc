floattype = "float64"

def get_numpy_floattype():
    import numpy as np
    return getattr(np, floattype)

def get_numba_floattype():
    import numba
    return getattr(numba, floattype)
