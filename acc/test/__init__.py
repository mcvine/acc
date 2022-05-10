import os
USE_CUDASIM = os.environ.get('NUMBA_ENABLE_CUDASIM', 'false') in ['true', '1', 'yes']
if not USE_CUDASIM:
    USE_CUDA = os.environ.get('USE_CUDA', 'true') in ['true', '1', 'yes']
    import numba.cuda
    USE_CUDA = USE_CUDA and numba.cuda.is_available()
else:
    USE_CUDA = False
