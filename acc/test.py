import os
USE_CUDA = os.environ.get('USE_CUDA', 'true') in ['true', '1', 'yes']
import numba.cuda
USE_CUDA = USE_CUDA and numba.cuda.is_available()
