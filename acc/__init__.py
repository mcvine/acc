# mcvine.acc subpackage

try:
    from acc._version import __version__
except ImportError:
    __version__ = 'unknown'

from numba import cuda

# code to get number of cores
# implementaiton from https://stackoverflow.com/questions/63823395/how-can-i-get-the-number-of-cuda-cores-in-my-gpu-using-python-and-numba
# the above dictionary should result in a value of "None" if a cc match
# is not found.  The dictionary needs to be extended as new devices become
# available, and currently does not account for all Jetson devices
cc_cores_per_SM_dict = {
    (2,0) : 32,
    (2,1) : 48,
    (3,0) : 192,
    (3,5) : 192,
    (3,7) : 192,
    (5,0) : 128,
    (5,2) : 128,
    (6,0) : 64,
    (6,1) : 128,
    (7,0) : 64,
    (7,5) : 64,
    (8,0) : 64,
    (8,6) : 128
    }
def get_number_of_cores_current_device():
    device = cuda.get_current_device()
    sms = getattr(device, 'MULTIPROCESSOR_COUNT')
    cc = device.compute_capability
    cores_per_sm = cc_cores_per_SM_dict[cc]
    total_cores = cores_per_sm*sms
    print("GPU compute capability: " , cc)
    print("GPU total number of SMs: " , sms)
    print("total cores: " , total_cores)
    return total_cores
