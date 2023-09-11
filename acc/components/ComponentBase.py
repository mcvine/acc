"""
Requirement for a component
* component class
  - inherit from ComponentBase
  - ctor must create self.propagate_params
* `propagate` method
  - first argument: `neutron`
  - other args: match comp.propagate_params
"""
import numpy as np
import numba
from numba import cuda
import math

from mcni.AbstractComponent import AbstractComponent
from numba.core.types import Array, Float

from .. import config
from ..config import get_max_registers


class Curator(type):

    components = []

    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if not name.endswith('Base'):
            if not hasattr(x, 'propagate'):
                raise TypeError(f"{name} should define `propagate` method")
            x.propagate = x.register_propagate_method(x.propagate)
            Curator.components.append(x)
        return x

def change_floattype(newtype):
    # do we really need each component type to have individual floattype
    # or one floattype for the base class is enough?
    ComponentBase._floattype = newtype
    for c in Curator.components:
        c.change_floattype(newtype)

class ComponentBase(AbstractComponent, metaclass=Curator):

    _floattype = "float64"

    propagate_params = ()

    # flag whether this is a multiple scattering component
    is_multiplescattering = False
    NUM_MULTIPLE_SCATTER = 10

    @property
    def NP_FLOAT(self):
        return self.get_numpy_floattype()

    @property
    def NB_FLOAT(self):
        return self.get_numba_floattype()

    @property
    def floattype(self):
        return self.__class__._floattype

    @classmethod
    def change_floattype(cls, newtype):
        cls._floattype = newtype
        cls.propagate = cls.register_propagate_method(cls.propagate)

    @classmethod
    def get_floattype(cls):
        # TODO: fix this to use a class property metaclass?
        return cls._floattype

    def get_numpy_floattype(self):
        return getattr(np, self.__class__._floattype)

    def get_numba_floattype(self):
        return getattr(numba, self.__class__._floattype)

    def process(self, neutrons):
        from time import time
        from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
        t1 = time()
        neutron_array = neutrons_as_npyarr(neutrons)
        neutron_array.shape = -1, ndblsperneutron
        neutron_array_dtype_api = neutron_array.dtype
        neutron_array_dtype_int = self.get_numpy_floattype()
        needs_cast = neutron_array_dtype_api != neutron_array_dtype_int
        if needs_cast:
            neutron_array = neutron_array.astype(neutron_array_dtype_int)
        t2 = time()
        from ..config import ntotalthreads, threads_per_block
        neutron_array = self.call_process(
            self.__class__.process_kernel,
            neutron_array,
            ntotthreads=ntotalthreads, threads_per_block = threads_per_block,
        )
        t3 = time()
        if needs_cast:
            neutron_array = neutron_array.astype(neutron_array_dtype_api)
        good = neutron_array[:, -1] > 0
        neutrons.resize(int(good.sum()), neutrons[0])
        neutrons.from_npyarr(neutron_array[good])
        t4 = time()
        print(self.name, ":prepare input array: ", t2-t1)
        print(self.name, ":call_process: ", t3-t2)
        print(self.name, ":prepare output neutrons: ", t4-t3)
        return neutrons

    def call_process(
            self, process_kernel, in_neutrons,
            ntotthreads=None, threads_per_block=None,
    ):
        # Use provided values or fall back to defaults in config
        threads_per_block = threads_per_block or config.threads_per_block
        ntotthreads = ntotthreads or config.ntotalthreads

        N = len(in_neutrons)
        ntotthreads = min(N, ntotthreads)
        nblocks = math.ceil(ntotthreads / threads_per_block)
        actual_nthreads = threads_per_block * nblocks
        n_neutrons_per_thread = math.ceil(N / actual_nthreads)
        print("%s blocks, %s threads, %s neutrons per thread" % (
            nblocks, threads_per_block, n_neutrons_per_thread))
        self.check_kernel_launch(process_kernel, threads_per_block,
                                 in_neutrons, n_neutrons_per_thread, self.propagate_params)
        process_kernel[nblocks, threads_per_block](
            in_neutrons, n_neutrons_per_thread, self.propagate_params)
        cuda.synchronize()
        return in_neutrons

    @classmethod
    def register_propagate_method(cls, propagate):
        new_propagate = cls._adjust_propagate_type(propagate)

        cls.process_kernel = make_process_kernel(new_propagate)
        # cls.print_kernel_info(cls.process_kernel)
        return new_propagate

    @classmethod
    def _adjust_propagate_type(cls, propagate):
        # disable float switching if in cudasim mode
        if config.ENABLE_CUDASIM:
            return propagate

        # no longer need float switching in newer numba
        if int(numba.__version__.split(".")[1]) >= 56:
            cls.print_kernel_info(propagate)
            return propagate

        from numba.cuda.compiler import DeviceFunction

        if not isinstance(propagate, DeviceFunction):
            raise RuntimeError(
                "invalid propagate function ({}, {}) registered, ".format(
                    propagate, type(propagate))
                + "does propagate have a signature defined?")

        args = propagate.args

        # reconstruct the numba args with the correct floattype
        newargs = []
        for arg in args:
            if isinstance(arg, Array) and isinstance(arg.dtype, Float):
                newargs.append(arg.copy(dtype=getattr(numba, cls._floattype)))
            elif isinstance(arg, Float):
                newargs.append(Float(name=cls._floattype))
            else:
                # copy other args through
                newargs.append(arg)
        newargs = tuple(newargs)

        # DeviceFunction in Numba < 0.54.1 does not have a lineinfo property
        if int(numba.__version__.split(".")[1]) < 54:
            new_propagate = DeviceFunction(pyfunc=propagate.py_func,
                                           return_type=propagate.return_type,
                                           args=newargs,
                                           inline=propagate.inline,
                                           debug=propagate.debug)
        else:
            new_propagate = DeviceFunction(pyfunc=propagate.py_func,
                                           return_type=propagate.return_type,
                                           args=newargs,
                                           inline=propagate.inline,
                                           debug=propagate.debug,
                                           lineinfo=propagate.lineinfo)
        #cls.print_kernel_info(new_propagate)
        return new_propagate

    @classmethod
    def print_kernel_info(cls, kernel):
        try:
            print("{} kernel ({}):".format(cls.__name__, type(kernel)))
            dispatch_type = None
            if int(numba.__version__.split(".")[1]) >= 56:
                from numba.cuda.dispatcher import CUDADispatcher
                dispatch_type = CUDADispatcher
            else:
                from numba.cuda.compiler import Dispatcher, DeviceFunction
                dispatch_type = Dispatcher

            if isinstance(kernel, dispatch_type):
                print("      specialized? {}".format(kernel.specialized))
                print("      using {} registers".format(kernel.get_regs_per_thread()))
                print("      inspect types: '{}'".format(kernel.inspect_types()))
                print("      nopython sigs: '{}'".format(kernel.nopython_signatures))
                print("      _func: '{}'".format(kernel.py_func))
                print("      sigs = '{}'".format(kernel.sigs))
            elif isinstance(kernel, DeviceFunction):
                print("      _func: '{}'".format(kernel.py_func))
                print("      repr = '{}'".format(kernel.__repr__))
                print("      return type = '{}'".format(kernel.return_type))
                print("      args = '{}'".format(kernel.args))
                print("      cres = '{}'".format(kernel.cres))
        except Exception as e:
            print(e)
            return

    def check_kernel_launch(self, kernel, threads_per_block, *args):
        """
        Helper to check the register usage of a kernel.
        This will raise a RuntimeError if using too many to avoid a CUDA launch error.

        Parameters:
        kernel: Dispatcher object from @cuda.jit()
        threads_per_block: Threads per block to launch this kernel with
        *args: parameters used to launch the kernel
        """
        if config.ENABLE_CUDASIM:
            return

        max_registers = get_max_registers(threads_per_block)

        specialized = kernel.specialize(*args)
        used_regs = specialized.get_regs_per_thread()

        if used_regs > max_registers:
            raise RuntimeError("{}: kernel {} is using too many registers for the block size. using {} but max for block size {} is {}".format(
                self.__class__, kernel.py_func, used_regs, threads_per_block, max_registers))


def make_process_kernel(propagate):
    @cuda.jit()
    def process_kernel(neutrons, n_neutrons_per_thread, args):
        N = len(neutrons)
        thread_index = cuda.grid(1)
        start_index = thread_index*n_neutrons_per_thread
        end_index = min(start_index+n_neutrons_per_thread, N)
        for i in range(start_index, end_index):
            propagate(neutrons[i], *args)
        return
    return process_kernel
