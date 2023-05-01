import os, tempfile
import numpy as np
from numba.core import config
if config.ENABLE_CUDASIM:
    def xoroshiro128p_uniform_float32(rng_states, threadindex):
        return np.random.random()
    from numba.cuda.random import xoroshiro128p_type
else:
    from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type

class Coder:

    def __init__(self, workdir=None):
        self.workdir = workdir = workdir or os.path.abspath(".mcvine.acc.coder")
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        self.modules = dict()

    def getModule(self, type, N):
        container = coder.createDir(type)
        modulepath = os.path.join(container, f'compiled_{N}.py')
        key = type, N
        if os.path.exists(modulepath) and key not in self.modules:
            self.modules[key] = modulepath
        return modulepath

    def createUniqueDir(self, prefix):
        return tempfile.mkdtemp(prefix=prefix, dir=self.workdir)

    def createDir(self, name):
        wd = os.path.join(self.workdir, name)
        if not os.path.exists(wd):
            os.makedirs(wd)
        return wd

    @classmethod
    def unrollLoop(cls, N, indent=4*' ', before_loop=None, in_loop=None, after_loop=None):
        lines=[indent+line for line in before_loop or []]
        for i in range(N):
            lines+=[indent+line.format(i=i) for line in in_loop or []]
        lines+=[indent+line for line in after_loop or []]
        return lines

coder = Coder()
