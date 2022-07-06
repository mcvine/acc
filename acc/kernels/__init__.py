from numba import cuda
from mcni import units

class ScatterFuncFactory:

    def render(self, kernel):
        return kernel.identify(self)

    def onIsotropicKernel(self, kernel):
        from .isotropic import scatter
        return scatter

scatter_func_factory = ScatterFuncFactory()
