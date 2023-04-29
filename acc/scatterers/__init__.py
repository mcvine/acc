import numba
from numba import cuda
from mcni import units

class ScatterFuncFactory:

    def render(self, scatterer):
        """returns cuda device function tuple (scatter, interact_path1)
        """
        scatter, interact_path1 =  scatterer.identify(self)
        return scatter, interact_path1

    def onCompositeKernel(self, composite):
        elements = composite.elements()
        weights = []
        for element in elements:
            weights.append(getattr(element, 'weight', 1.))
            continue
        s = sum(weights)
        for w, element in zip(weights, elements):
            element.weight = w/s
        from .composite import makeKernelMethods
        return makeKernelMethods(composite)

    def onHomogeneousScatterer(self, hs):
        return


scatter_func_factory = ScatterFuncFactory()
