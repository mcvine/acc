import numba
from numba import cuda
from mcni import units

class ScatterFuncFactory:

    def render(self, scatterer):
        """returns cuda device function tuple (scatter, interact_path1)
        """
        return scatterer.identify(self)

    def onCompositeScatterer(self, composite):
        elements = composite.elements()
        from .composite_scatterer import factory
        return factory(composite)

    def onHomogeneousScatterer(self, hs):
        from .homogeneous_scatterer import factory
        shape = hs.shape()
        kernel = hs.kernel()
        mcweights = [float(w) for w in hs.mcweights]
        packing_factor = float(hs.packing_factor)
        methods = factory(shape, kernel, mcweights, packing_factor)
        return methods


scatter_func_factory = ScatterFuncFactory()
