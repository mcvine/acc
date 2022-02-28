from numba import cuda

from ..ComponentBase import ComponentBase as base
class MonitorBase(base):

    category = 'monitors'
    filename = None # set in ctor

    def save(self, scale_factor=1.):
        h = self.getHistogram(scale_factor=scale_factor)
        import histogram.hdf as hh
        hh.dump(h, self.filename)
        return

