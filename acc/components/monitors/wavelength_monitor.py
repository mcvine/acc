# -*- python -*-
#

category = 'monitors'

import math
from numba import cuda
import numba as nb, numpy as np
from mcni.utils.conversion import V2K
from ...config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()

from .MonitorBase import MonitorBase as base
class Wavelength_monitor(base):

    def __init__(
            self, name,
            xmin=0., xmax=0., ymin=0., ymax=0.,
            xwidth=0., yheight=0.,
            Lmin=0., Lmax=10., nchan=200,
            filename = "IL.h5",
    ):
        self.name = name
        self.filename = filename
        if xwidth > 0:
            xmax = xwidth/2; xmin = -xmax
        if yheight > 0:
            ymax = yheight/2; ymin = -ymax
        assert xmin < xmax
        assert ymin < ymax
        dL = (Lmax-Lmin)/nchan
        self.L_centers = np.arange(Lmin+dL/2, Lmax, dL)
        self.out_N = np.zeros(nchan)
        self.out_p = np.zeros(nchan)
        self.out_p2 = np.zeros(nchan)
        self.propagate_params = (
            xmin, xmax, ymin, ymax,
            Lmin, Lmax, nchan,
            self.out_N, self.out_p, self.out_p2
        )

    def getHistogram(self, scale_factor=1.):
        import histogram as H
        axes = [('wavelength', self.L_centers, 'angstrom')]
        return H.histogram(
            'I(wavelength)', axes,
            data=self.out_p*scale_factor,
            errors=self.out_p2*scale_factor*scale_factor)


from ...neutron import absorb, prop_z0

@cuda.jit(device=True)
def propagate(
        neutron,
        xmin, xmax, ymin, ymax,
        Lmin, Lmax, nchan,
        out_N, out_p, out_p2
):
    t0 = neutron[-2]
    x,y,z, t = prop_z0(neutron)
    if t0>t:
        return
    p = neutron[-1]
    vx,vy,vz = neutron[3:6]
    #
    if x<=xmin or x>=xmax or y<=ymin or y>=ymax:
        return
    v = math.sqrt(vx*vx+vy*vy+vz*vz)
    L = 2*math.pi/(v*V2K)
    if L<=Lmin or L>=Lmax:
        return
    iL = int(math.floor( (L-Lmin)/(Lmax-Lmin)*nchan ))
    cuda.atomic.add(out_N, iL, 1)
    cuda.atomic.add(out_p, iL, p)
    cuda.atomic.add(out_p2, iL, p*p)
    return

Wavelength_monitor.register_propagate_method(propagate)
