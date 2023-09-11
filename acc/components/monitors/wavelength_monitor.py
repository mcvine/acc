# -*- python -*-
#

category = 'monitors'

import math
from numba import cuda, void, int64
import numba as nb, numpy as np
from mcni.utils.conversion import V2K
from ...config import get_numba_floattype, get_numpy_floattype
from ...neutron import absorb, prop_z0
NB_FLOAT = get_numba_floattype()

from .MonitorBase import MonitorBase as base
class Wavelength_monitor(base):

    def __init__(
            self, name,
            xmin=0., xmax=0., ymin=0., ymax=0.,
            xwidth=0., yheight=0.,
            Lmin=0., Lmax=10., nchan=200,
            filename = "IL.h5",
            **kwargs
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
        self.out = np.zeros((3,nchan))
        self.out_N = self.out[0]
        self.out_p = self.out[1]
        self.out_p2 = self.out[2]
        self.propagate_params = (
            np.array([xmin, xmax, ymin, ymax, Lmin, Lmax]),
            nchan, self.out
        )

    def getHistogram(self, scale_factor=1.):
        import histogram as H
        axes = [('wavelength', self.L_centers, 'angstrom')]
        return H.histogram(
            'I(wavelength)', axes,
            data=self.out_p*scale_factor,
            errors=self.out_p2*scale_factor*scale_factor)

    @cuda.jit(
        void(NB_FLOAT[:], NB_FLOAT[:], int64, NB_FLOAT[:, :]),
        device=True)
    def propagate(neutron, limits, nchan, out):
        p = neutron[-1]
        if p < 0.0:
            return
        xmin, xmax, ymin, ymax, Lmin, Lmax = limits
        t0 = neutron[-2]
        x,y,z, t = prop_z0(neutron)
        if t0>t:
            return

        vx,vy,vz = neutron[3:6]
        #
        if x<=xmin or x>=xmax or y<=ymin or y>=ymax:
            return
        v = math.sqrt(vx*vx+vy*vy+vz*vz)
        L = 2*math.pi/(v*V2K)
        if L<=Lmin or L>=Lmax:
            return
        iL = int(math.floor( (L-Lmin)/(Lmax-Lmin)*nchan ))
        cuda.atomic.add(out, ( 0, iL ), 1)
        cuda.atomic.add(out, ( 1, iL ), p)
        cuda.atomic.add(out, ( 2, iL ), p*p)
        return
