# -*- python -*-
#

category = 'monitors'

import math
from numba import cuda, void, int64
import numba as nb, numpy as np
from mcni.utils.conversion import V2K, SE2V, K2V
from ...config import get_numba_floattype, get_numpy_floattype
from ...neutron import absorb, prop_z0
NB_FLOAT = get_numba_floattype()
RAD2DEG = 180./math.pi

from .MonitorBase import MonitorBase as base
class PSD_monitor(base):

    def __init__(
            self, name,
            xmin=0., xmax=0., ymin=0., ymax=0.,
            xwidth=0., yheight=0.,
            nx=100, ny=100,
            filename = "psd.h5",
    ):
        """
        Initialize this PSD_monitor component.

        Parameters
        ----------

        xwidth : float
            Width in meter
        yheight : float
            height in meter
        """
        self.name = name
        self.filename = filename
        if xwidth > 0:
            xmax = xwidth/2; xmin = -xmax
        if yheight > 0:
            ymax = yheight/2; ymin = -ymax
        assert xmin < xmax
        assert ymin < ymax
        dx = (xmax-xmin)/nx
        dy = (ymax-ymin)/ny
        self.x_centers = np.arange(xmin+dx/2, xmax, dx)
        self.y_centers = np.arange(ymin+dy/2, ymax, dy)
        shape = 3, nx, ny
        self.out = np.zeros(shape)
        self.out_N = self.out[0]
        self.out_p = self.out[1]
        self.out_p2 = self.out[2]
        self.propagate_params = (
            np.array([xmin, xmax, ymin, ymax]),
            nx, ny, self.out
        )

    def getHistogram(self, scale_factor=1.):
        import histogram as H
        axes = [('x', self.x_centers, 'm'), ('y', self.y_centers, 'm')]
        return H.histogram(
            'Iyx', axes,
            data=self.out_p*scale_factor,
            errors=self.out_p2*scale_factor*scale_factor)

    @cuda.jit(
        void(NB_FLOAT[:], NB_FLOAT[:], int64, int64, NB_FLOAT[:, :, :]),
        device=True)
    def propagate(neutron, limits, nx, ny, out):
        xmin, xmax, ymin, ymax = limits
        t0 = neutron[-2]
        x,y,z, t = prop_z0(neutron)
        if t0>t:
            return
        p = neutron[-1]
        if x<=xmin or x>=xmax or y<=ymin or y>=ymax:
            return
        ix = int(math.floor( (x-xmin)/(xmax-xmin)*nx ))
        iy = int(math.floor( (y-ymin)/(ymax-ymin)*ny ))
        cuda.atomic.add(out, (0,ix,iy), 1)
        cuda.atomic.add(out, (1,ix,iy), p)
        cuda.atomic.add(out, (2,ix,iy), p*p)
        return

