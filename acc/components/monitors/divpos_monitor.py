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
class DivPos_monitor(base):

    def __init__(
            self, name,
            xmin=0., xmax=0., ymin=0., ymax=0.,
            xwidth=0., yheight=0.,
            maxdiv=2.,
            npos=20., ndiv=20.,
            filename = "divpos.h5",
    ):
        """
        Initialize this Source_simple component.

        Parameters
        ----------

        xwidth : float
            Width in meter
        yheight : float
            Height in meter
        """
        self.name = name
        self.filename = filename
        if xwidth > 0:
            xmax = xwidth/2; xmin = -xmax
        if yheight > 0:
            ymax = yheight/2; ymin = -ymax
        assert xmin < xmax
        assert ymin < ymax
        dx = (xmax-xmin)/npos
        self.x_centers = np.arange(xmin+dx/2, xmax, dx)
        ddiv = 2*maxdiv/ndiv
        self.div_centers = np.arange(-maxdiv+ddiv/2, maxdiv, ddiv)
        shape = ndiv, npos
        self.out_N = np.zeros(shape)
        self.out_p = np.zeros(shape)
        self.out_p2 = np.zeros(shape)
        self.propagate_params = (
            xmin, xmax, ymin, ymax, xwidth, yheight, maxdiv,
            npos, ndiv, self.out_N, self.out_p, self.out_p2
        )

    def getHistogram(self, scale_factor=1.):
        import histogram as H
        axes = [('x', self.x_centers, 'm'), ('div', self.div_centers, 'deg')]
        return H.histogram(
            'Idiv_x', axes,
            data=self.out_p.T*scale_factor,
            errors=self.out_p2.T*scale_factor*scale_factor)

    @cuda.jit(
        void(NB_FLOAT[:],
             NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
             int64, int64,
             NB_FLOAT[:, :], NB_FLOAT[:, :], NB_FLOAT[:, :]),
        device=True)
    def propagate(
            neutron,
            xmin, xmax, ymin, ymax, xwidth, yheight, maxdiv,
            npos, ndiv,
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
        div = math.atan(vx/vz)*RAD2DEG
        if div>=maxdiv or div<=-maxdiv:
            return
        ix = int(math.floor( (x-xmin)/(xmax-xmin)*npos ))
        idiv = int(math.floor( (div+maxdiv)/(2*maxdiv)*ndiv ))
        cuda.atomic.add(out_N, (idiv,ix), 1)
        cuda.atomic.add(out_p, (idiv,ix), p)
        cuda.atomic.add(out_p2, (idiv,ix), p*p)
        return

