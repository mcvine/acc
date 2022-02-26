# -*- python -*-
#

import math
from numba import cuda
import numba as nb, numpy as np
import time

from mcni.AbstractComponent import AbstractComponent
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcni.utils.conversion import V2K, SE2V, K2V

category = 'monitors'

from ...config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()
RAD2DEG = 180./math.pi

class DivPos_monitor(AbstractComponent):

    def __init__(
            self, name,
            xmin=0., xmax=0., ymin=0., ymax=0.,
            xwidth=0., yheight=0.,
            maxdiv=2.,
            npos=20., ndiv=20.
    ):
        """
        Initialize this Source_simple component.

        Parameters
        ----------

        xwidth : float
            Width in meter
        yheight : float
            Width in meter
        """
        self.name = name
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


@cuda.jit(device=True)
def propagate(
        neutron,
        xmin, xmax, ymin, ymax, xwidth, yheight, maxdiv,
        npos, ndiv,
        out_N, out_p, out_p2
):
    x,y,z,vx,vy,vz = neutron[:6]
    p = neutron[-1]
    t = neutron[-2]
    # propagate to z=0
    dt = (0-z)/vz; x+=vx*dt; y+=vy*dt; z=0; t+=dt
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
