# -*- python -*-
#

category = 'monitors'

import math
from numba import cuda, void, int64
import numba as nb, numpy as np
from mcni.utils.conversion import V2K, SE2V, K2V
from ...config import get_numba_floattype, get_numpy_floattype
from ...neutron import absorb, prop_z0, prop_dt_inplace
from ...geometry.arrow_intersect import cu_device_intersect_sphere
NB_FLOAT = get_numba_floattype()
RAD2DEG = 180./math.pi

from .MonitorBase import MonitorBase as base
class PSD_monitor_4Pi(base):

    def __init__(
            self, name,
            radius = 1.,
            nphi=90, ntheta=90,
            filename = "psd_monitor_4pi.h5",
    ):
        """
        Initialize this component.
        """
        self.name = name
        self.radius = radius
        self.filename = filename
        thetamin, thetamax = -math.pi/2, math.pi/2
        phimin, phimax = -math.pi, math.pi
        dphi = (phimax-phimin)/nphi
        self.phi_centers = np.arange(phimin+dphi/2, phimax, dphi)
        dtheta = (thetamax-thetamin)/ntheta
        self.theta_centers = np.arange(-thetamax+dtheta/2, thetamax, dtheta)
        shape = 3, nphi, ntheta
        self.out = np.zeros(shape)
        self.out_N = self.out[0]
        self.out_p = self.out[1]
        self.out_p2 = self.out[2]
        self.propagate_params = (
            np.array([phimin, phimax, thetamin, thetamax]),
            radius, nphi, ntheta, self.out
        )

    def getHistogram(self, scale_factor=1.):
        import histogram as H
        axes = [('phi', self.phi_centers, 'radian'),
                ('theta', self.theta_centers, 'radian')]
        return H.histogram(
            'I_thetaphi', axes,
            data=self.out_p*scale_factor,
            errors=self.out_p2*scale_factor*scale_factor)

    @cuda.jit(
        void(NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT, int64, int64, NB_FLOAT[:, :, :]),
        device=True)
    def propagate(neutron, limits, radius, nphi, ntheta, out):
        phimin, phimax, thetamin, thetamax = limits
        x,y,z, vx,vy,vz = neutron[:6]
        t1, t2 = cu_device_intersect_sphere(x,y,z, vx,vy,vz, radius)
        if t2<0: return
        if t1<0: t1 = t2
        prop_dt_inplace(neutron, t1)
        x,y,z = neutron[:3]
        phi = math.atan2(x,z)
        theta = math.asin(y/radius)
        iphi = int(nphi*(phi/2/math.pi+0.5))
        if iphi>=nphi: iphi = nphi-1
        elif iphi < 0: iphi = 0
        itheta = int(ntheta*(theta+math.pi/2)/math.pi+0.5)
        if itheta>=ntheta: itheta = ntheta-1
        elif itheta < 0: itheta = 0
        p = neutron[-1]
        cuda.atomic.add(out, (0, iphi, itheta), 1)
        cuda.atomic.add(out, (1, iphi, itheta), p)
        cuda.atomic.add(out, (2, iphi, itheta), p*p)
        return

