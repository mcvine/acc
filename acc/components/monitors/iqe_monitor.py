# -*- python -*-
#

category = 'monitors'

import math
from numba import cuda, void, int64
import numba as nb, numpy as np
from mcni.utils.conversion import V2K, SE2V, K2V, VS2E
from ...config import get_numba_floattype, get_numpy_floattype
from ...neutron import absorb, prop_z0, e2v
NB_FLOAT = get_numba_floattype()

from .MonitorBase import MonitorBase as base
class IQE_monitor(base):

    def __init__(
            self, name,
            Ei = 60.,
            Qmin=0., Qmax=10., nQ=100,
            Emin=-45., Emax=45., nE=90,
            max_angle_in_plane = 120, min_angle_in_plane = 0,
            max_angle_out_of_plane = 30, min_angle_out_of_plane = -30,
            filename = "iqe.h5",
    ):
        """
        Initialize this IQE_monitor component.
        """
        self.name = name
        self.filename = filename
        dQ = (Qmax-Qmin)/nQ
        self.Q_centers = np.arange(Qmin+dQ/2, Qmax, dQ)
        dE = (Emax-Emin)/nE
        self.E_centers = np.arange(Emin+dE/2, Emax, dE)
        shape = 3, nQ, nE
        self.out = np.zeros(shape)
        self.out_N = self.out[0]
        self.out_p = self.out[1]
        self.out_p2 = self.out[2]
        self.propagate_params = (
            np.array([Ei, Qmin, Qmax, Emin, Emax,
                      max_angle_in_plane, min_angle_in_plane,
                      max_angle_out_of_plane, min_angle_out_of_plane,
                      ]),
            nQ, nE, self.out
        )

    def getHistogram(self, scale_factor=1.):
        import histogram as H
        axes = [('Q', self.Q_centers, '1./angstrom'), ('E', self.E_centers, 'meV')]
        return H.histogram(
            'IQE', axes,
            data=self.out_p*scale_factor,
            errors=self.out_p2*scale_factor*scale_factor)

    @cuda.jit(
        void(NB_FLOAT[:], NB_FLOAT[:], int64, int64, NB_FLOAT[:, :, :]),
        device=True)
    def propagate(neutron, params, nQ, nE, out):
        (Ei, Qmin, Qmax, Emin, Emax,
         max_angle_in_plane, min_angle_in_plane,
         max_angle_out_of_plane, min_angle_out_of_plane) = params
        vx,vy,vz = neutron[3:6]
        angle_in_plane = math.atan2( vx,vz )/math.pi*180.
        if vy!=0:
            theta = math.atan( math.sqrt(vx*vx+vz*vz)/vy )/math.pi*180.
            if vy>0: angle_out_of_plane = 90.-theta
            else: angle_out_of_plane = 90. + theta
        else:
            angle_out_of_plane = 0.

        if min_angle_in_plane < angle_in_plane \
	       and angle_in_plane < max_angle_in_plane \
	       and min_angle_out_of_plane < angle_out_of_plane \
	       and angle_out_of_plane < max_angle_out_of_plane:

            # determine Q and E
            vi = e2v(Ei)
            Q = math.sqrt( vx*vx + vy*vy + (vz-vi)*(vz-vi) )*V2K
            E = Ei - (vx*vx+vy*vy+vz*vz) * VS2E
            # find out the bin numbers and add to the histogram
            if (Q>=Qmin and Q<Qmax) and (E>=Emin and E<Emax) :
                iQ=int( math.floor( (Q-Qmin)/(Qmax-Qmin)*nQ ) )
                iE=int( math.floor( (E-Emin)/(Emax-Emin)*nE ) )
                p = neutron[-1]
                cuda.atomic.add(out, (0,iQ,iE), 1)
                cuda.atomic.add(out, (1,iQ,iE), p)
                cuda.atomic.add(out, (2,iQ,iE), p*p)
        return
