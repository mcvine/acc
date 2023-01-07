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
        self.Ei = Ei
        self.nQ, self.Qmin, self.Qmax = nQ, Qmin, Qmax
        self.nE, self.Emin, self.Emax = nE, Emin, Emax
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

    def copy(self):
        (Ei, Qmin, Qmax, Emin, Emax,
         max_angle_in_plane, min_angle_in_plane,
         max_angle_out_of_plane, min_angle_out_of_plane,
         ), nQ, nE, out = self.propagate_params
        return self.__class__(
            self.name,
            Ei = Ei,
            Qmin=Qmin, Qmax=Qmax, nQ=nQ,
            Emin=Emin, Emax=Emax, nE=nE,
            max_angle_in_plane = max_angle_in_plane,
            min_angle_in_plane = min_angle_in_plane,
            max_angle_out_of_plane = max_angle_out_of_plane,
            min_angle_out_of_plane = min_angle_out_of_plane,
            filename=self.filename)

    def getHistogram(self, scale_factor=1.):
        h = self._getHistogram(scale_factor)
        n = getNormalization(self, N=None, epsilon=1e-7)
        return h/n

    def _getHistogram(self, scale_factor=1.):
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

def getNormalization(monitor, N=None, epsilon=1e-7):
    # randomly shoot neutrons to monitor in 4pi solid angle
    print("* start computing normalizer...")
    if N is None:
        N = monitor.nQ * monitor.nE * 10000
    import mcni, random, mcni.utils.conversion as conversion, math, os
    import numpy as np

    # incident velocity
    vi = conversion.e2v(monitor.Ei)
    # monitor copy
    mcopy = monitor.copy()
    # send neutrons to monitor copy
    N1 = 0; dN = int(2e7)
    print("  - total neutrons needed :", N)
    while N1 < N:
        n = min(N-N1, dN)
        neutrons = makeNeutrons(monitor.Ei, monitor.Emin, monitor.Emax, n)
        mcopy.process(neutrons)
        N1 += n
        print("  - processed %s" % N1)
        continue
    h = mcopy._getHistogram(1.) # /N)
    # for debug
    # import histogram.hdf as hh
    # hh.dump(h, 'tmp.h5', '/', 'c')
    h.I[h.I<epsilon] = 1
    #
    print("  - done computing normalizer")
    return h

def makeNeutrons(Ei, Emin, Emax, N):
    import mcni
    neutrons = mcni.neutron_buffer(N)
    # randomly select E, the energy transfer
    E = Emin + np.random.random(N) * (Emax-Emin)
    # the final energy
    Ef = Ei - E
    # the final velocity
    from mcni.utils.conversion import e2v
    vi = e2v(Ei)
    vf = e2v(Ef)
    # choose cos(theta) between -1 and 1
    cos_t = np.random.random(N) * 2 - 1
    # theta
    theta = np.arccos(cos_t)
    # sin(theta)
    sin_t = np.sin(theta)
    # phi: 0 - 2pi
    phi = np.random.random(N) * 2 * np.pi
    # compute final velocity vector
    vx,vy,vz = vf*sin_t*np.cos(phi), vf*sin_t*np.sin(phi), vf*cos_t
    # neutron position, spin, tof are set to zero
    x = y = z = sx = sy = t = np.zeros(N, dtype="float64")
    # probability
    prob = np.ones(N, dtype="float64") * (vf/vi)
    # XXX: this assumes a specific data layout of neutron struct
    n_arr = np.array([x,y,z,vx,vy,vz, sx,sy, t, prob]).T.copy()
    neutrons.from_npyarr(n_arr)
    return neutrons
