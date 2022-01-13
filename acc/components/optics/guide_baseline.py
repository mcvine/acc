#!/usr/bin/env python
#
# Copyright (c) 2021 by UT-Battelle, LLC.

from math import inf, isnan, tanh, sqrt, ceil
from numba import cuda, guvectorize
import numpy as np

from mcni.AbstractComponent import AbstractComponent
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron

category = 'optics'

# mcni.utils.conversion.v2k
from mcni.utils.conversion import V2K
# In mcstas header
# V2K = 1.58825361e-3
@cuda.jit(device=True, inline=True)
def v2k(v):
    """v in m/s, k in inverse AA """
    return v * V2K

@cuda.jit(device=True, inline=True)
def calc_reflectivity(Q, R0, Qc, alpha, m, W):
    """
    Calculate the mirror reflectivity for a neutron.

    Returns:
    float: the reflectivity for the neutron's given momentum change
    """
    R = R0
    if Q > Qc:
        tmp = (Q - m * Qc) / W
        if tmp < 10:
            R *= (1 - tanh(tmp)) * (1 - alpha * (Q - Qc)) / 2
        else:
            R = 0
    return R

max_bounces = 100000
@cuda.jit(device=True)
def propagate(
        ww, hh, hw1, hh1, l,
        R0, Qc, alpha, m, W,
        in_neutron, out_neutron,
):
    for i in range(10):
        out_neutron[i] = in_neutron[i]
    x,y,z,vx,vy,vz = in_neutron[:6]
    t = in_neutron[-2]
    prob = in_neutron[-1]
    # propagate to z=0
    dt = -z/vz
    x+=vx*dt; y+=vy*dt; z=0; t+=dt
    # check opening
    if (x<=-hw1 or x>=hw1 or y<=-hh1 or y>=hh1):
        out_neutron[-1] = 0
        return
    # bouncing loop
    for nb in range(max_bounces):
        av = l*vx; bv = ww*vz
        ah = l*vy; bh = hh*vz;
        vdotn_v1 = bv + av;         # Left vertical
        vdotn_v2 = bv - av;         # Right vertical
        vdotn_h1 = bh + ah;         # Lower horizontal
        vdotn_h2 = bh - ah;         # Upper horizontal
        # Compute the dot products of (O - r) and n as c1+c2 and c1-c2 
        cv1 = -hw1*l - z*ww; cv2 = x*l;
        ch1 = -hh1*l - z*hh; ch2 = y*l;
        # Compute intersection times.
        t1 = (l - z)/vz; # for guide exit
        i = 0;
        if vdotn_v1 < 0:
            t2 = (cv1 - cv2)/vdotn_v1
            if t2 < t1:
                t1 = t2;
                i = 1
        if vdotn_v2 < 0:
            t2 = (cv1 + cv2)/vdotn_v2
            if t2<t1:
                t1 = t2;
                i = 2
        if vdotn_h1 < 0:
            t2 = (ch1 - ch2)/vdotn_h1
            if t2<t1:
                t1 = t2;
                i = 3;
        if vdotn_h2 < 0 :
            t2 = (ch1 + ch2)/vdotn_h2
            if t2 < t1:
                t1 = t2;
                i = 4;
        if i == 0:
            break;                    # Neutron left guide.

        # propagate time t1 to move to reflection point
        x+=vx*t1; y+=vy*t1; z+=vz*t1; t+=t1

        # reflection
        if i==1:                     # Left vertical mirror
            nlen2 = l*l + ww*ww;
            q = V2K*(-2)*vdotn_v1/sqrt(nlen2);
            d = 2*vdotn_v1/nlen2;
            vx = vx - d*l;
            vz = vz - d*ww;
        elif i==2:                   # Right vertical mirror
            nlen2 = l*l + ww*ww;
            q = V2K*(-2)*vdotn_v2/sqrt(nlen2);
            d = 2*vdotn_v2/nlen2;
            vx = vx + d*l;
            vz = vz - d*ww;
        elif i== 3:                   # Lower horizontal mirror
            nlen2 = l*l + hh*hh;
            q = V2K*(-2)*vdotn_h1/sqrt(nlen2);
            d = 2*vdotn_h1/nlen2;
            vy = vy - d*l;
            vz = vz - d*hh;
        elif i== 4:                   # Upper horizontal mirror
            nlen2 = l*l + hh*hh;
            q = V2K*(-2)*vdotn_h2/sqrt(nlen2);
            d = 2*vdotn_h2/nlen2;
            vy = vy + d*l;
            vz = vz - d*hh;
        R = calc_reflectivity(q, R0, Qc, alpha, m, W)
        prob*=R
        if prob<=0: break
        continue
    out_neutron[:6] = x,y,z,vx,vy,vz
    out_neutron[-2] = t
    out_neutron[-1] = prob
    return

@cuda.jit
def process_kernel(
        ww, hh, hw1, hh1, l,
        R0, Qc, alpha, m, W,
        in_neutrons, out_neutrons,
):
    x = cuda.grid(1)
    if x < len(in_neutrons):
        propagate(
            ww, hh, hw1, hh1, l,
            R0, Qc, alpha, m, W,
            in_neutrons[x], out_neutrons[x],
        )
    return

def call_process(
        ww, hh, hw1, hh1, l,
        R0, Qc, alpha, m, W,
        in_neutrons, out_neutrons,
):
    N = len(in_neutrons)
    threadsperblock = 512
    nblocks = ceil(N/threadsperblock)
    print(nblocks, threadsperblock)
    process_kernel[nblocks, threadsperblock](
        ww, hh, hw1, hh1, l,
        R0, Qc, alpha, m, W,
        in_neutrons, out_neutrons,
    )
    cuda.synchronize()

@guvectorize(
    ["float32, float32, float32, float32, float32, "
     "float32, float32, float32, float32, float32, "
     "float32[:, :], float32[:, :]"],
    "(),(),(),(),(), (),(),(),(),(),  (m,n)->(m,n)",
    target="cuda")
def guv_process(
        ww, hh, hw1, hh1, l,
        R0, Qc, alpha, m, W,
        neutrons_in, neutrons_out,
):
    """
    Vectorized kernel for propagation of neutrons through guide via GPU.
    Neutrons with a weight of 0 set in neutrons_out are absorbed and any
    other written characteristics are undefined.

    Parameters:
    R0: low-angle reflectivity
    Qc: critical scattering vector
    alpha: slope of reflectivity
    m: m-value of material (0 is complete absorption)
    W: width of supermirror cutoff
    neutrons_in (array): neutrons to propagate through the guide
    neutrons_out (array): write target,
        returns the neutrons as they emerge from the exit of the guide,
        indexed identically as from neutrons_in
    """
    for index in range(neutrons_in.shape[0]):
        propagate(
            ww, hh, hw1, hh1, l,
            R0, Qc, alpha, m, W,
            neutrons_in[index], neutrons_out[index],
        )


class Guide(AbstractComponent):

    def __init__(
            self, name,
            w1, h1, w2, h2, l,
            R0=0.99, Qc=0.0219, alpha=6.07, m=2, W=0.003):
        """
        Initialize this Guide component.
        The guide is centered on the z-axis with the entrance at z=0.

        Parameters:
        name (str): the name of this component
        w1 (m): width at the guide entry
        h1 (m): height at the guide entry
        w2 (m): width at the guide exit
        h2 (m): height at the guide exit
        l (m): length of guide
        R0: low-angle reflectivity
        Qc: critical scattering vector
        alpha: slope of reflectivity
        m: m-value of material (0 is complete absorption)
        W: width of supermirror cutoff
        """
        self.name = name
        ww = .5*(w2-w1); hh = .5*(h2 - h1)
        hw1 = 0.5*w1; hh1 = 0.5*h1
        self._params = (
            float(ww), float(hh), float(hw1), float(hh1), float(l),
            float(R0), float(Qc), float(alpha), float(m), float(W),
        )

    def process_using_guv(self, neutrons):
        """
        Propagate a buffer of particles through this guide.
        Adjusts the buffer to include only the particles that exit,
        at the moment of exit.

        Parameters:
        neutrons: a buffer containing the particles
        """
        neutron_array = neutrons_as_npyarr(neutrons).astype("float32")
        neutron_array.shape = -1, ndblsperneutron
        neutrons_out = np.empty_like(neutron_array)
        guv_process(*self._params, neutron_array, neutrons_out)
        neutrons_out = neutrons_out.astype("float64")
        neutrons.from_npyarr(neutrons_out)
        mask = neutrons_out[:, -1]>0
        neutrons.resize(np.count_nonzero(mask), neutrons[0])
        neutrons.from_npyarr(neutrons_out[mask])
        return neutrons

    def process(self, neutrons):
        """
        Propagate a buffer of particles through this guide.
        Adjusts the buffer to include only the particles that exit,
        at the moment of exit.

        Parameters:
        neutrons: a buffer containing the particles
        """
        neutron_array = neutrons_as_npyarr(neutrons)
        neutron_array.shape = -1, ndblsperneutron
        neutrons_out = np.empty_like(neutron_array)
        call_process(*self._params, neutron_array, neutrons_out)
        neutrons.from_npyarr(neutrons_out)
        mask = neutrons_out[:, -1]>0
        neutrons.resize(np.count_nonzero(mask), neutrons[0])
        neutrons.from_npyarr(neutrons_out[mask])
        return neutrons
