#!/usr/bin/env python
#
# Copyright (c) 2021 by UT-Battelle, LLC.

from math import sqrt, ceil

from numba import cuda, void
import numpy as np
from mcni.utils.conversion import V2K

category = 'optics'

from ._guide_utils import calc_reflectivity
from ..ComponentBase import ComponentBase
from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

max_bounces = 100000


class Guide(ComponentBase):

    def __init__(self, name, option, l, mx, my, R0=0.99, Qcx=0.021, Qcy=0.021,
                 alphax=6.07, alphay=6.07, W=0.003, **kwargs):
        """
        Initialize this Guide component.
        The guide is centered on the z-axis with the entrance at z=0.

        Parameters:
        name (str): the name of this component
        option (str): component options, specify segment file as "file={filename}"
        l (m): total length of guide
        mx: m-value of material (0 is complete absorption) for horizontal (top and bottom) mirrors
        my: m-value of material (0 is complete absorption) for vertical (left and right) mirrors
        R0: low-angle reflectivity
        Qcx: critical scattering vector for horizontal mirrors
        Qcy: critical scattering vector for vertical mirrors
        alphax: slope of reflectivity for horizontal mirrors
        alphay: slope of reflectivity for vertical mirrors
        W: width of supermirror cutoff
        """
        super().__init__(**kwargs)

        self.name = name
        self.filename = None

        if option.startswith("file="):
            self.filename = option.lstrip("file=")

        if not self.filename:
            raise RuntimeError("Expected segment file passed to guide")

        self.h1, self.h2, self.w1, self.w2 = self.load_segments()
        self.nseg = len(self.h1)
        self.l_seg = l / self.nseg

        print("l = {}, num seg = {}, l seg = {}".format(l, self.nseg, self.l_seg))

        # copy segment data to GPU
        self.h1_d = cuda.to_device(self.h1)
        self.h2_d = cuda.to_device(self.h2)
        self.w1_d = cuda.to_device(self.w1)
        self.w2_d = cuda.to_device(self.w2)

        # prepare temporary arrays on GPU
        self.ww_d = cuda.device_array(self.nseg, dtype=self.NP_FLOAT)
        self.hh_d = cuda.device_array(self.nseg, dtype=self.NP_FLOAT)
        self.whalf_d = cuda.device_array(self.nseg, dtype=self.NP_FLOAT)
        self.hhalf_d = cuda.device_array(self.nseg, dtype=self.NP_FLOAT)
        self.lwhalf_d = cuda.device_array(self.nseg, dtype=self.NP_FLOAT)
        self.lhhalf_d = cuda.device_array(self.nseg, dtype=self.NP_FLOAT)

        threadsperblock = 512
        nblocks = ceil(self.nseg / threadsperblock)
        print(nblocks, threadsperblock)
        prep_inputs = cuda.jit(self.prep_inputs_kernel)
        prep_inputs[nblocks, threadsperblock](self.w1_d, self.w2_d, self.h1_d, self.h2_d, self.l_seg, self.ww_d,
                                              self.hh_d, self.whalf_d, self.hhalf_d, self.lwhalf_d, self.lhhalf_d)
        cuda.synchronize()

        self.propagate_params = (
            self.w1_d, self.w2_d, self.h1_d, self.h2_d, float(self.l_seg), self.ww_d, self.hh_d,
            self.whalf_d, self.hhalf_d, self.lwhalf_d, self.lhhalf_d, float(R0), float(Qcx),
            float(Qcy), float(alphax), float(alphay), float(mx), float(my), float(W)
        )

        self.propagate = self.register_propagate_method(self.propagate)
        #print(self.process_kernel)

        self.print_kernel_info(self.propagate)

        import mcni
        neutrons = mcni.neutron_buffer(1)
        neutrons[0] = mcni.neutron(r=(0,0,0), v=(0,0,1000), prob=1, time=0)
        self.process(neutrons)

    def load_segments(self):
        """
        Loads parameters from file for each guide segment
        Returns: [h1, h2, w1, w2]
        """

        file = open(self.filename, "r")
        if not file:
            raise RuntimeError("Could not load segments from {}".format(self.filename))

        h1 = []
        h2 = []
        w1 = []
        w2 = []
        while line := file.readline():
            # skip over comments
            if line.startswith("c"):
                continue
            x, y, z, w = line.split(" ", 3)
            h1.append(x)
            h2.append(y)
            w1.append(z)
            w2.append(w)

        h1 = np.asarray(h1[:-1], dtype=self.NP_FLOAT)
        h2 = np.asarray(h2[:-1], dtype=self.NP_FLOAT)
        w1 = np.asarray(w1[:-1], dtype=self.NP_FLOAT)
        w2 = np.asarray(w2[:-1], dtype=self.NP_FLOAT)

        file.close()

        return h1, h2, w1, w2

    @classmethod
    def prep_inputs_kernel(cls, w1, w2, h1, h2, l_seg, ww, hh, whalf, hhalf, lwhalf,
                    lhhalf):
        """
        Helper kernel to pre-compute different values for each segment on the device
        w1, w2, h1, h2, l_seg are inputs
        ww, hh, whalf, hhalf, lwhalf, lhhalf are outputs
        """
        ind = cuda.grid(1)
        if ind > len(w1):
            return
        w1_ = w1[ind]
        w2_ = w2[ind]
        h1_ = h1[ind]
        h2_ = h2[ind]

        ww[ind] = 0.5 * (w2_ - w1_)
        hh[ind] = 0.5 * (h2_ - h1_)
        whalf[ind] = 0.5 * w1_
        hhalf[ind] = 0.5 * h1_
        lwhalf[ind] = l_seg * (0.5 * w1_)
        lhhalf[ind] = l_seg * (0.5 * h1_)

    @cuda.jit(void(
        NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:],
        NB_FLOAT,
        NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:],
        NB_FLOAT[:],
        NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
        NB_FLOAT
    ), device=True)
    def propagate(in_neutron, w1, w2, h1, h2, l_seg, ww, hh,
                  whalf, hhalf, lwhalf, lhhalf,
                  R0, Qcx, Qcy, alphax, alphay,
                  mx, my, W):
        x, y, z, vx, vy, vz = in_neutron[:6]
        t = in_neutron[-2]
        prob = in_neutron[-1]
        # propagate to z=0
        dt = -z / vz
        x += vx * dt;
        y += vy * dt;
        z = 0.0;
        t += dt

        # TODO: change this to a parameter possibly, impact of len() on perf?
        nsegments = len(w1)

        for seg in range(nsegments):
            zr = seg * l_seg
            # propagate to segment entrance
            dt = (zr - z) / vz
            x += vx * dt
            y += vy * dt
            z += vz * dt
            if (x <= -0.5 * w1[seg] or x >= 0.5 * w1[seg] or
                    y <= -hhalf[seg] or y >= hhalf[seg]):
                in_neutron[-1] = 0
                return

            # shift origin to center of channel hit
            x += 0.5 * w1[seg]
            edge = cuda.libdevice.floor(x / w1[seg]) * w1[seg]
            if (x - edge > w1[seg]):
                in_neutron[-1] = 0
                return
            x -= (edge + 0.5 * w1[seg])
            hadj = edge + (0.5 * w1[seg]) - 0.5 * w1[seg]

            # bouncing loop
            for nb in range(max_bounces):
                av = l_seg * vx
                bv = ww[seg] * vz
                ah = l_seg * vy
                bh = hh[seg] * vz
                vdotn_v1 = bv + av  # Left vertical
                vdotn_v2 = bv - av  # Right vertical
                vdotn_h1 = bh + ah  # Lower horizontal
                vdotn_h2 = bh - ah  # Upper horizontal
                # Compute the dot products of (O - r) and n as c1+c2 and c1-c2
                cv1 = -whalf[seg] * l_seg - (z - zr) * ww[seg]
                cv2 = x * l_seg
                ch1 = -hhalf[seg] * l_seg - (z - zr) * hh[seg]
                ch2 = y * l_seg
                # Compute intersection times.
                t1 = (zr + l_seg - z) / vz  # for guide exit
                i = 0
                if vdotn_v1 < 0:
                    t2 = (cv1 - cv2) / vdotn_v1
                    if t2 < t1:
                        t1 = t2
                        i = 1
                if vdotn_v2 < 0:
                    t2 = (cv1 + cv2) / vdotn_v2
                    if t2 < t1:
                        t1 = t2
                        i = 2
                if vdotn_h1 < 0:
                    t2 = (ch1 - ch2) / vdotn_h1
                    if t2 < t1:
                        t1 = t2
                        i = 3
                if vdotn_h2 < 0:
                    t2 = (ch1 + ch2) / vdotn_h2
                    if t2 < t1:
                        t1 = t2
                        i = 4
                if i == 0:
                    break  # Neutron left guide.

                # propagate time t1 to move to reflection point
                x += vx * t1
                y += vy * t1
                z += vz * t1
                t += t1

                # reflection
                if i == 1:  # Left vertical mirror
                    nlen2 = l_seg * l_seg + ww[seg] * ww[seg]
                    q = V2K * (-2.0) * vdotn_v1 / sqrt(nlen2)
                    d = 2.0 * vdotn_v1 / nlen2
                    vx = vx - d * l_seg
                    vz = vz - d * ww[seg]
                elif i == 2:  # Right vertical mirror
                    nlen2 = l_seg * l_seg + ww[seg] * ww[seg]
                    q = V2K * (-2.0) * vdotn_v2 / sqrt(nlen2)
                    d = 2.0 * vdotn_v2 / nlen2
                    vx = vx + d * l_seg
                    vz = vz - d * ww[seg]
                elif i == 3:  # Lower horizontal mirror
                    nlen2 = l_seg * l_seg + hh[seg] * hh[seg]
                    q = V2K * (-2.0) * vdotn_h1 / sqrt(nlen2)
                    d = 2.0 * vdotn_h1 / nlen2
                    vy = vy - d * l_seg
                    vz = vz - d * hh[seg]
                elif i == 4:  # Upper horizontal mirror
                    nlen2 = l_seg * l_seg + hh[seg] * hh[seg]
                    q = V2K * (-2.0) * vdotn_h2 / sqrt(nlen2)
                    d = 2.0 * vdotn_h2 / nlen2
                    vy = vy + d * l_seg
                    vz = vz - d * hh[seg]

                if (i <= 2 and mx == 0) or (i > 2 and my == 0):
                    x += hadj
                    break

                if i <= 2:
                    m = mx
                    if m <= 0.0:
                        m = abs(mx * w1[0] / w1[seg])
                    R = calc_reflectivity(q, R0, Qcx, alphax, m, W)
                    if R > 1e-10:
                        prob *= R
                    else:
                        in_neutron[-1] = 0
                        return
                else:
                    m = my
                    if m <= 0.0:
                        m = abs(my * h1[0] / h1[seg])
                    R = calc_reflectivity(q, R0, Qcy, alphay, m, W)
                    if R > 1e-10:
                        prob *= R
                    else:
                        in_neutron[-1] = 0
                        return
            x += hadj

        in_neutron[:6] = x, y, z, vx, vy, vz
        in_neutron[-2] = t
        in_neutron[-1] = prob

        return
