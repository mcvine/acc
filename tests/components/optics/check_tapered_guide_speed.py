#!/usr/bin/env python
# This script check the speed of GPU tapered guide component against the CPU one.
# It reads neutrons from an input data file and then let both components process them.
# Run save_neutrons_before_tapered_guide.py before running this script to generate
# the input neutron data file

import os, numpy as np, time
import sys

thisdir = os.path.dirname(__file__)

guide11_dat = os.path.join(thisdir, "data", "VERDI_V01_guide1.1")
guide11_len = 10.99
guide11_mx = 6.
guide11_my = 6.

DEFAULT_NEUTRON_FILE = 'data/before_guide.mcv'


def test(neutron_fn=DEFAULT_NEUTRON_FILE, niter=1):
    # load file and store copy to avoid reloading for multiple iterations
    from mcni.neutron_storage import load, neutrons_from_npyarr, \
        neutrons_as_npyarr, ndblsperneutron
    neutrons_orig = load(neutron_fn)
    neutrons_orig = neutrons_as_npyarr(neutrons_orig)
    neutrons_orig.shape = -1, ndblsperneutron

    from mcvine.acc.components.optics import guide_tapering
    g = guide_tapering.Guide(
        name='guide',
        option="file={}".format(guide11_dat),
        l=guide11_len, mx=guide11_mx, my=guide11_my,
    )
    times = []
    for i in range(niter):
        neutrons = neutrons_from_npyarr(neutrons_orig)
        t1 = time.time_ns()
        g.process(neutrons)
        delta = time.time_ns() - t1
        times.append(delta)
        print("GPU processing time: {} ms ({} s)".format(1e-6 * delta,
                                                         1e-9 * delta))

    if niter > 1:
        # report average time over niter
        times = np.asarray(times)
        avg = times.sum() / len(times)
        print("--------")
        print("Average GPU time ({} iters): {} ms ({} s)".format(niter,
                                                                 1e-6 * avg,
                                                                 1e-9 * avg))
        print("--------")

    neutrons = neutrons_from_npyarr(neutrons_orig)
    import mcvine.components as mc
    g = mc.optics.Guide_tapering(
        name='guide',
        option="file={}".format(guide11_dat),
        l=guide11_len, mx=guide11_mx, my=guide11_my,
    )
    t1 = time.time()
    g.process(neutrons)
    print("CPU processing time:", time.time() - t1)
    return


if __name__ == '__main__':
    filename = DEFAULT_NEUTRON_FILE
    iters = 1

    # usage: check_tapered_guide [-n [iter count]] [FILENAME]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-n":
            iters = int(sys.argv[i + 1])
            i += 1
        else:
            filename = arg

        i += 1

    test(filename, iters)
