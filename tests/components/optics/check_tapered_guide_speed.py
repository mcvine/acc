#!/usr/bin/env python
# This script check the speed of GPU tapered guide component against the CPU one.
# It reads neutrons from an input data file and then let both components process them.
# Run save_neutrons_before_tapered_guide.py before running this script to generate
# the input neutron data file

import os, numpy as np, time
thisdir = os.path.dirname(__file__)

guide11_dat = os.path.join(thisdir, "data", "VERDI_V01_guide1.1")
guide11_len = 10.99
guide11_mx = 6.
guide11_my = 6.

def test():
    from mcni.neutron_storage import load
    neutron_fn = 'data/before_tapered_guide-n1e7.mcv'
    neutrons = load(neutron_fn)
    from mcvine.acc.components.optics import guide_tapering
    g = guide_tapering.Guide(
        name = 'guide',
        option="file={}".format(guide11_dat),
        l=guide11_len, mx=guide11_mx, my=guide11_my,
    )
    t1 = time.time()
    g.process(neutrons)
    print("GPU processing time:", time.time()-t1)

    neutrons = load(neutron_fn)
    import mcvine.components as mc
    g = mc.optics.Guide_tapering(
        name = 'guide',
        option="file={}".format(guide11_dat),
        l=guide11_len, mx=guide11_mx, my=guide11_my,
    )
    t1 = time.time()
    g.process(neutrons)
    print("CPU processing time:", time.time()-t1)
    return

if __name__ == '__main__': test()
