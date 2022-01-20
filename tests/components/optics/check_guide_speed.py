#!/usr/bin/env python

import os, numpy as np, time
thisdir = os.path.dirname(__file__)

def test():
    from mcni.neutron_storage import load
    neutrons = load('data/before_guide.mcv')
    from mcvine.acc.components.optics import guide_baseline
    g = guide_baseline.Guide(
        name = 'guide',
        w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=10,
        R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
    )
    t1 = time.time()
    g.process(neutrons)
    print("GPU processing time:", time.time()-t1)

    neutrons = load('data/before_guide.mcv')
    import mcvine.components as mc
    g = mc.optics.Guide(
        name = 'guide',
        w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=10,
        R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
    )
    t1 = time.time()
    g.process(neutrons)
    print("CPU processing time:", time.time()-t1)
    return

if __name__ == '__main__': test()
