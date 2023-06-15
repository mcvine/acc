#!/usr/bin/env python

import os, shutil
import mcvine, mcvine.components as mc
thisdir = os.path.dirname(__file__)

def instrument(z_sample=1., cpu=False):
    assert z_sample > 0.1
    instrument = mcvine.instrument()
    from mcvine.acc.components.sources.source_simple import Source_simple
    source = Source_simple(
        'src',
        radius = 0., width = 0.01, height = 0.01, dist = z_sample-0.1,
        xw = 0.008, yh = 0.008,
        Lambda0 = 10., dLambda = 9.5, E0=0., dE=0.0,
        flux=1.0, gauss=False, N=1
    )
    instrument.append(source, position=(0,0,0.))

    # from HSS_isotropic_sphere import HSS
    # sample = HSS("sample")
    from Composite_IsotropicSphere import Composite
    sample = Composite("sample")
    instrument.append(sample, position=(0,0,z_sample))

    from mcvine.acc.components.monitors.psd_monitor_4pi import PSD_monitor_4Pi
    monitor = PSD_monitor_4Pi('psd4pi', radius = 1.0, nphi=100, ntheta=120, filename="psd4pi.h5")
    instrument.append(monitor, position=(0,0,z_sample))

    return instrument

def run(outdir):
    import mcvine.acc.run_script as mars
    mars.run(__file__, outdir, ncount=100)

def main():
    outdir="out"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    run(outdir)

if __name__ == '__main__': main()
