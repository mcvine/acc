#!/usr/bin/env python
import math
import mcvine
from mcvine.acc.components.sources.source_simple import Source_simple
from mcvine.acc.components.optics.guide import Guide
from mcvine.acc.components.optics.multi_disk_chopper import MultiDiskChopper
from mcvine.acc.components.optics.disk_chopper import DiskChopper
from mcvine.acc.components.monitors.wavelength_monitor import Wavelength_monitor

# python -m mcvine.acc.run_script --overwrite_datafiles --workdir ./out.chess_test/ tests/chess_test_instrument.py
def instrument():
    Ei = 81.7
    
    fhToF=(2286.3*(29.3))/math.sqrt(Ei)/1000000.0
    fm1ToF=(2286.3*(30.0-(0.025)/2.0))/math.sqrt(Ei)/1000000.0
    fm2ToF=(2286.3*(30.0+(0.025)/2.0))/math.sqrt(Ei)/1000000.0

    # DiskCombo = 8:
    locdeg="0_18_54_90_126_180_216_270"
    angledeg="8_8_8_8_8_8_8_8"
    nslitsRRM=8

    instrument = mcvine.instrument()
    source = Source_simple(
        'src',
        radius=0.015, 
        dist=0.751, 
        xw=0.03, 
        yh=0.03, 
        E0=Ei, 
        dE=81.2
    )
    instrument.append(source, position=(0,0,0.))

    guide1 = Guide(
        'guide1',
        w1=0.03, h1=0.03, w2=0.03, h2=0.03, l=28.54,
        # TODO: check the below parameters?
        #R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
    )
    instrument.append(guide1, position=(0,0,0.75))

    Ltmp = Wavelength_monitor(name="tmpmon",
                              filename="L_Guide.h5",
                              xwidth=0.1, yheight=0.1,
                              Lmin=0, Lmax=20,
                              nchan=1000
                              )
    instrument.append(Ltmp, position=(0, 0, 29.3))

    Hchopper = MultiDiskChopper(name="Hchopper",
                                slit_center=locdeg, slit_width=angledeg,
                                nslits=nslitsRRM, delta_y=0.32,
                                nu=15, jitter=6E-05,
                                delay=fhToF, radius=0.35,
                                abs_out=True)
    instrument.append(Hchopper, position=(0,0,29.3))

    guide2 = Guide(
        'guide2',
        w1=0.03, h1=0.03, w2=0.03, h2=0.03, l=0.67,
        # TODO: check the below parameters?
        #R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
    )
    instrument.append(guide2, position=(0,0,29.31))

    chopper1 = DiskChopper(name="Mchopper1", theta_0=8, 
                           radius = 0.35,
                           yheight=0.045, nu=-300,
                           nslit=1, jitter=1E-06,
                           delay=fm1ToF)
    instrument.append(chopper1, position=(0,0,30.0 - 0.025 / 2))

    chopper2 = DiskChopper(name="Mchopper2", theta_0=8, 
                           radius=0.35,
                           yheight=0.045, nu=300,
                           nslit=1, jitter=1E-06,
                           delay=fm2ToF)
    instrument.append(chopper2, position=(0,0,30.0 + 0.025 / 2),
                      orientation=(180.0, 0.0, 0.0))

    Lsample = Wavelength_monitor(name="mon",
                                 filename = "L_Sample.h5",
                                 xwidth=0.1, yheight=0.1,
                                 Lmin=0, Lmax=20,
                                 nchan=1000,
                                 #restore_neutron=True,
                                 #nL= 1000
                                 )
    instrument.append(Lsample, position=(0, 0, 30.1))

    return instrument
