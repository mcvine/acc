#!/usr/bin/env python

import mcvine.components as mc
import mcvine
from mcvine.acc.components.optics.disk_chopper import DiskChopper
from mcvine.acc.components.sources.source_simple import Source_simple
from mcvine.acc.components.monitors.wavelength_monitor import Wavelength_monitor
import os
import sys
thisdir = os.path.abspath(os.path.dirname(__file__))
if thisdir not in sys.path:
    sys.path.insert(0, thisdir)


def source(ctor, Ei):
    return ctor(
        name='src', radius=0.015, dist=0.751,
        xw=0.03, yh=0.03, E0=Ei, dE=50
    )


def source_cpu(Ei): return source(mc.sources.Source_simple, Ei)
def source_gpu(Ei): return source(Source_simple, Ei)


def chopper(ctor, name='diskchopper'):
    return ctor(name=name, theta_0=8, radius=0.35,
                yheight=0.045, nu=-300, nslit=1, jitter=1E-06)


def chopper_cpu(name): return chopper(mc.optics.DiskChopper_v2, name)
def chopper_gpu(name): return chopper(DiskChopper, name)

'''
def monitor(ctor):
    return ctor(name='monitor', nchan = 1000, xwidth = 0.1,
                yheight = 0.1, Lmin = 0, Lmax = 20, filename="IL.h5")

def monitor_cpu(): return monitor(mc.monitors.L_monitor)
def monitor_gpu(): return monitor(Wavelength_monitor)
'''

def monitor(ctor):
    return ctor('mon', nx=250, ny=250, xwidth=0.1, yheight=0.1, filename="psd.h5")

def monitor_cpu(): return monitor(mc.monitors.PSD_monitor)

def instrument(is_acc=False, Ei=81.7):
    instrument = mcvine.instrument()

    if is_acc:
        instrument.append(source_gpu(Ei), position=(0, 0, 0))
        instrument.append(chopper_gpu('chopper1'), position=(0, 0, 30.0-0.025/2))
        instrument.append(chopper_gpu('chopper2'), position=(0, 0, 30.0+0.025/2))
    else:
        instrument.append(source_cpu(Ei), position=(0, 0, 0))
        instrument.append(chopper_cpu('chopper1'), position=(0, 0, 30.0-0.025/2))
        instrument.append(chopper_cpu('chopper2'), position=(0, 0, 30.0+0.025/2))
    instrument.append(monitor_cpu(), position=(0, 0, 30.1))

    return instrument
