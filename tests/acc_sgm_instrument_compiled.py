#!/usr/bin/env python

script = '/home/97n/dv/mcvine/acc/tests/acc_sgm_instrument.py'
from mcvine.acc.run_script import loadInstrument, calcTransformations, saveMonitorOutputs

from numba import cuda
import numba as nb
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc.neutron import abs2rel
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()

from mcvine.acc.components.sources.source_simple import Source_simple as comp0
from mcvine.acc.components.optics.guide import Guide as comp1
from mcvine.acc.components.monitors.divpos_monitor import DivPos_monitor as comp2

propagate0 = comp0.propagate
propagate1 = comp1.propagate
propagate2 = comp2.propagate

@cuda.jit
def process_kernel_no_buffer(
    rng_states, N, n_neutrons_per_thread,
    args
):
    args0, args1, args2, offsets, rotmats = args
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    neutron = cuda.local.array(shape=10, dtype=NB_FLOAT)
    r = cuda.local.array(3, dtype=NB_FLOAT)
    v = cuda.local.array(3, dtype=NB_FLOAT)
    for i in range(start_index, end_index):
        propagate0(thread_index, rng_states,  neutron, *args0)
        abs2rel(neutron[:3], neutron[3:6], rotmats[0], offsets[0], r, v)
        propagate1( neutron, *args1)
        abs2rel(neutron[:3], neutron[3:6], rotmats[1], offsets[1], r, v)
        propagate2( neutron, *args2)

from mcvine.acc.components.sources.SourceBase import SourceBase
class InstrumentBase(SourceBase):
    def __init__(self, instrument):
        offsets, rotmats = calcTransformations(instrument)
        self.propagate_params = tuple(c.propagate_params for c in instrument.components)
        self.propagate_params += (offsets, rotmats)
        return
    def propagate(self):
        pass
InstrumentBase.process_kernel_no_buffer = process_kernel_no_buffer

def run(ncount, **kwds):
    instrument = loadInstrument(script, **kwds)
    InstrumentBase(instrument).process_no_buffer(ncount)
    saveMonitorOutputs(instrument, scale_factor=1.0/ncount)
