
script = '/home/97n/dv/mcvine/acc/tests/acc_sgm_instrument.py'
kwds_fn = '/tmp/tmpzpc2nq6m'
import pickle
kwds = pickle.load(open(kwds_fn, 'rb'))
from mcvine.acc.run_script import loadInstrument
instrument = loadInstrument(script, **kwds)

from mcni.instrument_simulator import default_simulator as ds
nct = ds.neutron_coordinates_transformer
comps = instrument.components
geometer = instrument.geometer
transformations = [
    nct.relativePositionOrientation(
        geometer.position(comps[i]), geometer.orientation(comps[i]),
        geometer.position(comps[i+1]), geometer.orientation(comps[i+1]),
    ) for i in range(len(comps)-1)
]
offsets = []; rotmats = []
for offset, rotmat in transformations:
    offsets.append(offset); rotmats.append(rotmat)
    print(offset); print(rotmat)
    continue
import numpy as np
offsets = np.array(offsets); rotmats = np.array(rotmats)

from numba import cuda
import numba as nb
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc import vec3
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()

import mcvine.acc.components.sources.source_simple as compmod0
import mcvine.acc.components.optics.guide as compmod1
import mcvine.acc.components.monitors.divpos_monitor as compmod2

@cuda.jit
def process_kernel_no_buffer(
    rng_states, N, n_neutrons_per_thread,
    args0, args1, args2
):
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    neutron = cuda.local.array(shape=10, dtype=NB_FLOAT)
    r = cuda.local.array(3, dtype=NB_FLOAT)
    v = cuda.local.array(3, dtype=NB_FLOAT)
    for i in range(start_index, end_index):
        compmod0.propagate(thread_index, rng_states,  neutron, *args0)
        vec3.copy(neutron[:3], r); vec3.copy(neutron[3:6], v)
        offset, rotmat = offsets[0], rotmats[0]
        vec3.abs2rel(r, rotmat, offset, neutron[:3])
        vec3.mXv(rotmat, v, neutron[3:6])
        compmod1.propagate( neutron, *args1)
        vec3.copy(neutron[:3], r); vec3.copy(neutron[3:6], v)
        offset, rotmat = offsets[1], rotmats[1]
        vec3.abs2rel(r, rotmat, offset, neutron[:3])
        vec3.mXv(rotmat, v, neutron[3:6])
        compmod2.propagate( neutron, *args2)

from mcvine.acc.components.sources.SourceBase import SourceBase
class Instrument(SourceBase):

    def __init__(self):
        self.propagate_params = [c.propagate_params for c in comps]
        return

Instrument.process_kernel_no_buffer = process_kernel_no_buffer

def run(ncount):
    Instrument().process_no_buffer(ncount)
def main():
    N = 5
    N = 1e8
    run(N)
    mon1 = comps[-1]
    from matplotlib import pyplot as plt
    plt.pcolormesh(mon1.x_centers, mon1.div_centers, mon1.out_p/N)
    plt.colorbar()
    plt.clim(0, 1e-6)
    plt.show()
    return

if __name__ == '__main__': main()
