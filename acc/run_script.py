#
# Jiao Lin <jiao.lin@gmail.com>
#

import os, sys, yaml, warnings
from mcni import run_ppsd, run_ppsd_in_parallel
from .components.StochasticComponentBase import StochasticComponentBase

def run(script, workdir, ncount, **kwds):
    """run a mcvine.acc simulation script on one node. The script must define the instrument.

Parameters:

* script: path to instrument script. the script must either create an instrument or provide a method to do so
* workdir: working dir
* ncount: neutron count

"""
    kwds_fn = _saveKwds(kwds)
    instrument = loadInstrument(script, **kwds)
    comps = instrument.components
    ncount = int(ncount)
    workdir = os.path.abspath(workdir)
    nargs = 0
    modules = []
    body = []
    for i, comp in enumerate(comps):
        nargs += len(comp.propagate_params)
        modules.append(comp.__module__)
        prefix = (
            "thread_index, rng_states, "
            if isinstance(comp, StochasticComponentBase)
            else ""
        )
        if i>0:
            body.append("vec3.copy(neutron[:3], r); vec3.copy(neutron[3:6], v)")
            body.append("offset, rotmat = offsets[{}], rotmats[{}]".format(i-1, i-1))
            body.append("vec3.abs2rel(r, rotmat, offset, neutron[:3])")
            body.append("vec3.mXv(rotmat, v, neutron[3:6])")
        body.append("compmod{}.propagate({} neutron, *args{})".format(i, prefix, i))
        continue
    module_imports = ['import {} as compmod{}'.format(m, i) for i, m in enumerate(modules)]
    module_imports = '\n'.join(module_imports)
    args = [f'args{i}' for i in range(len(comps))]
    args = ', '.join(args)
    indent = 8*' '
    body = '\n'.join([indent+line for line in body])
    text = cuda_mod_template.format(
        script = script, kwds_fn = kwds_fn,
        module_imports = module_imports,
        args=args, propagate_body=body
    )
    cuda_script_fn = "cuda_" + os.path.basename(script)
    with open(cuda_script_fn, 'wt') as stream:
        stream.write(text)
    return

def calcTransformations(instrument):
    """given a mcni.Instrument instance, calculate transformation matrices and
    offset vectors from one component to the next
    """
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
    return offsets, rotmats


cuda_mod_template = """
script = {script!r}
kwds_fn = {kwds_fn!r}
import pickle
kwds = pickle.load(open(kwds_fn, 'rb'))
from mcvine.acc.run_script import loadInstrument, calcTransformations
instrument = loadInstrument(script, **kwds)
offsets, rotmats = calcTransformations(instrument)

from numba import cuda
import numba as nb
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc import vec3
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()

{module_imports}

@cuda.jit
def process_kernel_no_buffer(
    rng_states, N, n_neutrons_per_thread,
    {args}
):
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    neutron = cuda.local.array(shape=10, dtype=NB_FLOAT)
    r = cuda.local.array(3, dtype=NB_FLOAT)
    v = cuda.local.array(3, dtype=NB_FLOAT)
    for i in range(start_index, end_index):
{propagate_body}

from mcvine.acc.components.sources.SourceBase import SourceBase
class Instrument(SourceBase):

    def __init__(self):
        self.propagate_params = [c.propagate_params for c in instrument.components]
        return

Instrument.process_kernel_no_buffer = process_kernel_no_buffer

def run(ncount):
    Instrument().process_no_buffer(ncount)
"""

def loadInstrument(script, **kwds):
    import imp
    m = imp.load_source('mcvinesim', script)
    assert hasattr(m, 'instrument')
    instrument = m.instrument
    from mcni.Instrument import Instrument
    if not isinstance(instrument, Instrument):
        assert callable(instrument) # has to be a  method that creates instrument
        instrument = instrument(**_getRelevantKwds(instrument, kwds))
    return instrument

def _saveKwds(kwds):
    import pickle, tempfile
    f = tempfile.NamedTemporaryFile(delete=False)
    kwds_fn = f.name
    pickle.dump(kwds, f)
    f.close()
    return kwds_fn

def _getRelevantKwds(method, kwds):
    """return kwd args for the given method, and remove them from the given kwds"""
    import inspect
    argspec = inspect.getargspec(method)
    d = dict()
    for a in kwds:
        if a not in argspec.args:
            warnings.warn("Unrecognized kwd: {!r}".format(a))
    for a in argspec.args:
        if a in kwds:
            d[a] = kwds[a]
            del kwds[a]
    return d
