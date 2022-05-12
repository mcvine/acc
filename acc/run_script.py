#
# Jiao Lin <jiao.lin@gmail.com>
#

import os, sys, yaml, warnings, imp, hashlib
from mcni import run_ppsd, run_ppsd_in_parallel
from .components.StochasticComponentBase import StochasticComponentBase

def run(script, workdir, ncount,
        overwrite_datafiles=True,
        ntotalthreads=int(1e6), threads_per_block=512,
        use_buffer=False,
        **kwds):
    """run a mcvine.acc simulation script on one node. The script must define the instrument.

Parameters:

* script: path to instrument script. the script must either create an instrument or provide a method to do so
* workdir: working dir
* ncount: neutron count

"""
    from mcvine.run_script import _check_workdir
    _check_workdir(workdir, overwrite_datafiles)
    os.makedirs(workdir)
    curdir = os.path.abspath(os.curdir)
    compiled_script_path = os.path.join(workdir, 'compiled_mcvine_acc_instrument.py')
    if use_buffer:
        compiled_script = compile_buffered(script,
                                           compiled_script=compiled_script_path,
                                           **kwds)
    else:
        compiled_script = compile(script, compiled_script=compiled_script_path,
                                  **kwds)
    m = imp.load_source('mcvinesim', compiled_script)
    os.chdir(workdir)
    try:
        m.run(ncount, ntotalthreads=ntotalthreads, threads_per_block=threads_per_block, **kwds)
    finally:
        os.chdir(curdir)
    return

def compile(script, compiled_script=None, **kwds):
    """compile a mcvine.acc simulation script. The script must define the instrument.

Parameters:

* script: path to instrument script. the script must either create an instrument or provide a method to do so
"""
    script = os.path.abspath(script)
    instrument = loadInstrument(script, **kwds)
    comps = instrument.components
    modules = []
    body = []
    for i, comp in enumerate(comps):
        modules.append((comp.__module__, comp.__class__.__name__))
        prefix = (
            "thread_index, rng_states, "
            if isinstance(comp, StochasticComponentBase)
            else ""
        )
        if i>0:
            body.append("abs2rel(neutron[:3], neutron[3:6], rotmats[{}], offsets[{}], r, v)".format(i-1, i-1))
        body.append("propagate{}({} neutron, *args{})".format(i, prefix, i))
        continue
    module_imports = ['from {} import {} as comp{}'.format(m, c, i) for i, (m, c) in enumerate(modules)]
    module_imports = '\n'.join(module_imports)
    propagate_defs = ['propagate{} = comp{}.propagate'.format(i, i) for i in range(len(comps))]
    propagate_defs = '\n'.join(propagate_defs)
    args = [f'args{i}' for i in range(len(comps))]
    args = ', '.join(args)
    indent = 8*' '
    body = '\n'.join([indent+line for line in body])
    text = compiled_script_template.format(
        script = script,
        module_imports = module_imports,
        propagate_definitions = propagate_defs,
        args=args, propagate_body=body
    )
    if compiled_script is None:
        f, ext = os.path.splitext(script)
        kwds_str = str(kwds)
        uid = hashlib.sha224(kwds_str.encode("UTF-8")).hexdigest()[:8]
        compiled_script = f + "_compiled_" + uid + ext
    with open(compiled_script, 'wt') as stream:
        stream.write(text)
    return compiled_script


def compile_buffered(script, compiled_script=None, **kwds):
    script = os.path.abspath(script)
    instrument = loadInstrument(script, **kwds)
    comps = instrument.components
    modules = []
    body = []
    for i, comp in enumerate(comps):
        modules.append((comp.__module__, comp.__class__.__name__))
        prefix = (
            "rng_states, "
            if isinstance(comp, StochasticComponentBase)
            else ""
        )
        if i>0:
            body.append("transform_kernel[nblocks, tpb](neutrons_d, n_neutrons_per_thread, rotmats[{}], offsets[{}])".format(i-1, i-1))
        body.append("process_kernel{}[nblocks, tpb]({} neutrons_d, n_neutrons_per_thread, args[{}])".format(i, prefix, i))
        continue
    module_imports = ['from {} import {} as comp{}'.format(m, c, i) for i, (m, c) in enumerate(modules)]
    module_imports = '\n'.join(module_imports)
    propagate_defs = ['process_kernel{} = comp{}.process_kernel'.format(i, i) for i in range(len(comps))]
    propagate_defs = '\n'.join(propagate_defs)
    args = [f'args{i}' for i in range(len(comps))]
    args = ', '.join(args)
    indent = 4*' '
    body = '\n'.join([indent+line for line in body])
    text = compiled_script_template_buffered.format(
        script = script,
        module_imports = module_imports,
        propagate_definitions = propagate_defs,
        args=args, instrument_body=body
    )
    if compiled_script is None:
        f, ext = os.path.splitext(script)
        kwds_str = str(kwds)
        uid = hashlib.sha224(kwds_str.encode("UTF-8")).hexdigest()[:8]
        compiled_script = f + "_compiled_" + uid + ext
    with open(compiled_script, 'wt') as stream:
        stream.write(text)
    return compiled_script


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


compiled_script_template = """#!/usr/bin/env python

script = {script!r}
from mcvine.acc.run_script import loadInstrument, calcTransformations, saveMonitorOutputs

from numba import cuda
import numba as nb
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc.neutron import abs2rel
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()

{module_imports}

{propagate_definitions}

@cuda.jit
def process_kernel_no_buffer(
    rng_states, N, n_neutrons_per_thread,
    args
):
    {args}, offsets, rotmats = args
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    neutron = cuda.local.array(shape=10, dtype=NB_FLOAT)
    r = cuda.local.array(3, dtype=NB_FLOAT)
    v = cuda.local.array(3, dtype=NB_FLOAT)
    for i in range(start_index, end_index):
{propagate_body}

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

def run(ncount, ntotalthreads=None, threads_per_block=None, **kwds):
    instrument = loadInstrument(script, **kwds)
    InstrumentBase(instrument).process_no_buffer(
        ncount, ntotalthreads=ntotalthreads, threads_per_block=threads_per_block)
    saveMonitorOutputs(instrument, scale_factor=1.0/ncount)
"""

compiled_script_template_buffered = """#!/usr/bin/env python

script = {script!r}
from mcvine.acc.run_script import loadInstrument, calcTransformations, saveMonitorOutputs

from numba import cuda
import numba as nb
import numpy as np
import math
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc.neutron import abs2rel
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()

{module_imports}

{propagate_definitions}


@cuda.jit()
def transform_kernel(neutrons, n_neutrons_per_thread, rotmat, offset):
    '''
    Kernel to adjust neutrons between two components
    '''
    N = len(neutrons)
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    r = cuda.local.array(3, dtype=NB_FLOAT)
    v = cuda.local.array(3, dtype=NB_FLOAT)
    for i in range(start_index, end_index):
        neutron = neutrons[i]
        abs2rel(neutron[:3], neutron[3:6], rotmat, offset, r, v)


def instrument_kernel(rng_states, N, n_neutrons_per_thread, nblocks, tpb, args):
    '''
    Driver function to run all kernels needed for the instrument
    '''

    {args}, offsets, rotmats = args

    # Create neutron device buffer
    neutrons = np.zeros((N, 10), dtype=np.float64)
    neutrons_d = cuda.to_device(neutrons)

{instrument_body}

    cuda.synchronize()

    #neutrons = neutrons_d.copy_to_host()


from mcvine.acc.components.sources.SourceBase import SourceBase
class InstrumentBase(SourceBase):
    def __init__(self, instrument):
        offsets, rotmats = calcTransformations(instrument)
        self.propagate_params = tuple(c.propagate_params for c in instrument.components)
        self.propagate_params += (offsets, rotmats)
        return
    def propagate(self):
        pass
InstrumentBase.process_kernel_no_buffer = instrument_kernel

def run(ncount, ntotalthreads=int(1e6), threads_per_block=512, **kwds):
    instrument = loadInstrument(script, **kwds)

    ntotthreads = min(ncount, int(ntotalthreads))
    nblocks = math.ceil(ntotthreads / threads_per_block)
    actual_nthreads = threads_per_block * nblocks
    n_neutrons_per_thread = math.ceil(ncount / actual_nthreads)
    print("%s blocks, %s threads, %s neutrons per thread" % (nblocks, threads_per_block, n_neutrons_per_thread))
    rng_states = create_xoroshiro128p_states(actual_nthreads, seed=1)

    base = InstrumentBase(instrument)

    instrument_kernel(rng_states, ncount, n_neutrons_per_thread, nblocks, threads_per_block, base.propagate_params)
    cuda.synchronize()

    saveMonitorOutputs(instrument, scale_factor=1.0/ncount)
"""


def saveMonitorOutputs(instrument, scale_factor=1.0):
    for comp in instrument.components:
        from .components.monitors.MonitorBase import MonitorBase
        if isinstance(comp, MonitorBase):
            comp.save(scale_factor=scale_factor)

def loadInstrument(script, **kwds):
    m = imp.load_source('mcvinesim', script)
    assert hasattr(m, 'instrument')
    instrument = m.instrument
    from mcni.Instrument import Instrument
    if not isinstance(instrument, Instrument):
        assert callable(instrument) # has to be a  method that creates instrument
        kwds = _getRelevantKwds(instrument, kwds)
        instrument = instrument(**kwds)
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
