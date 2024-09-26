#!/usr/bin/env python
#
# Jiao Lin <jiao.lin@gmail.com>
#

import os, sys, yaml, warnings, imp, hashlib
from mcni import run_ppsd, run_ppsd_in_parallel
from .components.ComponentBase import ComponentBase
from .components.StochasticComponentBase import StochasticComponentBase

def run(script, workdir, ncount,
        overwrite_datafiles=True,
        ntotalthreads=int(1e6), threads_per_block=512,
        use_buffer=False,
        buffer_size=int(1e8),
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
        m.run(ncount, ntotalthreads=ntotalthreads, threads_per_block=threads_per_block, buffer_size=buffer_size, **kwds)
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

    ms_loop = False
    ms_indent = 0
    ms_comps = []
    ms_loop_ind = -1
    for i, comp in enumerate(comps):
        assert isinstance(comp, ComponentBase), f"{comp} is not a mcvine.acc component"
        if comp.is_multiplescattering:
            ms_comps.append(comp)

    modules = []
    body = []
    for i, comp in enumerate(comps):
        modules.append((comp.__module__, comp.__class__.__name__))
        prefix = (
            "thread_index, rng_states, "
            if isinstance(comp, StochasticComponentBase)
            else ""
        )
        if comp.is_multiplescattering:
            prefix += "out_neutrons, "

        if getattr(comp.__class__, 'requires_neutron_index_in_processing', False):
            prefix += "neutron_index, "

        if ms_loop:
            # TODO: fix for > 1 MS components
            if i>0:
                line = "{}applyTransformation(out_neutrons[ms{}][:3], out_neutrons[ms{}][3:6], rotmats[{}], offsets[{}], r, v)".format(' '*ms_indent, ms_loop_ind, ms_loop_ind, i-1, i-1)
                body.append(line)
            line = "{}propagate{}({} out_neutrons[ms{}], *args{})".format(
                ' '*ms_indent, i, prefix, ms_loop_ind, i)
            body.append(line)
        else:
            if i>0:
                body.append("{}applyTransformation(neutron[:3], neutron[3:6], rotmats[{}], offsets[{}], r, v)".format(' '*ms_indent, i-1, i-1))
            n_ms = f'num_ms{i} = ' if comp.is_multiplescattering else ''
            body.append("{}{}propagate{}({} neutron, *args{})".format(
                ' '*ms_indent, n_ms, i, prefix, i))

        if comp.is_multiplescattering and i + 1 < len(comps):
            # insert a multiple scattering loop for remaining components
            body.append("{}for ms{} in range(num_ms{}):".format(' '*ms_indent, i, i))
            ms_indent += 4
            ms_loop = True
            ms_loop_ind = i
        continue
    module_imports = ['from {} import {} as comp{}'.format(m, c, i) for i, (m, c) in enumerate(modules)]
    module_imports = '\n'.join(module_imports)
    propagate_defs = ['propagate{} = comp{}.propagate'.format(i, i) for i in range(len(comps))]
    propagate_defs = '\n'.join(propagate_defs)
    args = [f'args{i}' for i in range(len(comps))]
    args = ', '.join(args)
    indent = 8*' '
    body = '\n'.join([indent+line for line in body])
    if len(ms_comps) == 0:
        text = compiled_script_template.format(
            script = script,
            module_imports = module_imports,
            propagate_definitions = propagate_defs,
            args=args, propagate_body=body
        )
    else:
        # handle multiple scattering
        max_scattering = 0
        max_scattering_comp = None
        # find the maximum number of scattered neutrons to define the output buffer size
        for comp in ms_comps:
            max_scattering = max(max_scattering, comp.NUM_MULTIPLE_SCATTER)
            max_scattering_comp = comp

        # pre-define the number of neutrons scattered by each multiple-scattering component
        scattering_nums = ['NUM_MS{} = comp{}.NUM_MULTIPLE_SCATTER'.format(i, i) if comp.is_multiplescattering else '' for i, comp in enumerate(comps)]
        for i, comp in enumerate(comps):
            if comp == max_scattering_comp:
                scattering_nums.append("NUM_MS = NUM_MS{}".format(i))
        scattering_nums = '\n'.join(scattering_nums)

        text = compiled_script_ms_template.format(
            script = script,
            module_imports = module_imports,
            num_multiple_scattering = scattering_nums,
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
    indent = 8*' '
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
import numpy as np, numba as nb
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc.neutron import applyTransformation
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype, get_max_registers
NB_FLOAT = get_numba_floattype()
global_tpb = mcvine.acc.config.threads_per_block

{module_imports}

{propagate_definitions}

@cuda.jit(max_registers=get_max_registers(global_tpb))
def process_kernel_no_buffer(
    rng_states, N, n_neutrons_per_thread,
    args
):
    {args}, offsets, rotmats, neutron_counter = args
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    neutron = cuda.local.array(shape=10, dtype=NB_FLOAT)
    r = cuda.local.array(3, dtype=NB_FLOAT)
    v = cuda.local.array(3, dtype=NB_FLOAT)
    for neutron_index in range(start_index, end_index):
        cuda.atomic.add(neutron_counter, 0, 1)
{propagate_body}

from mcvine.acc.components.sources.SourceBase import SourceBase
class _Base(SourceBase): # has to be named Base in definition
    def __init__(self, instrument):
        offsets, rotmats = calcTransformations(instrument)
        self.neutron_counter = neutron_counter = np.zeros(1, dtype=int)
        self.propagate_params = tuple(c.propagate_params for c in instrument.components)
        self.propagate_params += (offsets, rotmats, neutron_counter)
        return
    def propagate(self): return
InstrumentWrapper = _Base
InstrumentWrapper.process_kernel_no_buffer = process_kernel_no_buffer

def run(ncount, ntotalthreads=None, threads_per_block=None, **kwds):
    instrument = loadInstrument(script, **kwds)
    if threads_per_block:
        # update threads per block used in instrument kernel
        global_tpb = threads_per_block
    iw = InstrumentWrapper(instrument)
    iw.process_no_buffer(
        ncount, ntotalthreads=ntotalthreads, threads_per_block=threads_per_block)
    processed = iw.neutron_counter[0]
    assert processed == ncount, (
        "Processed neutron count "+str(processed)
        + " does not match requested neutron count " + str(int(ncount))
    )
    saveMonitorOutputs(instrument, scale_factor=1.0/ncount)
"""

compiled_script_ms_template = """#!/usr/bin/env python

script = {script!r}
from mcvine.acc.run_script import loadInstrument, calcTransformations, saveMonitorOutputs

from numba import cuda
import numpy as np, numba as nb
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc.neutron import applyTransformation
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype, get_max_registers
NB_FLOAT = get_numba_floattype()
global_tpb = mcvine.acc.config.threads_per_block

{module_imports}

{propagate_definitions}

{num_multiple_scattering}

@cuda.jit(max_registers=get_max_registers(global_tpb))
def process_kernel_no_buffer(
    rng_states, N, n_neutrons_per_thread,
    args
):
    {args}, offsets, rotmats, neutron_counter = args
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    neutron = cuda.local.array(shape=10, dtype=NB_FLOAT)
    r = cuda.local.array(3, dtype=NB_FLOAT)
    v = cuda.local.array(3, dtype=NB_FLOAT)
    out_neutrons = cuda.local.array(shape=(NUM_MS,10), dtype=NB_FLOAT)
    for neutron_index in range(start_index, end_index):
        cuda.atomic.add(neutron_counter, 0, 1)
{propagate_body}

from mcvine.acc.components.sources.SourceBase import SourceBase
class _Base(SourceBase): # has to be named Base in definition
    def __init__(self, instrument):
        offsets, rotmats = calcTransformations(instrument)
        self.neutron_counter = neutron_counter = np.zeros(1, dtype=int)
        self.propagate_params = tuple(c.propagate_params for c in instrument.components)
        self.propagate_params += (offsets, rotmats, neutron_counter)
        return
    def propagate(self): return
InstrumentWrapper = _Base
InstrumentWrapper.process_kernel_no_buffer = process_kernel_no_buffer

def run(ncount, ntotalthreads=None, threads_per_block=None, **kwds):
    instrument = loadInstrument(script, **kwds)
    print(instrument)
    if threads_per_block:
        # update threads per block used in instrument kernel
        global_tpb = threads_per_block
    iw = InstrumentWrapper(instrument)
    iw.process_no_buffer(
        ncount, ntotalthreads=ntotalthreads, threads_per_block=threads_per_block)
    processed = iw.neutron_counter[0]
    assert processed == ncount, (
        "Processed neutron count "+str(processed)
        + " does not match requested neutron count " + str(int(ncount))
    )
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


def instrument_kernel(rng_states, N, n_neutrons_per_thread, nblocks, tpb, buffer_size, args):
    '''
    Driver function to run all kernels needed for the instrument
    '''

    {args}, offsets, rotmats = args

    # buffer size should be a power of 2, so get closest one
    assert buffer_size > tpb
    buffer_size = int(2**math.ceil(math.log2(buffer_size)))

    iters = math.ceil(N / buffer_size)
    max_blocks = math.floor( buffer_size / (n_neutrons_per_thread * tpb))
    print(" Total N split into %s iterations of buffer size %s" % (iters, buffer_size))

    blocks_left = nblocks

    i = 0
    while blocks_left > 0:

        # Create neutron device buffer
        neutrons = np.zeros((buffer_size, 10), dtype=np.float64)
        neutrons_d = cuda.to_device(neutrons)

        # adjust launch config for this iter
        nblocks = min(blocks_left, max_blocks)
        print("   iter %s - nblocks = %s" % (i+1, nblocks))

{instrument_body}

        cuda.synchronize()

        blocks_left -= nblocks
        i += 1

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

def run(ncount, ntotalthreads=int(1e6), threads_per_block=512, buffer_size=int(1e6), **kwds):
    instrument = loadInstrument(script, **kwds)

    ntotthreads = min(ncount, int(ntotalthreads))
    nblocks = math.ceil(ntotthreads / threads_per_block)
    actual_nthreads = threads_per_block * nblocks
    n_neutrons_per_thread = math.ceil(ncount / actual_nthreads)
    print("%s blocks, %s threads, %s neutrons per thread" % (nblocks, threads_per_block, n_neutrons_per_thread))
    rng_states = create_xoroshiro128p_states(actual_nthreads, seed=1)

    base = InstrumentBase(instrument)

    instrument_kernel(rng_states, ncount, n_neutrons_per_thread, nblocks, threads_per_block, buffer_size, base.propagate_params)
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

import click
@click.command()
@click.argument("script")
@click.option("--workdir", default="output", help="work directory")
@click.option("--ncount", default=int(1e6), help="neutron count")
@click.option("--overwrite_datafiles", default=False, help="overwrite datafiles", is_flag=True)
@click.option("--total_threads", default=int(1e6), help="number of total CUDA threads")
@click.option("--threads_per_block", default=512, help="number of threads per block")
@click.option("--additional-kargs", default=None, help='addiontal kwd args in a yaml file')
def main(
        script, workdir, ncount,
        overwrite_datafiles=False,
        total_threads=None, threads_per_block=None,
        additional_kargs = None,
):
    if additional_kargs:
        kwds = yaml.safe_load(open(additional_kargs))
    else:
        kwds = dict()
    run(script, workdir, ncount,
        overwrite_datafiles=overwrite_datafiles,
        ntotalthreads=total_threads, threads_per_block=threads_per_block,
        **kwds)
    return

if __name__ == '__main__': main()
