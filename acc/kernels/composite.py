import os
import numpy as np
from numba import cuda
from numba.core import config
from ..config import get_numba_floattype

def makeKernelMethods(composite):
    # average_scattering_coefficient=False):
    mod = makeKernelModule(composite)
    import imp
    m = imp.load_source('composite', mod)
    return m.scatter, m.scattering_coeff, m.absorb

_modules = {}
def makeKernelModule(composite):
    "make cuda device methods for the composite kernel"
    if composite in _modules: return _modules[composite]
    import pickle as pkl, tempfile
    # save composite to be loaded by the "compiled" module
    tmpdir = tempfile.mkdtemp(prefix='composite_kernel_', dir=os.curdir)
    pklpath = os.path.abspath(os.path.join(tmpdir, 'composite.pkl'))
    pkl.dump(composite, open(pklpath, 'wb'))
    # make "compiled" composite module
    nkernels = len(composite.elements())
    # 1. scatter
    element_scatter_method_defs = [
        f'scatter_{ikernel}, scattering_coeff_{ikernel}, absorb_{ikernel} = kernel_funcs_list[{ikernel}]'
        for ikernel in range(nkernels)]
    element_scatter_method_defs = '\n'.join(element_scatter_method_defs)
    indent = 4*' '
    lines = _create_select_kernel_func_lines(
        nkernels,
        method='scatter',
        args = 'threadindex, rng_states, neutron',
        indent = indent
    )
    if_clause_for_scatter_method = '\n'.join([indent + line for line in lines])
    # 2. scattering_coeff
    lines = [f'r += scattering_coeff_{i}(neutron)' for i in range(nkernels)]
    add_scattering_coeff = '\n'.join(indent+l for l in lines)
    # 3. absorb
    lines = _create_select_kernel_func_lines(
        nkernels,
        method='absorb',
        args = 'neutron',
        indent = indent
    )
    if_clause_for_absorb_method = '\n'.join([indent + line for line in lines])
    # all together
    content = template.format(**locals())
    modulepath = os.path.join(tmpdir, 'compiled_composite.py')
    open(modulepath, 'wt').write(content)
    _modules[composite] = modulepath
    return modulepath

def _create_select_kernel_func_lines(nkernels, method, args, indent=4*' '):
    lines = []
    for i in range(nkernels-1):
        lead = 'if'
        if i>0: lead = 'elif'
        lines.append(f'{lead} r < cumulative_weights[{i}]:')
        lines.append(f'{indent}ret = {method}_{i}({args})')
        lines.append(f'{indent}ikernel={i}')
    lines.append(f'else:')
    lines.append(f'{indent}ret = {method}_{nkernels-1}({args})')
    lines.append(f'{indent}ikernel={nkernels-1}')
    lines.append(f'neutron[-1]/=weights[ikernel]')
    lines.append(f'return ret')
    return lines

template = """import os, pickle as pkl
from numba import cuda
from mcvine.acc._numba import xoroshiro128p_uniform_float32

pklpath = {pklpath!r}
composite = pkl.load(open(pklpath, 'rb'))
elements = composite.elements()
Nkernels = len(elements)
weights = [element.weight for element in elements]
cumulative_weights = [weights[0]]
for w in weights[1:]:
    cumulative_weights.append(w+cumulative_weights[-1])
scale_scattering_coeff = 1./Nkernels if composite.average else 1

# create cuda functions for kernels
from mcvine.acc.kernels import scatter_func_factory
kernel_funcs_list = []
for element in elements:
    kernel_funcs = scatter_func_factory.render(element)
    kernel_funcs_list.append(kernel_funcs)
    continue

# and assign names
{element_scatter_method_defs}

@cuda.jit(device=True)
def scatter(threadindex, rng_states, neutron):
    r = xoroshiro128p_uniform_float32(rng_states, threadindex)
{if_clause_for_scatter_method}

@cuda.jit(device=True)
def scattering_coeff(neutron):
    r = 0.0
{add_scattering_coeff}
    return r*scale_scattering_coeff

@cuda.jit(device=True)
def absorb(neutron):
    r = xoroshiro128p_uniform_float32(rng_states, threadindex)
{if_clause_for_absorb_method}
"""
