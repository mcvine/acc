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
    return m.scatter, m.scattering_coeff, m.absorb, m.absorption_coeff

_modules = {}
def makeKernelModule(composite):
    "make cuda device methods for the composite kernel"
    from .._numba import coder
    if composite in _modules: return _modules[composite]
    import pickle as pkl, tempfile
    # save composite to be loaded by the "compiled" module
    tmpdir = coder.createUniqueDir("composite_kernel_")
    pklpath = os.path.abspath(os.path.join(tmpdir, 'composite.pkl'))
    pkl.dump(composite, open(pklpath, 'wb'))
    # make "compiled" composite module
    nkernels = len(composite.elements())
    #
    indent = 4*' '
    # 1. scatter
    element_scatter_method_defs = [
        f'scatter_{ik}, scattering_coeff_{ik}, absorb_{ik}, absorption_coeff_{ik} = kernel_funcs_list[{ik}]'
        for ik in range(nkernels)
    ]
    element_scatter_method_defs = '\n'.join(element_scatter_method_defs)
    scatter_method = '\n'.join(_create_scatter_method(nkernels, indent=indent))
    # 2. scattering_coeff
    scattering_coeff_method = '\n'.join(_create_scattering_coeff_method(nkernels, indent=4*' '))
    # 3. absorb
    absorb_method = '\n'.join(_create_absorb_method(nkernels, indent=indent))
    # 4. absorption_coeff
    abs_coeff_method = '\n'.join(_create_abs_coeff_method(nkernels, indent=4*' '))
    # all together
    content = template.format(**locals())
    modulepath = os.path.join(tmpdir, 'compiled_composite.py')
    open(modulepath, 'wt').write(content)
    _modules[composite] = modulepath
    return modulepath

def _create_scattering_coeff_method(nkernels, indent=4*' '):
    header = [
        '@cuda.jit(device=True)',
        'def scattering_coeff(neutron):'
    ]
    from .._numba import coder
    loop = coder.unrollLoop(
        N = nkernels,
        indent = indent,
        before_loop = ['r = 0.0'],
        in_loop = ['r += scattering_coeff_{i}(neutron)'],
        after_loop = ['return r*scale_scattering_coeff']
    )
    return header + loop

def _create_scatter_method(nkernels, indent=4*' '):
    header = [
        '@cuda.jit(device=True)',
        'def scatter(threadindex, rng_states, neutron):'
    ]
    loop = _create_select_kernel_func_lines(
        nkernels,
        method='scatter',
        args = 'threadindex, rng_states, neutron',
        indent = indent
    )
    return header + loop

def _create_abs_coeff_method(nkernels, indent=4*' '):
    header = [
        '@cuda.jit(device=True)',
        'def absorption_coeff(neutron):'
    ]
    from .._numba import coder
    loop = coder.unrollLoop(
        N = nkernels,
        indent = indent,
        before_loop = ['r = 0.0'],
        in_loop = ['r += absorption_coeff_{i}(neutron)'],
        after_loop = ['return r/Nkernels']
    )
    return header + loop

def _create_absorb_method(nkernels, indent=4*' '):
    header = [
        '@cuda.jit(device=True)',
        'def absorb(threadindex, rng_states, neutron):'
    ]
    loop = _create_select_kernel_func_lines(
        nkernels,
        method='absorb',
        args = 'threadindex, rng_states, neutron',
        indent = indent
    )
    return header + loop

def _create_select_kernel_func_lines(nkernels, method, args, indent=4*' '):
    from .._numba import coder
    return coder.unrollLoop(
        N = nkernels,
        indent = indent,
        before_loop = [
            'r = xoroshiro128p_uniform_float32(rng_states, threadindex)'
        ],
        in_loop = [
            'if r < cumulative_weights[{i}]:',
            f'    ret = {method}_'+'{i}'+f'({args})',
            '    neutron[-1]/=weights[{i}]',
            '    return ret'
        ],
        after_loop = []
    )

template = """import os, pickle as pkl, numpy as np
from numba import cuda
from mcvine.acc._numba import xoroshiro128p_uniform_float32

pklpath = {pklpath!r}
composite = pkl.load(open(pklpath, 'rb'))
elements = composite.elements()
Nkernels = len(elements)
weights = [float(element.weight) for element in elements]
cumulative_weights = [weights[0]]
for w in weights[1:]:
    cumulative_weights.append(w+cumulative_weights[-1])
cumulative_weights = np.array(cumulative_weights)
weights = np.array(weights)
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

{scatter_method}

{absorb_method}

{scattering_coeff_method}

{abs_coeff_method}
"""

# Example code generated
"""
import os, pickle as pkl, numpy as np
from numba import cuda
from mcvine.acc._numba import xoroshiro128p_uniform_float32

pklpath = '/home/username/mcvinesim/.mcvine.acc.coder/composite_kernel_g9xcahde/composite.pkl'
composite = pkl.load(open(pklpath, 'rb'))
elements = composite.elements()
Nkernels = len(elements)
weights = [float(element.weight) for element in elements]
cumulative_weights = [weights[0]]
for w in weights[1:]:
    cumulative_weights.append(w+cumulative_weights[-1])
cumulative_weights = np.array(cumulative_weights)
weights = np.array(weights)
scale_scattering_coeff = 1./Nkernels if composite.average else 1

# create cuda functions for kernels
from mcvine.acc.kernels import scatter_func_factory
kernel_funcs_list = []
for element in elements:
    kernel_funcs = scatter_func_factory.render(element)
    kernel_funcs_list.append(kernel_funcs)
    continue

# and assign names
scatter_0, scattering_coeff_0, absorb_0, absorption_coeff_0 = kernel_funcs_list[0]
scatter_1, scattering_coeff_1, absorb_1, absorption_coeff_1 = kernel_funcs_list[1]

@cuda.jit(device=True)
def scatter(threadindex, rng_states, neutron):
    r = xoroshiro128p_uniform_float32(rng_states, threadindex)
    if r < cumulative_weights[0]:
        ret = scatter_0(threadindex, rng_states, neutron)
        neutron[-1]/=weights[0]
        return ret
    if r < cumulative_weights[1]:
        ret = scatter_1(threadindex, rng_states, neutron)
        neutron[-1]/=weights[1]
        return ret

@cuda.jit(device=True)
def absorb(threadindex, rng_states, neutron):
    r = xoroshiro128p_uniform_float32(rng_states, threadindex)
    if r < cumulative_weights[0]:
        ret = absorb_0(threadindex, rng_states, neutron)
        neutron[-1]/=weights[0]
        return ret
    if r < cumulative_weights[1]:
        ret = absorb_1(threadindex, rng_states, neutron)
        neutron[-1]/=weights[1]
        return ret

@cuda.jit(device=True)
def scattering_coeff(neutron):
    r = 0.0
    r += scattering_coeff_0(neutron)
    r += scattering_coeff_1(neutron)
    return r*scale_scattering_coeff

@cuda.jit(device=True)
def absorption_coeff(neutron):
    r = 0.0
    r += absorption_coeff_0(neutron)
    r += absorption_coeff_1(neutron)
    return r/Nkernels
"""
