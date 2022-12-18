import os
import numpy as np
from numba import cuda
from numba.core import config
from ..config import get_numba_floattype


def makeKernelModule(composite):
    "make cuda device methods for the composite kernel"
    import pickle as pkl, tempfile
    # save composite to be loaded by the "compiled" module
    tmpdir = tempfile.mkdtemp(prefix='composite_kernel_', dir=os.curdir)
    pklpath = os.path.abspath(os.path.join(tmpdir, 'composite.pkl'))
    pkl.dump(composite, open(pklpath, 'wb'))
    # make "compiled" composite module
    nkernels = len(composite.elements())
    element_scatter_method_defs = [
        f'scatter_{ikernel} = kernel_funcs_list[{ikernel}][0]'
        for ikernel in range(nkernels)]
    element_scatter_method_defs = '\n'.join(element_scatter_method_defs)
    lines = []
    args = 'threadindex, rng_states, neutron'
    indent = 4*' '
    for i in range(nkernels-1):
        lead = 'if'
        if i>0: lead = 'elif'
        lines.append(f'{lead} r < cumulative_weights[{i}]:')
        lines.append(f'{indent}ret = scatter_{i}({args})')
        lines.append(f'{indent}iscatter={i}')
    lines.append(f'else:')
    lines.append(f'{indent}scatter_{nkernels-1}({args})')
    lines.append(f'{indent}iscatter={nkernels-1}')
    lines.append(f'neutron[-1]*=weights[iscatter]')
    lines.append(f'return ret')
    if_clause_for_scatter_method = '\n'.join([indent + line for line in lines])
    content = template.format(**locals())
    modulepath = os.path.join(tmpdir, 'compiled_composite.py')
    open(modulepath, 'wt').write(content)
    # average_scattering_coefficient=False):
    return

template = """import os, pickle as pkl
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type

pklpath = {pklpath!r}
composite = pkl.load(open(pklpath, 'rb'))
elements = composite.elements()
Nkernels = len(elements)
weights = [element.weight for element in elements]
cumulative_weights = [weights[0]]
for w in weights[1:]:
    cumulative_weights.append(w+cumulative_weights[-1])

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
"""
