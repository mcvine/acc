#!/usr/bin/env python

import os
import shutil
import pytest
import histogram.hdf as hh
import numpy as np
from mcni import neutron_buffer
from mcvine.acc import test
from mcvine import run_script

thisdir = os.path.dirname(__file__)


def test_cpu():
    instr = os.path.join(thisdir, "fccAl_DGSSXRes_plate_instrument.py")
    outdir = 'out.fccAl_DGSSXRes_plate'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    ncount = 1e5
    run_script.run1(
        instr, outdir,
        ncount=ncount, buffer_size=int(ncount),
        is_acc=False,
    )
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1024), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    instr = os.path.join(thisdir, "fccAl_DGSSXRes_plate_instrument.py")
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    classname = "fccAl_DGSSXRes_plate"
    compare_acc_nonacc(
        classname,
        [],
        {"float32": 4e-10, "float64": 4e-10},
        num_neutrons, debug,
        instr=instr,
        interactive=interactive,
        acc_component_spec=dict(is_acc=True),
        nonacc_component_spec=dict(is_acc=False),
    )
    classname = classname.lower()
    mcvine_cpu_outdir = f"out.debug-mcvine_{classname}_cpu_instrument"
    mcvine_gpu_outdir = f"out.debug-{classname}_gpu_instrument"

    mcvine_hist_fn = os.path.join(
        mcvine_cpu_outdir, "step0", "after_sample.mcv")
    assert os.path.exists(
        mcvine_hist_fn), "Missing histogram {}".format(mcvine_hist_fn)

    import mcvine.components as mc
    cpu_result = mc.sources.NeutronFromStorage(
        name="source", path=mcvine_hist_fn)
    neutrons = neutron_buffer(num_neutrons)
    cpu_result.process(neutrons)

    gpu_result = mc.sources.NeutronFromStorage(name="source", path=os.path.join(
        mcvine_gpu_outdir, "step0", "after_sample.mcv"))
    neutrons_gpu = neutron_buffer(num_neutrons)
    gpu_result.process(neutrons_gpu)

    for i in range(num_neutrons):
        cpu_r = np.linalg.norm(neutrons[i].state.position)
        cpu_v = np.linalg.norm(neutrons[i].state.velocity)
        gpu_r = np.linalg.norm(neutrons_gpu[i].state.position)
        gpu_v = np.linalg.norm(neutrons_gpu[i].state.velocity)

        assert np.isclose(cpu_r, gpu_r)
        assert np.isclose(cpu_v, gpu_v)


def main():
    import journal
    journal.info("instrument").activate()
    # test_cpu()
    test_compare_mcvine(num_neutrons=int(1e2), interactive=True, debug=True)
    return


if __name__ == '__main__':
    main()
