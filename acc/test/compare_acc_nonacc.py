#!/usr/bin/env python

from mcvine.acc import config
# config.floattype = "float32"

import os
import histogram.hdf as hh
import numpy as np
import shutil
from mcvine import run_script
from mcvine.acc import test
from mcvine.acc.config import floattype

def compare_acc_nonacc(
        className, monitors, tolerances, num_neutrons, debug,
        workdir=None, instr=None, interactive=False,
        acc_component_spec = None, nonacc_component_spec = None,
        relerr_tolerances = None
):
    """
    Tests the acc implementation of an instrument against mcvine.
    Requires existence of instrument spec module for testing the component
    named {classname}_instrument.py.
    The instrument module should have `instrument` method taking
    `is_acc` as a kw arg to differenciate between using the acc or the non-acc
    component.
    By default, `factory` kw arg is also passed to the instrument method
    but it is OK to ignore.
    Alternatively, the instrument method can implement arbitrary args,
    and acc_component_spec and nonacc_component_spec here can be used
    to differentiate the acc and non-acc simulation runs.

    Parameters:
    className (str): the name of the instrument class under test, e.g., "Guide"
    monitors (list): the names of the monitors to use in testing, e.g., "Ixy"
    tolerances (dict): tolerance of float comparisons, e.g., {"float32": 1e-8}
    relerr_tolerances (dict): tolerances for relative error.
        e.g. dict(float32=dict(threshold=0.05, outlier_fraction=0.05))
    num_neutrons (int): how many neutrons to use in the testing
    debug (bool): if to save the neutrons that exit the instrument
    """
    if debug:
        assert num_neutrons < 1001
    classname = className.lower()

    # Run the mcvine instrument first
    instr = instr or os.path.join(workdir, f"{classname}_instrument.py")
    mcvine_outdir = f"out.debug-mcvine_{classname}_cpu_instrument"
    if os.path.exists(mcvine_outdir):
        shutil.rmtree(mcvine_outdir)
    # there is inconsistency in the simulation instrument implementations.
    # for example, tests/components/optics/guide_instrument.py
    # implements "is_acc", "factory" and "module" kwargs,
    # but tests/components/optics/slit_instrument.py
    # only implements "is_acc".
    # The implementation here works, but we just need to make sure
    # the {classname}_instrument.py implements "is_acc" kwd correctly.
    nonacc_component_spec = nonacc_component_spec or dict(
        factory=f"mcvine.components.optics.{className}",
        is_acc = False,
    )
    run_script.run1(
        instr, mcvine_outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        save_neutrons_after=debug,
        overwrite_datafiles=True,
        **nonacc_component_spec
    )
    # Run our accelerated implementation
    outdir = f"out.debug-{classname}_gpu_instrument"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    acc_component_spec = acc_component_spec or dict(
        module=f"mcvine.acc.components.optics.{classname}",
        is_acc = True,
    )
    run_script.run1(
        instr, outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        save_neutrons_after=debug,
        overwrite_datafiles=True,
        **acc_component_spec
    )

    # Compare output files
    tolerance = tolerances[floattype]
    for monitor in monitors:
        mcvine_hist_fn = os.path.join(mcvine_outdir, monitor + ".h5")
        assert os.path.exists(mcvine_hist_fn), "Missing histogram {}".format(mcvine_hist_fn)
        mcvine = hh.load(mcvine_hist_fn)
        mcvine_acc = hh.load(os.path.join(outdir, monitor + ".h5"))
        relerr_hist = (mcvine_acc-mcvine)/mcvine
        if interactive:
            from histogram import plot as plotHist
            plotHist(mcvine)
            plotHist(mcvine_acc)
            plotHist(relerr_hist)
        assert mcvine.shape() == mcvine_acc.shape()
        if tolerance is not None:
            assert np.allclose(mcvine.I, mcvine_acc.I, atol=tolerance)
        if relerr_tolerances is not None:
            relerr = np.abs(relerr_hist.I)
            relerr_tolerance = relerr_tolerances[floattype]
            threshold = relerr_tolerance['threshold']
            outlier_fraction = relerr_tolerance['outlier_fraction']
            assert (relerr>threshold).sum() < outlier_fraction * relerr.size
