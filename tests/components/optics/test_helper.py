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

thisdir = os.path.dirname(__file__)
interactive = False


def compare_mcvine(className, monitors, tolerances, num_neutrons, debug):
    """
    Tests the acc cpu implementation of an instrument against mcvine.

    Parameters:
    className (str): the name of the instrument class under test, e.g., "Guide"
    monitors (list): the names of the monitors to use in testing, e.g., "Ixy"
    tolerances (dict): tolerance of float comparisons, e.g., "float32": 1e-8
    num_neutrons (int): how many neutrons to use in the testing
    debug (bool): if to save the neutrons that exit the instrument
    """
    if debug:
        assert num_neutrons < 1001
    classname = className.lower()

    # Run the mcvine instrument first
    instr = os.path.join(thisdir, f"{classname}_instrument.py")
    mcvine_outdir = f"out.debug-mcvine_{classname}_cpu_instrument"
    if os.path.exists(mcvine_outdir):
        shutil.rmtree(mcvine_outdir)
    run_script.run1(
        instr, mcvine_outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        factory=f"mcvine.components.optics.{className}",
        save_neutrons_after=debug,
        overwrite_datafiles=True,
        is_acc=False)

    # Run our accelerated implementation
    outdir = f"out.debug-{classname}_gpu_instrument"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    run_script.run1(
        instr, outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        module=f"mcvine.acc.components.optics.{classname}",
        save_neutrons_after=debug,
        overwrite_datafiles=True,
        is_acc=True)

    # Compare output files
    tolerance = tolerances[floattype]
    global interactive
    for monitor in monitors:
        mcvine = hh.load(os.path.join(mcvine_outdir, monitor + ".h5"))
        mcvine_acc = hh.load(os.path.join(outdir, monitor + ".h5"))
        if interactive:
            from histogram import plot as plotHist
            plotHist(mcvine)
            plotHist(mcvine_acc)
            plotHist((mcvine_acc - mcvine) / mcvine)
        assert mcvine.shape() == mcvine_acc.shape()
        assert np.allclose(mcvine.data().storage(), mcvine_acc.data().storage(),
                           atol=tolerance)
