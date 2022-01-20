#!/usr/bin/env python

import os, shutil, time
from mcvine import run_script

thisdir = os.path.dirname(__file__)
interactive = False


def main():
    num_neutrons = int(1e8)
    # Run the mcvine instrument first
    instr = os.path.join(thisdir, "guide_instrument.py")
    mcvine_outdir = 'out.save_neutrons_before_guide'
    if os.path.exists(mcvine_outdir):
        shutil.rmtree(mcvine_outdir)
    run_script.run1(
        instr, mcvine_outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        guide_factory = "mcvine.components.optics.Guide",
        save_neutrons_before_guide=True,
        save_neutrons_after_guide=False,
        overwrite_datafiles=True)
    return

if __name__ == '__main__': main()
