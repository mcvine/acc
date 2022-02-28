#!/usr/bin/env python

import os
thisdir = os.path.abspath(os.path.dirname(__file__))

script = os.path.join(thisdir, 'acc_sgm_instrument.py')
workdir = 'out.acc_sgm'
ncount = int(1e5)

from mcvine.acc import run_script
run_script.compile(script)
run_script.run(script, 'out.test_run_script', ncount=1e8)
