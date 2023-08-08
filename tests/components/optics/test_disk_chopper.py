#!/usr/bin/env python

import os
import pytest
from mcvine.acc import test

thisdir = os.path.dirname(__file__)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a disk chopper against mcvine
    """
    instr = os.path.join(thisdir, "diskchopper_instrument.py")
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "DiskChopper",
        #["IL"],
        ["psd"],
        {"float32": 1e-7, "float64": 1e-8},
        num_neutrons, debug,
        instr = instr,
        interactive=interactive,
        acc_component_spec = dict(is_acc=True),
        nonacc_component_spec = dict(is_acc=False)
    )

def debug():
    test_compare_mcvine(debug=True, num_neutrons=100, interactive=True)
    return


def main():
    test_compare_mcvine(num_neutrons=int(5e6), interactive=True)
    return


if __name__ == '__main__':
    main()
    # debug()
