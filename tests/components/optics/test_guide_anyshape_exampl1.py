#!/usr/bin/env python

import os
import pytest
from mcvine.acc import test

thisdir = os.path.dirname(__file__)

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "guide_anyshape_example1", # {}_instrument.py will be the instrument script
        ["Ixy", "Ixdivx", "Ixdivy"],
        {"float32": 1e-6, "float64": 1e-7},
        num_neutrons, debug,
        interactive=interactive, workdir = thisdir,
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine2(num_neutrons=int(1e7), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    geometry = os.path.join(thisdir, './data/guide_anyshape_straight_3.5cmX3.5cmX10mX4cmX4cm.off')
    nonacc_component_kargs = dict(
        w1=0.035, h1=0.035, w2=0.04, h2=0.04, l=10,
    )
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "guide_anyshape_example1", # {}_instrument.py will be the instrument script
        ["Ixy", "Ixdivx", "Ixdivy"],
        {"float32": 1e-6, "float64": 1e-7},
        num_neutrons, debug,
        interactive=interactive, workdir = thisdir,
        acc_component_spec=dict(geometry=geometry),
        nonacc_component_spec=dict(nonacc_component_kargs=nonacc_component_kargs),
    )

def debug():
    test_compare_mcvine(debug=True, num_neutrons=100, interactive=True)
    return


def main():
    test_compare_mcvine(num_neutrons=int(1e6), interactive=True)
    return


if __name__ == '__main__':
    main()
    # debug()
