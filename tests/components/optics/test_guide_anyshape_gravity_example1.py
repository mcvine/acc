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
        acc_component_spec=dict(
            acc_component_factory = 'mcvine.acc.components.optics.guide_anyshape_gravity.Guide_anyshape_gravity',
        ),
        nonacc_component_spec=dict(
            nonacc_component_factory = 'mcvine.components.optics.Guide_gravity',
            nonacc_component_kargs = dict(
                G = -9.80665,
                w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=10,
            )
        )
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
