#!/usr/bin/env python

import os
import pytest

from mcvine.acc import test

thisdir = os.path.dirname(__file__)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of an arm against mcvine
    """
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "Arm",
        ["Ixy", "Ixdivx", "Ixdivy"],
        {"float32": 1e-10, "float64": 1e-25},
        num_neutrons, debug,
        interactive=interactive, workdir = thisdir,
    )


def main():
    test_compare_mcvine(num_neutrons=int(1e6), interactive=True)
    return


if __name__ == '__main__':
    main()
