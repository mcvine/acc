#!/usr/bin/env python

import os
import pytest

from mcvine.acc import test

thisdir = os.path.dirname(__file__)
interactive = False


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False):
    """
    Tests the acc cpu implementation of an arm against mcvine
    """
    import test_helper
    test_helper.compare_mcvine("Arm",
                               ["Ixy", "Ixdivx", "Ixdivy"],
                               {"float32": 1e-10, "float64": 1e-25},
                               num_neutrons, debug)


def main():
    global interactive
    interactive = True
    test_compare_mcvine(num_neutrons=int(1e6))
    return


if __name__ == '__main__':
    main()
