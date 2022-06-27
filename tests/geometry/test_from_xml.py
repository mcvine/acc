#!/usr/bin/env python

import pytest, os

import os, numpy as np, time
from numba import cuda

from mcvine.acc import test
from mcvine.acc.geometry import locate, location

thisdir = os.path.dirname(__file__)

def test_union_example1():
    from instrument.nixml import parse_file
    parsed = parse_file(os.path.join(thisdir, 'union_example1.xml'))
    union1 = parsed[0]
    f = locate.LocateFuncFactory()
    cudakernel = f.render(union1)
    assert cudakernel(0,0,0) == location.inside
    assert cudakernel(0,0,0.03) == location.inside
    assert cudakernel(0,0,0.049999) == location.inside
    assert cudakernel(0,0,0.05) == location.onborder
    assert cudakernel(0,0,0.050001) == location.outside
    assert cudakernel(0,0.0249999, 0) == location.inside
    assert cudakernel(0,0.025, 0) == location.onborder
    assert cudakernel(0,0.0250001, 0) == location.outside
    return

def main():
    test_union_example1()
    return

if __name__ == '__main__': main()
