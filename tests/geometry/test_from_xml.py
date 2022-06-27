#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, numpy as np, time
from numba import cuda

def test_union_example1():
    from instrument.nixml import parse_file
    parsed = parse_file('union_example1.xml')
    union1 = parsed[0]
    print(union1)
    return

def main():
    test_union_example1()
    return

if __name__ == '__main__': main()
