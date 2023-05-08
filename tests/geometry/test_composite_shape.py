#!/usr/bin/env python

import pytest, os, time
import math, numpy as np
from numba import cuda

from instrument.nixml import parse_file
from mcvine.acc import test
from mcvine.acc.geometry import locate, location, arrow_intersect
from mcvine.acc.geometry.composite import makeModule

thisdir = os.path.dirname(__file__)

def test_makeModule():
    makeModule(4, overwrite=True)
    return

