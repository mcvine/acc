#!/usr/bin/env python

import os
import pytest

from mcvine.acc.components.optics import offio
from mcvine.acc import test

thisdir = os.path.dirname(__file__)

def test_load():
    path = './data/chess-guide-example/ST17_10_1.off'
    assert offio.load(path).shape == (4,4,3)
