#!/usr/bin/env python

import os
import pytest

from mcvine.acc.components.optics import offio
from mcvine.acc import test

thisdir = os.path.dirname(__file__)

def test_load():
    path = os.path.join(thisdir, './data/chess-guide-example/ST17_10_1.off')
    vertices, faces = offio.load(path)
    assert faces.shape == (4,4)
