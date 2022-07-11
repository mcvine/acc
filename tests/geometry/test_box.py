#!/usr/bin/env python

import pytest
import numpy as np

from mcvine.acc import test
from mcvine.acc.geometry import locate, location


# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_locate_wrt_box():
    assert locate.cu_device_locate_wrt_box(0, 0, 0, 0.02, 0.03, 0.04) == location.inside
    assert locate.cu_device_locate_wrt_box(0.01, 0, 0, 0.02, 0.03, 0.04) == location.onborder
    assert locate.cu_device_locate_wrt_box(0.0101, 0, 0, 0.02, 0.03, 0.04) == location.outside

    return
