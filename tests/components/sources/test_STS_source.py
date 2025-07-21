#!/usr/bin/env python

import os, pytest
import numpy as np
thisdir = os.path.dirname(__file__)
dat = os.path.join(thisdir, 'BL20-CY-123D-STS-Min-2G-source_mctal-195_sp.dat')
from mcvine.acc import test

from mcni import neutron_buffer, neutron
from mcvine.acc.components.sources.STS_source import STS_source

@pytest.mark.skip(reason="needs mcstas2_components.monitors.E_monitorboostpython")
def test_load_sts_source():
    from mcvine.acc.components.sources._SNS_source_utils import sts_source_load

    x, y, xylen, tvec, mat, params = sts_source_load(dat)
    assert (len(x) == xylen + 2)
    assert (len(y) == xylen + 2)
    assert (mat.shape[0] == xylen + 2)
    assert (mat.shape[1] == len(tvec))
    assert (len(params) == 7)

    expected_params = np.array([700.0, 15.0, 0.04105, 5.5, 0.03, 0.03, 0.0])
    np.testing.assert_array_equal(params, expected_params)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component():
    src = STS_source('src', dat, Emin=3, Emax=82, xwidth=0.03, yheight=0.03, dist=2.5, focus_xw = .03, focus_yh = .03)
    neutrons = src.process(neutron_buffer(10))
    for n in neutrons:
        print(n)
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_long(ncount = 1e6):
    src = STS_source('src', dat, Emin=3, Emax=82, xwidth=0.03, yheight=0.03, dist=2.5, focus_xw = .03, focus_yh = .03)
    neutrons = src.process(neutron_buffer(int(ncount)))
    return


def test_mcstas_component_long(ncount=1e6):
    import mcvine.components as mc
    src = mc.sources.STS_Source(
        'src', filename=dat,
        Emin=3, Emax=82,
        xwidth = 0.03, yheight=0.03,
        dist = 2.5, focus_xw=0.03, focus_yh=0.03
    )
    neutrons = src.process(neutron_buffer(int(ncount)))
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_acc_nonacc(num_neutrons=int(1e6), debug=False, interactive=False):
    """
    Compare acc and non-acc component
    """
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    rtol = dict(
        threshold = 1./np.sqrt(num_neutrons/100) * 10,
        outlier_fraction = 0.05,
    )
    compare_acc_nonacc(
        "STS_source",
        ["Ixy", "IE"],
        tolerances = dict(float32 = None, float64 = None), # use relerr_tolerances instead
        num_neutrons=num_neutrons,
        debug=debug,
        interactive=interactive, workdir = thisdir,
        relerr_tolerances=dict(float32 = rtol, float64 = rtol),
    )


def main():
    # test_component()
    # test_component_long(1e7)
    # test_mcstas_component_long(1e7)
    test_compare_acc_nonacc(num_neutrons=int(1e6), debug=False, interactive=True)
    return

if __name__ == '__main__': main()
