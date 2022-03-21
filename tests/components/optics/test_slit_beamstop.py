#!/usr/bin/env python

from mcvine.acc import config
# config.floattype = "float32"

import os
import histogram.hdf as hh
import numpy as np
import pytest
import shutil
from collections import deque
from math import gcd, pow, prod, sqrt
from mcvine import run_script
from mcvine.acc import test
from mcvine.acc.config import get_numpy_floattype
from mcvine.acc.components.optics.beamstop import Beamstop
from mcvine.acc.components.optics.slit import Slit
from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr

thisdir = os.path.dirname(__file__)
interactive = False


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("className", ["Slit", "Beamstop"])
def test_compare_mcvine(className, num_neutrons=int(1e7), debug=False):
    """
    Tests the acc cpu implementation of a slit or beamstop against mcvine
    """
    if debug:
        assert num_neutrons < 1001
    classname = className.lower()

    # Run the mcvine instrument first
    instr = os.path.join(thisdir, f"{classname}_instrument.py")
    mcvine_outdir = f"out.debug-mcvine_{classname}_cpu_instrument"
    if os.path.exists(mcvine_outdir):
        shutil.rmtree(mcvine_outdir)
    run_script.run1(
        instr, mcvine_outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        factory=f"mcvine.components.optics.{className}",
        save_neutrons_after=debug,
        overwrite_datafiles=True)

    # Run our accelerated implementation
    outdir = f"out.debug-{classname}_gpu_instrument"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    run_script.run1(
        instr, outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        module=f"mcvine.acc.components.optics.{classname}",
        save_neutrons_after=debug,
        overwrite_datafiles=True)

    # Compare output files
    mcvine_Ixy = hh.load(os.path.join(mcvine_outdir, "Ixy.h5"))
    mcvine_Ixdivx = hh.load(os.path.join(mcvine_outdir, "Ixdivx.h5"))
    Ixy = hh.load(os.path.join(outdir, "Ixy.h5"))
    Ixdivx = hh.load(os.path.join(outdir, "Ixdivx.h5"))

    global interactive
    if interactive:
        from histogram import plot as plotHist
        plotHist(mcvine_Ixy)
        plotHist(mcvine_Ixdivx)
        plotHist(Ixy)
        plotHist(Ixdivx)
        plotHist((Ixy-mcvine_Ixy)/mcvine_Ixy)
    assert mcvine_Ixy.shape() == Ixy.shape()
    assert mcvine_Ixdivx.shape() == Ixdivx.shape()

    tolerance = 1e-7 if get_numpy_floattype() == np.float32 else 1e-25
    assert np.allclose(mcvine_Ixy.data().storage(), Ixy.data().storage(),
                       atol=tolerance)
    assert np.allclose(mcvine_Ixdivx.data().storage(), Ixdivx.data().storage(),
                       atol=tolerance)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("cut", np.arange(0.15, 1, 0.2))
# Test a centered rectangular slit and beamstop with a range of cutoffs.
def test_width_height_cuts(invert, cut):
    width = 7
    height = 5
    if invert:
        component = Beamstop("beamstop", width=width, height=height, cut=cut)
    else:
        component = Slit("slit", width=width, height=height, cut=cut)

    def passes(x, y, p):
        is_within = abs(x) <= width / 2 and abs(y) <= height / 2
        return is_within != invert and p >= cut

    help_test_slit_beamstop(component, passes)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("radius", np.arange(1.5, 5, 1))
# Test a range of circular slits and beamstops with default cutoff.
def test_radii(invert, radius):
    if invert:
        component = Beamstop("beamstop", radius=radius)
    else:
        component = Slit("slit", radius=radius)

    def passes(x, y, p):
        is_within = pow(x, 2) + pow(y, 2) <= pow(radius, 2)
        return is_within != invert

    help_test_slit_beamstop(component, passes)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("xmin", [-7.5, 2.5])
@pytest.mark.parametrize("xmax", [3.5, 9.5])
@pytest.mark.parametrize("ymin", [-7.5, 2.5])
@pytest.mark.parametrize("ymax", [3.5, 9.5])
# Test a range of uncentered rectangular slits and beamstops with a non-default
# cutoff.
def test_mins_maxs(invert, xmin, xmax, ymin, ymax):
    if invert:
        component = Beamstop(
            "beamstop", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cut=0.65)
    else:
        component = Slit(
            "slit", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cut=0.65)

    def passes(x, y, p):
        is_within = xmin <= x <= xmax and ymin <= y <= ymax
        return is_within != invert and p >= 0.65
    help_test_slit_beamstop(component, passes)


# Helper function for testing a slit or beamstop with a range of neutrons.
def help_test_slit_beamstop(component, passes):
    # Where the neutron is when it reaches the component.
    xs_arrived = np.arange(-9, 10)  # 19
    ys_arrived = np.arange(-9, 10)  # 19
    # How the neutron position is different before it reaches the component.
    dxs = np.arange(-3, 4)  # 7
    dys = np.arange(-3, 4)  # 7
    # Try various neutron weights.
    ws = deque(np.arange(0.2, 1.1, 0.2))
    # Pattern of weights should be irregular against geometry of test data.
    assert gcd(len(ws), prod(map(len, [xs_arrived, ys_arrived, dxs, dys]))) == 1
    # Fill a neutron buffer with a range of neutrons that approach the plane of
    # the component.
    neutrons_start = []
    neutrons_end_expected = []
    (any_pass, any_fail) = (False, False)
    for x_arrived in xs_arrived:
        for y_arrived in ys_arrived:
            for dx in dxs:
                for dy in dys:
                    # Construct an incoming neutron 2m before.
                    distance = sqrt(sum([2, pow(dx, 2), pow(dy, 2)]))
                    position_start = [x_arrived + dx, y_arrived + dy, -2]
                    velocity = [n / distance for n in [-dx, -dy, 2]]  # 1 m/s
                    w = ws[0]
                    ws.rotate()  # A different weight for the next neutron.
                    neutrons_start.append(neutron(
                        r=position_start, v=velocity, prob=w))
                    if passes(x_arrived, y_arrived, w):
                        # Construct a neutron expected from the component.
                        position_arrived = [x_arrived, y_arrived, 0]
                        neutrons_end_expected.append(neutron(
                            r=position_arrived, v=velocity, prob=w,
                            time=distance))
                        any_pass = True
                    else:
                        any_fail = True
    assert any_pass and any_fail
    neutrons_end_expected = neutron_buffer_from_array(neutrons_end_expected)

    # Propagate the neutrons.
    neutrons_buffer = neutron_buffer_from_array(neutrons_start)
    component.process(neutrons_buffer)

    # Compare the actual neutrons with the expectation.
    tolerance = 1e-6 if get_numpy_floattype() == np.float32 else 1e-15
    assert np.allclose(neutrons_as_npyarr(neutrons_buffer),
                       neutrons_as_npyarr(neutrons_end_expected),
                       atol=tolerance)


def neutron_buffer_from_array(neutrons):
    buffer = neutron_buffer(len(neutrons))
    index = 0
    for neutron in neutrons:
        buffer[index] = neutron
        index += 1
    assert index
    return buffer


def debug():
    global interactive
    interactive = True
    test_compare_mcvine(debug=True, num_neutrons=100)
    return


def main():
    global interactive
    interactive = True
    className = "Beamstop"
    test_compare_mcvine(num_neutrons=int(1e6), className=className)
    return


if __name__ == '__main__':
    main()
    # debug()
