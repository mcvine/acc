#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)
dat = os.path.join(thisdir, 'source_rot2_cdr_cyl_3x3_20190417.dat')

from mcni import neutron_buffer, neutron
from mcvine.acc.components.sources.SNS_source_cpu import SNS_source, generate
from mcvine.acc.components.sources._SNS_source_utils import init

def test_generate():
    INorm2, Es, Pvec, ts, Ptmat, EPmin, EPmax, Eidx_range, tidx_range = init(3, 1500., dat)
    Eidx_start, Eidx_stop = Eidx_range
    tidx_start, tidx_stop = tidx_range
    E, t = generate(10, EPmin, EPmax, Es, Eidx_start, Eidx_stop, Pvec, ts, tidx_start, tidx_stop, Ptmat)
    print(E,t*1e6)
    return

def test_component():
    src = SNS_source(
        'src', dat, 5, 20, 0.03, 0.03,
        5, .03, .03,
    )
    neutrons = src.process(neutron_buffer(10))
    for n in neutrons:
        print(n)
    return

def test_component_n1e6():
    src = SNS_source(
        'src', dat, 5, 20, 0.03, 0.03,
        5, .03, .03,
    )
    neutrons = src.process(neutron_buffer(int(1e6)))
    return

def main():
    test_generate()
    # test_component()
    test_component_n1e6()
    return

if __name__ == '__main__': main()
