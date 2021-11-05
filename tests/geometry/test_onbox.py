#!/usr/bin/env python

import pytest, os
if os.environ.get('USE_CUDA').lower() == 'false':
    pytest.skip("No CUDA", allow_module_level=True)

from mcvine.acc.geometry import onbox

def test():
    # onbox.test_cu_device_update_intersections()
    # onbox.test_cu_device_intersect_box()
    onbox.test_cu_intersect_box()
    return

def main():
    test()

if __name__ == '__main__': main()
