#!/usr/bin/env python

import numpy as np

def histogram_is_close(
        h1, h2, min_nonzero_fraction=0.9,
        rtol=0.001, min_rdiff_fraction=0.9,
        atol=1e-10, min_adiff_fraction=0.9,
):
    """check if histogram h1 is close to the reference h2"""
    if h2.shape != h1.shape:
        return False
    rdiff = h1/h2-1
    isfinite = np.isfinite(rdiff)
    rdiff = rdiff[isfinite]
    adiff = (h1-h2)[~isfinite]
    # make sure there is enough data points that are not zero
    print(rdiff.size, h2.size)
    if rdiff.size<min_nonzero_fraction*h2.size: return False
    # make sure for a good portion of nonzero data points,
    # the relative diff is smaller than the given tolerance
    print ((np.abs(rdiff)<rtol).sum(), rdiff.size)
    if (np.abs(rdiff)<rtol).sum() < rdiff.size*min_rdiff_fraction:
        return False
    print ((np.abs(adiff)<atol).sum(), adiff.size)
    if (np.abs(adiff)<atol).sum() < adiff.size*min_adiff_fraction:
        return False
    return True
