#!/usr/bin/env python

import os
import histogram.hdf as hh
import numpy as np


def check_histogram_match(histfile_a, histfile_b, tolerance=1e-8, interactive=False):
    """
    Test helper to load two histogram hdf files and compare them within a tolerance

    Parameters
    ----------
    histfile_a : str
        path of first file to compare
    histfile_b : str
        path of second file to compare
    tolerance : float
        tolerance for float comparisons
    interactive: bool
        Whether to plot each histogram and the relative difference

    Returns
    -------
    bool
        True if data of both histogram files are within tolerance, false otherwise
    """

    assert os.path.exists(histfile_a)
    assert os.path.exists(histfile_b)

    hist_a = hh.load(histfile_a)
    hist_b = hh.load(histfile_b)

    if interactive:
        from histogram import plot as plotHist
        plotHist(hist_a)
        plotHist(hist_b)
        plotHist((hist_a - hist_b) / hist_a)

    assert hist_a.shape() == hist_b.shape()
    return np.allclose(hist_a.data().storage(), hist_b.data().storage(),
                       atol=tolerance)
