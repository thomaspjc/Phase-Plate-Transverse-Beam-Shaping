#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:42:30 2024

Function taken from Scikit-UED module to facilitate implementation

@author: Laurent P. René de Cotret. 
"""

import numpy as np
import scipy.fft as fft

def fft2freq(x, y, indexing="xy"):
    """
    Return the Discrete Fourier Transform sample frequencies for a 2D array defined on ``x`` and ``y``.
    Generalization of ``fftfreq``.

    Parameters
    ----------
    x, y : `~numpy.ndarray`, ndim 2
        Meshgrid-style arrays. Spacing must be uniform.
    indexing : {'ij', 'xy'}, optional
        Indexing used to generate ``x`` and ``y``.

    Returns
    -------
    kx, ky : `~numpy.ndarray`, ndim 2

    Raises
    ------
    ValueError : if ``indexing`` is invalid.
    """
    if indexing == "xy":
        extent_x, extent_y = x[0, :], y[:, 0]
    elif indexing == "ij":
        extent_x, extent_y = x[:, 0], y[0, :]
    else:
        raise ValueError(
            "Indexing should be either 'xy' or 'ij', not {}".format(indexing)
        )

    # Spacing assuming constant x and y spacing
    spacing_x = abs(extent_x[1] - extent_x[0])
    spacing_y = abs(extent_y[1] - extent_y[0])

    freqs_x = fft.fftfreq(len(extent_x), d=spacing_x)
    freqs_y = fft.fftfreq(len(extent_y), d=spacing_y)

    return np.meshgrid(freqs_x, freqs_y, indexing=indexing)