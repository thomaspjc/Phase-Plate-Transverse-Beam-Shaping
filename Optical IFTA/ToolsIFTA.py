#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:24:36 2024

Tool functions for the IFTA algorithms

@author: thomas
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from padding import fastOn, fastOff
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib.colors import LogNorm
import cv2
from tqdm import tqdm
from Targets import stanford, flatTop, superTruncGaussian
import pandas as pd
from PhysicalPlate import Box
import h5py

def normalize(array):
    """
    Normalizing a given array 

    Parameters
    ----------
    array : meshgrid
        meshgrid that should be normalized

    Returns
    -------
    normalized_array : meshgrid
        normalized meshgrid 

    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array



def hologram(phase):
    """
    

    Parameters
    ----------
    phase : array
        phase array from the ITFA

    Returns
    -------
    holo : array
        hologram 

    """
    phase = np.where(phase<0, phase+2*np.pi, phase)
    p_max = np.max(phase)
    p_min = np.min(phase)
    holo = pi * ((phase - p_min)/(p_max- p_min))
    return holo

def scalarProduct(field1, field2):
    """
    Compute the scalar product of two electric fields.

    Parameters:
    field1 (ndarray): First electric field (complex-valued).
    field2 (ndarray): Second electric field (complex-valued).

    Returns:
    float: The scalar product of the two fields.
    """
    # --- Flatten the fields to 1D arrays ---
    field1_flat = field1.flatten()
    field2_flat = field2.flatten()
    
    # --- Compute the scalar product --- 
    scalar_product_value = np.sum(field1_flat * np.conjugate(field2_flat))
    
    # --- Normalize by the product of norms to get the overlap --- 
    norm1 = np.sqrt(np.sum(np.abs(field1_flat)**2))
    norm2 = np.sqrt(np.sum(np.abs(field2_flat)**2))
    
    if norm1 != 0 and norm2 != 0:
        normalized_scalar_product = scalar_product_value / (norm1 * norm2)
    else:
        normalized_scalar_product = 0
    
    return np.abs(normalized_scalar_product)

def scalarPlotting (outputs, target, plotting = True):
    """
    

    Parameters
    ----------
    target : TYPE
        DESCRIPTION.
    outputs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    References
    -------
    Direct fabrication of arbitrary phase masks in optical glass via 
    ultra-short pulsed laser writing of refractive index modifications

    """
    quality = []
    for field in tqdm(range(len(outputs)), desc = 'Inner Product IFTA Verification'):
        quality += [scalarProduct(target, normalize(outputs[field]))]
    if plotting:
        plt.plot(np.arange(1, len(quality)+1),quality, '.-')
        plt.title('Quality check from scalar product')
        plt.xlabel("Iterations")
        plt.ylabel("Normalized Scalar Product")
        plt.xscale('log')
        maxQuality = quality[-1]*100
        textstr = f'Final Quality = {maxQuality:.4g}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.5, 0.1, textstr,  fontsize=14,transform=plt.gca().transAxes,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()
    return quality