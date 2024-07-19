#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:08:08 2024

GSA software to be implemented into the Fresnel GSA simulation

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


# --- Globals ---
wavelength = 253 * 1e-9
w0 = 4 * 1e-3
f = 1.2
#k0 = 2* pi / wavelength
extent = [-8 * w0, 8 * w0]
z0 = pi/wavelength * w0**2



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


def GSA(inputField, iteration = 30, target = None,
        size = 0, plot = False, scalarProduct = False, save = '',
        rounding = None, boxing = None):
    """
    

    Parameters
    ----------
    inputField : TYPE
        DESCRIPTION.
    iteration : TYPE, optional
        DESCRIPTION. The default is 30.
    target : TYPE, optional
        DESCRIPTION. The default is None.
    size : TYPE, optional
        DESCRIPTION. The default is 0.
    plot : Bool, optional
        DESCRIPTION. The default is False.
    scalarProduct : Bool, optional
        DESCRIPTION. The default is False.
    save : string, optional
        DESCRIPTION. The default is '' where the file will not be saved.

    Returns
    -------
    phase : TYPE
        DESCRIPTION.

    """
    # --- Intialising parameters --- 
    inputAmplitude = np.abs(inputField)
    phase = np.angle(np.array(inputField))
    #phase = np.random.rand(inputField.shape[0], inputField.shape[1])
    field = np.zeros_like(inputField)
    outputFP = []
    # --- if no target is assigned the default is a truncated super gaussian ---
    if target is None:
        target = superTruncGaussian(inputField, trunc = 60)
    
    for i in tqdm(range(iteration), desc = 'Iterations of IFTA'):
        
        # --- Input the initial Intensity and keep the varying phase --- 
        
        field.real = inputAmplitude * np.cos(phase)
        field.imag = inputAmplitude * np.sin(phase)
        
        # --- Transform to Frequency Domain --- 
        paddedField = fastOn(field, size)
        field = fftshift(fft2(paddedField))
        
        outputIter = np.power(np.abs(field), 2)
        
        outputFP += [outputIter] #output fourier plane intensity at each iteration
        outputIter = None
        
        
        # --- Replace the intensity with the target intensity --- 
        
        phase = np.angle(field) #Find the new phase after FFT
        
        field.real = target * np.cos(phase)
        field.imag = target * np.sin(phase)
        
        # --- Transform back to Carthesian --- 
        
        field = ifft2(ifftshift(field))
        field = fastOff(field, size)
        
        # --- Collect the new phase --- 
        phase = np.angle(field)
        
        # --- Adjusting the phase to error --- 
        if rounding != None:
            phase = np.round(phase, rounding)
        if boxing !=None: # Work on a function for this one
            phase = Box(phase, boxing)
       
    # --- If plot is set to True the outputs will be neatly plotted ---
    if plot:
        # --- Plotting the Fourier Plane Intensity --- 
        plt.imshow(outputFP[-1], cmap = 'viridis', extent = [extent[0], extent[1], extent[0], extent[1]])#, norm = LogNorm())
        plt.title('Fourier Plane Output Intensity')
        plt.xlabel('distance (m)')
        plt.ylabel('distance (m)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
        # --- Plotting the hologram -> Phase Distribution --- 
        plt.imshow(phase, cmap = 'inferno', extent = [extent[0], extent[1], extent[0], extent[1]])
        plt.title('Post-Phase-Plate Beam Phase')
        plt.xlabel('distance (m)')
        plt.ylabel('distance (m)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    # --- If set the scalar product will be computed and plotted if plot is asked --- 
    if scalarProduct:
        # --- Plotting the quality of the output throughout the process --- 
        sProduct = scalarPlotting(outputFP, target, plotting=plot)
       
    if save:
        with h5py.File(save, 'w') as file:
            file.create_dataset('Phase', data=phase)
            file.create_dataset('ScalarProduct', data=sProduct)
            file.create_dataset('Target', data=target)
    
    return phase











































