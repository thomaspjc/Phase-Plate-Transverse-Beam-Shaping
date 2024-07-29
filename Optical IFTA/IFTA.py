#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:48:11 2024

IFTA Algorithm using propagation

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
from ToolsIFTA import normalize, hologram, scalarProduct, scalarPlotting


# --- Globals ---
# --- Upgrade the Scipy FFT speed --- 
ncpu = -1



def InitializePropagator(inputBeam, z, wavelength = 253e-9, padding = 1,
                         extent = [-1.27 * 1e-2, 1.27 * 1e-2]):
    """
    Applying the propagator transfer function to an input Complex Beam 

    Parameters
    ----------
    inputBeam : np.array
        The complex beam to apply the transfer function to
    z : float
        The distance to propagate by
    wavelength : float, optional
        Single wavelength of the beam. The default is 253 nm
    w0 : float, optional
        The waist of the curve using 1/e of the amplitude. The default is 4 mm
    padding : integer, optional
        factor of 2 to pad the array with -> See padding.py. The default is 1.
    extent : array, optional
        Extent of the array to build. The default is set in the globals.
    plot : Bool, optional
        Boolean to choose if plots should be made. The default is False.

    References
    ---------
    I) Fourier Optics and Computational Imaging, Kedar Khare, Chap 11

    Returns
    -------
    outputBeam : np.array
        Output Beam in real space with no padding after applying the transfer function

    """
    
    # --- Extracting Parameters ---
    k0 = 2 * pi / wavelength
    
    # --- Step 1 : Transforming the input beam to k-space ---
    #Apply the padding to ensure high quality FFT
    paddedBeam = fastOn(inputBeam, padding)
    kBeam = fftshift(fft2(paddedBeam, workers = ncpu))
    kShape = kBeam.shape[0]
    

    # --- Step 2 Apply the propagator --- 
    #Creating k-Space coordinates
    kx_, ky_ = fftfreq(kShape, d = 2**(1+padding) * extent[1]/kShape), fftfreq(kShape, d = 2**(1+padding)*extent[1]/kShape)
    
    kx_ = fftshift(kx_)
    ky_ = fftshift(ky_)
    kx, ky = np.meshgrid(kx_, ky_)
    kSquare = kx**2 + ky**2
    

    #Propagator taken from K Khare (see ref)
    propagator = np.exp(1j * z * np.sqrt(k0**2 - 4 * pi**2 * (kx**2 + ky**2)))
    
    return propagator

def InitializeLens(inputBeam, f, wavelength = 253e-9,
                   extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
    """
    Applying a lens transformation to an incoming Complex Beam

    Parameters
    ----------
    inputBeam : np.array
        The complex beam to apply the transfer function to
    f : float
        The focal length of the lens to use (assumed symmetrical in x and y)
    wavelength : float, optional
        Single wavelength of the beam. The default is 253 nm
    w0 : float, optional
        The waist of the curve using 1/e of the amplitude. The default is 4 mm
    extent : array, optional
        Extent of the array to build. The default is set in the globals.
    plot : Bool, optional
        Boolean to choose if plots should be made. The default is False.
        
    References
    ----------
    I) 'Soft x-ray self-seeding simulation methods and their application for 
        the Linac Coherent Light Source', S. Serkez et al.

    Returns
    -------
    outputBeam : np.array
        Outgoing Complex Beam after Lens  transformation

    """
    
    # --- Extracting Parameters --- 
    k0 = 2 * pi / wavelength
    inputShape = inputBeam.shape
    
    # --- Building the transfer function ---
    x_, y_ = np.linspace(extent[0], extent[1], inputShape[0]), np.linspace(extent[0], extent[1], inputShape[1])
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    rSquare
    #Built the lens transfer function using Serkez's radius of curvature implementation
    lensShift = np.exp(-1j * k0 * rSquare/(2 * f))
    
    return lensShift


def IFTA(inputField, iteration = 30, f = 1.2, z = 1.2, target = None,
        size = 0, plot = False, scalarProduct = False, save = '',
        rounding = None, boxing = None, wavelength = 253e-9,
        extent = [-1.27 * 1e-2, 1.27 * 1e-2]):
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
    phase = np.random.rand(inputField.shape[0], inputField.shape[1]) * pi
    #phase = np.tile([[pi,-pi],[-pi,pi]], (int(inputField.shape[0]/2), int(inputField.shape[1]/2)))
    field = np.zeros_like(inputField)
    outputFP = []
    k0 = 2 * pi / wavelength
    
    # --- if no target is assigned the default is a truncated super gaussian ---
    if target is None:
        target = superTruncGaussian(inputField, trunc = 60)
        
    # --- Initialising the propagation matrices ---
    #This saves on computation time for each iteration
    
    #Initialising the propagator k space
    propagatorForward = InitializePropagator(inputField, z, padding = size)
    propagatorBackward = InitializePropagator(inputField, -z, padding = size)
    #Initialising the lens meshgrid
    lensShiftForward = InitializeLens(inputField, f)
    lensShiftBackward = InitializeLens(inputField, -f)
    
    
    for i in tqdm(range(iteration), desc = 'Iterations of IFTA'):
        
        # --- Input the initial Intensity and keep the varying phase --- 
        field = inputAmplitude * np.exp(1j * phase)
        
        # --- Propagate to Fourier Plane --- 
        #Apply the forward lens shift to the beam
        field *= lensShiftForward
        
        #Propagate the beam to the lens' Focal Plane 
        paddedField = fastOn(field, size)
        kPadded = fftshift(fft2(paddedField, workers = ncpu))
        kPadded *= propagatorForward
        outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
        field = fastOff(outputPadded, size)
        
        # --- Extracting Analytics --- 
        outputIter = np.power(np.abs(field), 2)
        outputFP += [outputIter] #output fourier plane intensity at each iteration
        
        
        # --- Replace the intensity with the target intensity --- 
        
        phase = np.angle(field) #Find the new phase after FFT
        field = target * np.exp(1j * phase)
        
        # --- Transform back to Carthesian --- 
        
        #Undoing the propagation
        paddedField = fastOn(field, size)
        kPadded = fftshift(fft2(paddedField, workers = ncpu))
        kPadded *= propagatorBackward
        outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
        field = fastOff(outputPadded, size)
        #Undoing the lens transformation
        field *= lensShiftBackward
        '''This method propagates forward
        #Undoing the propagation
        paddedField = fastOn(field, size)
        kPadded = fftshift(fft2(paddedField, workers = ncpu))
        kPadded *= propagatorForward
        outputPadded = ifft2(ifftshift(kPadded), workers = ncpu)
        field = fastOff(outputPadded, size)
        #Undoing the lens transformation
        field *= lensShiftForward'''
        
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
        plt.imshow(phase, cmap = 'vlag', extent = [extent[0], extent[1], extent[0], extent[1]])
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



if __name__ == "__main__":
    extent = [-1.27 * 1e-2, 1.27 * 1e-2]
    wavelength = 253 * 1e-9
    w0 = 4 * 1e-3
    f = 1.2

    # --- Testing the IFTA ---
    z0 = pi/wavelength * w0**2
    q0 = 1j * z0
    k0 = 2* pi / wavelength
    # --- Build a meshgrid to apply the gaussian too ---
    gridSize = 2 ** 10
    x_ = np.linspace(extent[0], extent[1], gridSize)
    y_ = np.linspace(extent[0], extent[1], gridSize)
    X, Y = np.meshgrid(x_, y_)
    rSquare = X**2 + Y**2
    
    # --- Creating the gaussian field using complex beam parameter ---    
    field = 1/q0 * np.exp(- 1j * k0 * rSquare / (2 * q0))
    plt.imshow(np.abs(field)**2)
    plt.show()
    
    target = flatTop(field, extent = extent, w0 = 6e-4, plot = True)
    phase = IFTA(field, plot = True, scalarProduct=True, iteration = 30,
                 target = target, size = 1)

