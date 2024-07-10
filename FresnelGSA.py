#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:11:54 2024

Fresnel Optics simulation of a Gaussian beam propagated through a phase plate
and to the fourier plane

Aiming to get the following parameters:
    Wavelength: 253 nm
    Beam size upstream of the phase plate: 8 mm
    Beam size in focus: 1.2 mm
    Target beam profile: cut Gaussian at 50%
    Focusing (Fourier) lens: f=1.2 m

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from padding import fastOn, fastOff
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib.colors import LogNorm
from GSA import GSA , stanford

# --- Globals ---
wavelength = 253 * 1e-9
w0 = 4 * 1e-3
f = 1.2
#k0 = 2* pi / wavelength
extent = [-8 * w0, 8 * w0]
z0 = pi/wavelength * w0**2

def Gaussian(sizeFactor = 11, wavelength = wavelength, w0 = w0, extent = extent, plot = False):
    
    # --- Extracting key paramters ---
    z0 = pi/wavelength * w0**2
    q0 = 1j * z0
    k0 = 2* pi / wavelength
    
    # --- Build a meshgrid to apply the gaussian too ---
    gridSize = 2 ** sizeFactor
    x_ = np.linspace(extent[0], extent[1], gridSize)
    y_ = np.linspace(extent[0], extent[1], gridSize)
    X, Y = np.meshgrid(x_, y_)
    rSquare = X**2 + Y**2
    
    # --- Creating the gaussian field using complex beam parameter ---    
    field = 1/q0 * np.exp(- 1j * k0 * rSquare / (2 * q0))

    # --- Plotting the field if required --- 
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
        #Imaginary part
        Imag = ax1.imshow(field.imag, extent = [extent[0], extent[1], extent[0], extent[1]])
        ax1.set_title("Imaginary Part")
        fig.colorbar(Imag, ax = ax1, orientation = 'horizontal')
        #Real part
        Real = ax2.imshow(field.real, extent = [extent[0], extent[1], extent[0], extent[1]])
        ax2.set_title("Real Part")
        fig.colorbar(Real, ax = ax2, orientation = 'horizontal')
        #Intensity
        Intensity = ax3.imshow(np.abs(field)**2, extent = [extent[0], extent[1], extent[0], extent[1]])
        ax3.set_title("Intensity")
        fig.colorbar(Intensity, ax = ax3, orientation = 'horizontal')
        #Extra decoration
        fig.suptitle("Input Gaussian Beam Parts", size = 20)
        ax1.set_xlabel('Beam size (m)')
        ax2.set_xlabel('Beam size (m)')
        ax3.set_xlabel('Beam size (m)')
        ax1.set_ylabel('Beam size (m)')
        fig.tight_layout()
        plt.show()
    
    return field


def Propagate(inputBeam, z, wavelength = wavelength, w0 = w0, padding = 1, extent = extent, plot = False):
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
        The wait of the curve using 1/e of the amplitude. The default is 4 mm
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
    kBeam = fftshift(fft2(paddedBeam))
    kShape = kBeam.shape[0]
    

    # --- Step 2 Apply the propagator --- 
    #Creating k-Space coordinates
    kx_, ky_ = fftfreq(kShape, d = 2**(1+padding) * extent[1]/kShape), fftfreq(kShape, d = 2**(1+padding)*extent[1]/kShape)
    
    kx_ = fftshift(kx_)
    ky_ = fftshift(ky_)
    kx, ky = np.meshgrid(kx_, ky_)
    

    #Propagator taken from K Khare (see ref)
    propagator =  np.exp(1j * z * np.sqrt(k0**2 - 4 * pi**2 * (kx**2 + ky**2)))
    
    #Apply the propagator in k space
    kPadded = kBeam * propagator
    
    # --- Step 3 : Return to real space ---
    #Return to Cartesian
    outputPadded = ifft2(ifftshift(kPadded))
    
    #Remove the padding
    outputBeam = fastOff(outputPadded, padding)
    
    return outputBeam 



def Lens(inputBeam, f, wavelength = wavelength, w0 = w0, extent = extent, plot = False):
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
        The wait of the curve using 1/e of the amplitude. The default is 4 mm
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
    
    #Built using reference I's radius of curvature implementation
    lensShift = np.exp(-1j * k0 * rSquare/(2 * f))
    
    # --- Applying the transfer function in real space ---
    outputBeam = inputBeam * lensShift
    
    # --- Possible Plotting ---
    if plot:
        plt.imshow(np.abs(outputBeam)**2)
    
    return outputBeam
    

def phasePlate(inputBeam, hologram = [30, None], wavelength = wavelength, w0 = w0, extent = extent, plot = False):
    if len(hologram) == 2:
        iterations, target = hologram
        hologram = GSA(inputBeam, iteration = iterations, target = target)
        
    inputPhase = np.angle(inputBeam)
    phasePlate = np.subtract(inputPhase, hologram)
        
    outputBeam = inputBeam * np.exp(-1j * phasePlate)  
    
    if plot:
        fig, (axA, axB) = plt.subplots(2, 2, figsize=(12, 10))
        outphase = np.angle(outputBeam)

        
        subtract = axA[0].imshow(np.subtract(outphase, hologram), cmap = 'Reds', extent = [extent[0], extent[1], extent[0], extent[1]])
        Real = axA[1].imshow(outputBeam.real, extent = [extent[0], extent[1], extent[0], extent[1]])
        Imag = axB[0].imshow(outputBeam.imag, extent = [extent[0], extent[1], extent[0], extent[1]])
        Intensity = axB[1].imshow(np.abs(outputBeam)**2, extent = [extent[0], extent[1], extent[0], extent[1]])
        
        fig.colorbar(subtract, ax = axA[0], orientation = 'vertical')
        fig.colorbar(Real, ax = axA[1], orientation = 'vertical')
        fig.colorbar(Imag, ax = axB[0], orientation = 'vertical')
        fig.colorbar(Intensity, ax = axB[1], orientation = 'vertical')



        axA[0].set_title("Difference in Hologram & Output Phase")
        axA[1].set_title("Real Part")
        axB[0].set_title("Imaginary part")
        axB[1].set_title("Intensity")
        
        axA[0].set_xlabel('Beam size (m)')
        axA[1].set_xlabel('Beam size (m)')
        axB[0].set_xlabel('Beam size (m)')
        axB[1].set_xlabel('Beam size (m)')
        axA[0].set_ylabel('Beam size (m)')
        fig.suptitle("Difference in Hologram and output Phase of the Beam", size = 20)
        fig.tight_layout()
        
    return outputBeam
