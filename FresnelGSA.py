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
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# --- Globals ---
wavelength = 253 * 1e-9
w0 = 4 * 1e-3
f = 1.2
#k0 = 2* pi / wavelength
extent = [-5* w0, 5* w0]

def Gaussian(sizeFactor = 11, wavelength = wavelength, w0 = w0, extent = extent, plot = False):
    
    # --- Extracting key paramters ---
    z0 = pi/wavelength * w0**2
    q0 = 1j * z0
    k0 = 2* pi / wavelength
    
    # --- Build a meshgrid to apply teh gaussian too ---
    gridSize = 2 ** sizeFactor
    x_ = np.linspace(extent[0], extent[1], gridSize)
    y_ = np.linspace(extent[0], extent[1], gridSize)
    X, Y = np.meshgrid(x_, y_)
    rSquare = X**2 + Y**2
    
    # --- Creating the gaussian field using complex beam parameter ---    
    field = 1/q0 * np.exp(- 1j * k0 * rSquare / (2* q0))

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
        ax3.set_title("Intenisty")
        fig.colorbar(Intensity, ax = ax3, orientation = 'horizontal')
        #Extra decoration
        fig.suptitle("Input Gaussian Beam Parts", size = 20)
        fig.tight_layout()
        ax1.set_xlabel('Beam size (m)')
        ax2.set_xlabel('Beam size (m)')
        ax3.set_xlabel('Beam size (m)')
        ax1.set_ylabel('Beam size (m)')
    
    return field
