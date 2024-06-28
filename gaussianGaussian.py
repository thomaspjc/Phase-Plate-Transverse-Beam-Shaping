#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:45:15 2024

This file works within the Gaussian Optics approximation to propagate a Gaussian beam

@author: thomas
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as pi


# --- Globals --- 
gridSize = 1000
phase = np.random.rand(gridSize, gridSize)
wavelength = 253*1e-9
w0 = 8/2*1e-3
z0 = pi/wavelength * w0**2
extent = [-8*w0, 8*w0, -8*w0, 8*w0]

def gaussian2(wavelength = 253*1e-9, w0 = 8/2*1e-3, A0 = 1):
    """
    Building an input Gaussian using Gaussian Optics as a base 
    Here z = 0

    Parameters
    ----------
    wavelength : TYPE, optional
        DESCRIPTION. The default is 253*1e-9.
    w0 : TYPE, optional
        DESCRIPTION. The default is 8/2*1e-3.
    A0 : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    gaussian : TYPE
        DESCRIPTION.
    """
    
    z0 = pi/wavelength * w0**2
    
    x = np.linspace(-8*w0, 8*w0, gridSize)
    y = np.linspace(-8*w0, 8*w0, gridSize)
    X, Y = np.meshgrid(x, y)
    
    rSquare = X**2 + Y**2
    
    field = A0 * np.exp(-rSquare/w0**2)
    
    gaussian = np.empty_like(field, dtype="complex")
    gaussian.real = field
    gaussian.imag = 0
    return gaussian
    

def gaussianPropagate(inputBeam, matrix, wavelength = 253*1e-9, w0 = 8/2*1e-3, phase = 0):
    """
    Using Guassian optics to propagate the beam through space

    Parameters
    ----------
    inputBeam : np.meshgrid
        the gaussian beam to be propagated
    matrix : 4x1 array
        [ABCD] matrix that describes the propagation in Gaussian Optics
    wavelength : TYPE, optional
        DESCRIPTION. The default is 253*1e-9.
    w0 : TYPE, optional
        DESCRIPTION. The default is 8/2*1e-3.
    phase : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    field : TYPE
        DESCRIPTION.
    """
    
    # --- Transfer matrix Parameters --- 
    A, B, C, D = matrix
    
    # --- Initial Parameters ---
    z0 = pi/wavelength * w0**2
    A0 = np.sqrt(np.abs(inputBeam)**2)

    # --- Building a meshgrid to apply to ---
    inputShape = inputBeam.shape
    x = np.linspace(-8*w0, 8*w0, inputShape[0])
    y = np.linspace(-8*w0, 8*w0, inputShape[1])
    X, Y = np.meshgrid(x, y)
    rSquare = X**2 + Y**2 #For simplicity
    
    
    # --- Applying Gaussian Optics --- 
    q0 = -1j * z0
    
    q = (A * q0 + B) / (C*q0 + D)
    
    R_z = 1/np.real(1/q)
    w_z = (pi/wavelength * np.imag(1/q))**(-1/2)
    
    A_z = np.max(A0)/np.sqrt(1+(B/z0)**2)
    #z0 is the same as we are only propagating using the B from the matrix
    
    # --- Building the output beam ---
    #Because we are at the focal point of the lens in this example
    field = A_z *\
            np.exp(-(rSquare)/(np.real(w_z**2))) *\
            np.exp(1j * pi/wavelength * (rSquare)/R_z) *\
            np.exp(1j * np.arctan(B/z0))
            
    return field
       
