#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:33:48 2024

Fresnel Optics simulation of a Gaussian beam propagated through a phase plate
and to the fourier plane

@author: thomas
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import cv2
from numpy import pi as pi

# --- Globals --- 
gridSize = 1000

def Gaussian(mu_x, mu_y, sigma_x, sigma_y, gridSize = 1000, A0 = 1, phase = True, plot = False):
    """
    Input Gaussian beam simulated from a Fresnel Slow envelope approximation 

    Returns
    -------
    None.

    """
    
    # --- Create a meshgrid --- 
    x = np.linspace(0, gridSize - 1, gridSize)
    y = np.linspace(0, gridSize - 1, gridSize)
    X, Y = np.meshgrid(x, y)
    
    
    # --- Defining the intensity ---
    intensity = A0 * np.exp(-(((X - mu_x)**2 / (2 * sigma_x**2)) +\
                    ((Y - mu_y)**2 / (2 * sigma_y**2))))
    # --- Setting the phase --- 
    if not isinstance(phase, np.ndarray):
        phase = np.zeros(gridSize)
    
    # --- Calculate the Gaussian distribution --- 
    gaussian = np.empty_like(intensity, dtype="complex")
    gaussian.real = intensity * np.cos(phase)
    gaussian.imag = intensity * np.sin(phase)
    
    # --- Plotting --- 
    if plot:
        plt.imshow(np.power(np.abs(gaussian.real), 2))
    
    return gaussian

phase = np.random.rand(gridSize, gridSize)

Gaussian(gridSize/2, gridSize/2, gridSize/5, gridSize/5, gridSize = gridSize, plot = True, phase = phase)
