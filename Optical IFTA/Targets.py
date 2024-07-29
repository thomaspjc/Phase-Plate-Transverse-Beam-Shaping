#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:36:57 2024

Targets to use in the GSA 

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

def superTruncGaussian(inputBeam, w0 = 8e-3, n = 1, trunc = None,
                       extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
    """
    Generates an array with the intensity pattern of a truncated super Gaussian beam transverse profile
    This function is used in association with GSA to be used as a target
    
    Parameters
    ----------
    inputBeam : np.array
        This input is used to set the size of the output target
    w0 : float, optional
        DESCRIPTION. The default is set in the globals.
    n : float, optional
        The power the Gaussian should be raised to before truncation. 
        The default is 1 no increase in power
    trunc : float, optional
        The truncation as a percentage of the beam waist w0. The default is None for no truncation
        For instance trunc = 80 would set the values R > 0.8 * w0 to 0
    extent : np.array, optional
        Extent of the array to build. The default is set in the globals.
    plot : Bool, optional
        Decides if outputs should be plotted. The default is False.

    Returns
    -------
    intensity : np.array
        Transverse Intensity pattern of a Truncated Super Gaussian Beam

    """
    
    shape = inputBeam.shape[0]
    
    x_ = np.linspace(extent[0], extent[1], shape)
    y_ = np.linspace(extent[0], extent[1], shape)
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    intensity = np.abs(np.exp(-(rSquare/ w0**2)**n))**2
    if trunc != None: 
        intensity[np.sqrt(rSquare) > trunc * w0 / 100] = 0
    
    if plot:
        plt.imshow(intensity, extent = [extent[0], extent[1], extent[0], extent[1]])
        plt.title("Normalized Super Truncated Guassian Target")
        plt.xlabel('distance (m)')
        plt.ylabel('distance (m)')
        plt.tight_layout()
        plt.colorbar()
        plt.show()
    
    return intensity

def flatTop(inputBeam, w0 = 6e-4, extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
    
    # --- Extracting data ---
    shape = inputBeam.shape[0]
    
    # --- Building the Circular array ---
    x_ = np.linspace(extent[0], extent[1], shape)
    y_ = np.linspace(extent[0], extent[1], shape)
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    Intensity = np.ones([int(shape), int(shape)])
    Intensity[(np.sqrt(rSquare) > w0)] = 0
    
    if plot:
        plt.imshow(Intensity, extent = [extent[0], extent[1], extent[0], extent[1]])
        plt.title("Normalized Circular Flat Top")
        plt.xlabel('distance (m)')
        plt.ylabel('distance (m)')
        plt.tight_layout()
        plt.colorbar()
        plt.show()
        
    return Intensity

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

def stanford(gridSize = 2**11, logopath = "stanfordLogo.png", plot = False):
    """
    Producing a Stanford logo beam target for the IFTA

    Parameters
    ----------
    gridSize : float, optional
        Size of the array to use for the output. The default is 2**11.
    logopath : string, optional
        file path for the stanford logo in folder. The default is "stanfordLogo.png".
    plot : Bool, optional
        Decides if outputs should be plotted. The default is False.

    Returns
    -------
    resized_logo : np.array
        Extracted Stanford logo with the size defined by the gridSize

    """
    # --- Extracting the image into an array --- 
    logo = cv2.imread(logopath, cv2.IMREAD_GRAYSCALE)

    # --- Resize the logo to the gridSize ---
    logo_height, logo_width = logo.shape
    scaling_factor = min(gridSize / logo_width, gridSize / logo_height)
    new_size = (int(logo_width * scaling_factor), int(logo_height * scaling_factor))
    resized_logo = cv2.resize(logo, new_size)
    resized_logo = normalize(resized_logo)
    
    # --- Plot the resized logo --- 
    if plot:
        plt.imshow(resized_logo)
        plt.title('Target')
        plt.colorbar()
        plt.show()
    return resized_logo

def donutFlatTop(inputBeam, w0 = 4e-3, w1 = (4e-3)/2,
                 extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
    
    # --- Extracting data ---
    shape = inputBeam.shape[0]
    
    # --- Building the Circular array ---
    x_ = np.linspace(extent[0], extent[1], shape)
    y_ = np.linspace(extent[0], extent[1], shape)
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    Intensity = np.ones([int(shape), int(shape)])
    Intensity[(np.sqrt(rSquare) > w0)] = 0
    Intensity[(np.sqrt(rSquare) < w1)] = 0
    
    if plot:
        plt.imshow(Intensity, extent = [extent[0], extent[1], extent[0], extent[1]])
        plt.title("Normalized Circular Flat Top")
        plt.xlabel('distance (m)')
        plt.ylabel('distance (m)')
        plt.tight_layout()
        plt.colorbar()
        plt.show()
        
    return Intensity

def hole(inputBeam, w0 = 4e-3, extent = [-1.27 * 1e-2, 1.27 * 1e-2], plot = False):
    
    # --- Extracting data ---
    shape = inputBeam.shape[0]
    
    # --- Building the Circular array ---
    x_ = np.linspace(extent[0], extent[1], shape)
    y_ = np.linspace(extent[0], extent[1], shape)
    X, Y = np.meshgrid(x_, y_)
    rSquare = (X**2 + Y**2)
    Intensity = inputBeam
    Intensity[(np.sqrt(rSquare) < w0)] = 0
    
    if plot:
        plt.imshow(Intensity, extent = [extent[0], extent[1], extent[0], extent[1]])
        plt.title("Normalized Circular Flat Top")
        plt.xlabel('distance (m)')
        plt.ylabel('distance (m)')
        plt.tight_layout()
        plt.colorbar()
        plt.show()
        
    return Intensity

