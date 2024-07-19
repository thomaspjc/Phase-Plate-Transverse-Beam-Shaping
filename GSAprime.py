#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:24:14 2024
GS Algorithm from scratch 

Still working on cleaning up the doc strings

@author: thomas
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import cv2
from numpy import pi as pi

# --- Globals --- 
gridSize = 1000



def square(gridSize):
    """
    Creates a square uniform intensity meshgrid
    Parameters
    ----------
    gridSize : TYPE Int
        Size of the n by n meshgrid that will be create

    Returns 
    -------
    im : TYPE numpy meshgrid
        n by n meshgrid 

    """
    im = np.zeros((gridSize, gridSize))
    im[400:600, 400:600] = 1
    plt.imshow(im)
    plt.title('Target')
    plt.colorbar()
    plt.show()
    return im

def normalize(array):
    """
    Normalizing a given array 

    Parameters
    ----------
    array : TYPE meshgrid
        meshgrid that should be normalized

    Returns
    -------
    normalized_array : TYPE meshgrid
        

    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def stanford(gridSize = 1000, logopath = "stanfordLogo.png"):
    """
    Producing a Stanford logo beam target for the IFTA

    Parameters
    ----------
    gridSize : TYPE, optional
        DESCRIPTION. The default is 1000.
    logopath : TYPE, optional
        DESCRIPTION. The default is "stanfordLogo.png".

    Returns
    -------
    resized_logo : TYPE
        DESCRIPTION.

    """
    logo = cv2.imread(logopath, cv2.IMREAD_GRAYSCALE) #extract the logo

    # --- Resize the logo to the gridSize ---
    logo_height, logo_width = logo.shape
    scaling_factor = min(gridSize / logo_width, gridSize / logo_height)
    new_size = (int(logo_width * scaling_factor), int(logo_height * scaling_factor))
    resized_logo = cv2.resize(logo, new_size)
    resized_logo = normalize(resized_logo)
    # --- Plot the resized logo --- 
    plt.imshow(resized_logo)
    plt.title('Target')
    plt.colorbar()
    plt.show()
    return resized_logo

def gaussian(gridSize, mu_x, mu_y, sigma_x, sigma_y):
    """
    Producing an input gaussian meshgrid for the IFTA

    Parameters
    ----------
    gridSize : float
        DESCRIPTION.
    mu_x : float
        mean in x
    mu_y : float
        Dmean in y
    sigma_x : float
        standard deviation in x
    sigma_y : float
        standard deviation in y

    Returns
    -------
    gaussian : meshgrid
        n by n meshgrid image of a gaussian beam

    """
    # --- Create a meshgrid --- 
    x = np.linspace(0, gridSize - 1, gridSize)
    y = np.linspace(0, gridSize - 1, gridSize)
    X, Y = np.meshgrid(x, y)

    # --- Calculate the Gaussian distribution --- 
    gaussian = np.exp(-(((X - mu_x)**2 / (2 * sigma_x**2)) + ((Y - mu_y)**2 / (2 * sigma_y**2))))

    # --- Plot the Gaussian meshgrid --- 
    plt.imshow(gaussian, cmap='viridis', extent=(0, gridSize, 0, gridSize))
    plt.colorbar()
    plt.title("Gaussian Distribution Meshgrid")
    plt.show()

    return gaussian


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
    for field in range(len(outputs)):
        quality += [scalarProduct(target, normalize(outputs[field]))]
    if plotting:
        plt.plot(np.arange(1, len(quality)+1),quality, '.-')
        plt.title('Quality check from scalar product')
        plt.xlabel("Iterations")
        plt.ylabel("Normalized Scalar Product")
        plt.show()
    return
    
def main(iteration = 30, target = square(1000)):
    """
    

    Parameters
    ----------
    iteration : TYPE, optional
        DESCRIPTION. The default is 30.
    target : TYPE, optional
        DESCRIPTION. The default is square(1000).

    Returns
    -------
    outputFP : TYPE
        DESCRIPTION.

    """
    # --- Intialising parameters --- 
    gridSize = 1000
    mu_x, mu_y = gridSize / 2, gridSize / 2  #Center the beam
    sigma_x, sigma_y = gridSize/5, gridSize/5  #Standard deviations

    
    # --- Creating the input beam --- 
    intensityInitial = gaussian(gridSize, mu_x, mu_y, sigma_x, sigma_y)
    phase = np.random.rand(gridSize, gridSize) #initial phase is random
    field = np.empty_like(target, dtype="complex")
    outputFP = []
    
    for i in range(iteration):
        
        # --- Input the initial Intensity and keep the varying phase --- 
        
        field.real = intensityInitial * np.cos(phase)
        field.imag = intensityInitial * np.sin(phase)
        
        # --- Transform to Frequency Domain --- 
        
        field = np.fft.fftshift(np.fft.fft2(field))
        
        outputIter = np.power(np.abs(field), 2)
        
        outputFP += [outputIter] #output fourier plane intensity at each iteration
        outputIter = None
        """plt.imshow(np.power(np.abs(output), 2), cmap = 'viridis')
        title = 'Fourier Plane Output Intensity ' + str(i + 1)
        plt.title(title)
        plt.colorbar()
        plt.show()"""
        
        # --- Replace the intensity with the target intensity --- 
        
        phase = np.angle(field) #Find the new phase after FFT
        
        field.real = target * np.cos(phase)
        field.imag = target * np.sin(phase)
        
        # --- Transform back to Carthesian --- 
        
        field = np.fft.ifft2(np.fft.ifftshift(field))
        
        
        # --- Collect the new phase --- 
        
        phase = np.angle(field)
        

    # --- Plotting the Fourier Plane Intensity --- 
    plt.imshow(outputFP[-1], cmap = 'viridis')#, norm = LogNorm())
    plt.title('Fourier Plane Output Intensity')
    plt.colorbar()
    plt.show()
    
    # --- Plotting the hologram -> Phase Distribution --- 
    holo = hologram(phase)
    plt.imshow(holo, cmap = 'inferno') 
    plt.colorbar()
    plt.title('Normalised Hologram')
    plt.show()
    
    # --- Plotting the quality of the output throughout the process --- 
    scalarPlotting(outputFP, target)
       
    return outputFP
    
outputExt = main(iteration = 30, target = stanford())


    
    
    
    
    
    
    
    
    
