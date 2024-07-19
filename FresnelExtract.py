#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:27:30 2024

Extracting a data file from the Fresnel GSA Alogrithm
Run through a bunch of different propagation distances

@author: thomas
"""


import FresnelGSA as Fresnel
import pandas as pd
import os
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Globals ---
wavelength = 253 * 1e-9
w0 = 4 * 1e-3
f = 1.2
#k0 = 2* pi / wavelength
extent = [-8 * w0, 8 * w0]
z0 = pi/wavelength * w0**2

filepath = 'FresnelData1.csv'


# --- Initialising the beam propagation ---
inputBeam = Fresnel.Gaussian(sizeFactor=13)
lensBeam = Fresnel.Lens(inputBeam, 1.2)
Z = np.arange(1.185, 1.218, 0.003)

# --- Going through iterations of the propagation distance ---


for z in tqdm(Z, desc = 'Computing'):
    
    # --- Propagating the beam a given distance z --- 
    propBeam = Fresnel.Propagate(lensBeam, z)
    
    # --- Finding the peak Intensity ---
    maxIntensity = np.max(np.abs(propBeam)**2)
    
    # --- Finding the pixel size ---
    shape = propBeam.shape[0]
    pixelSize = 2 * extent[1]/shape
    
    # --- Finding the beam waist --- 
    Intensity = np.abs(propBeam)**2
    IntensitySlice = Intensity[int(shape/2), :]
    waistMask = (IntensitySlice > maxIntensity/np.exp(2))
    
    waist_z = len(IntensitySlice[waistMask])/2 * pixelSize
    
    # --- Some extra plotting ---
    #plt.plot(IntensitySlice[waistMask])
    
    
    # --- Securing the data in a CSV ---
    data = {
        'Distance (m)' : [z],
        'Waist (m)' : [waist_z],
        'Max Intensity' : [maxIntensity],
        'Field' : [propBeam],
        'Pixel Size (m)' : [pixelSize]
        }
    
    dataOut = pd.DataFrame(data)
    if os.path.exists(filepath):
        dataOut.to_csv(filepath, mode='a', header=False, index=False)
    else:
        dataOut.to_csv(filepath, mode='w', header=True, index=False)

    












































