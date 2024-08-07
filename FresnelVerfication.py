#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:59:49 2024

Testing the validity of the Fresnel GSA program using Gaussian Optics

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from padding import fastOn, fastOff
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import pandas as pd

# --- Globals ---
wavelength = 253 * 1e-9
w0 = 4 * 1e-3
f = 1.2
#k0 = 2* pi / wavelength
extent = [-8 * w0, 8 * w0]
z0 = pi/wavelength * w0**2
q0 = 0 + 1j * z0




def Transfer(inputQ, transferMatrix):
    
    # --- Extracting the Parameters ---
    (A,B),(C,D) = transferMatrix[0], transferMatrix[1]
    # --- Applying the transfer Matrix ---
    qPrime = (A * inputQ + B)/(C*inputQ + D)
    
    # --- Extracting new parameters for the beam ---
    w_z = np.abs((-pi/wavelength * np.imag(1/qPrime))**(-1/2))
    R_z = ((1/qPrime).real)**(-1)
    
    return qPrime, w_z, R_z
    
def Lens(f):
    return np.array([[1,0],[-1/f, 1]])

def Propagate(z):
    return np.array([[1, z],[0, 1]])

qLens = Transfer(q0, Lens(f))[0]

zSpace = np.linspace(0, 2.1, int(1e4))
waist = np.transpose(np.array([Transfer(qLens, Propagate(z)) for z in zSpace]))[1]

mask = (zSpace > f-0.02) & (zSpace < f+0.02)

# --- Extracting Data from the Fresnel Simulation ---
filepath = 'FresnelData1.csv'
dataFresnel = pd.read_csv(filepath, header = None, index_col = None, skiprows = 1)
distanceFresnel = dataFresnel[0].values.astype(float)
waistFresnel = dataFresnel[1].values.astype(float)
pixelSize = dataFresnel[4].values.astype(float)
maskCloseUp = (distanceFresnel > 1.185) & (distanceFresnel < 1.215)
#For residuals need a specied waist
waistResidual = np.transpose(np.array([Transfer(qLens, Propagate(z)) for z in distanceFresnel]))[1]
residuals = waistResidual - waistFresnel

# --- Plotting ---
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

#Long Shot Plot
ax1 = plt.subplot(gs[0,0])
ax1.plot(zSpace, waist, color = 'C0')
ax1.plot(zSpace, -waist, color = 'C0')
ax1.plot(distanceFresnel, waistFresnel, 'rx')
ax1.errorbar(distanceFresnel, waistFresnel, yerr = pixelSize, fmt = '.', color = 'r', capsize = 2, alpha = 0.4)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_title("Full Propagated Width")
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel(f'Beam Waist ($1/e^2$) (m)')
#Close Up Plot
ax2 = plt.subplot(gs[0,1])
ax2.ticklabel_format(style='sci', axis='y')
ax2.plot(zSpace[mask], waist[mask], color = 'C0')
ax2.plot(zSpace[mask], -waist[mask], color = 'C0')
ax2.plot(distanceFresnel[maskCloseUp], waistFresnel[maskCloseUp], 'rx')
ax2.errorbar(distanceFresnel[maskCloseUp], waistFresnel[maskCloseUp], yerr = pixelSize[maskCloseUp], fmt = '.',
             color = 'r', capsize = 2, alpha = 0.4)
ax2.set_title("Width at Focal Point")
ax2.set_xlabel('Distance (m)')

#Residuals Plot
ax3 = plt.subplot(gs[1,:])
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.set_title("Residuals")
ax3.axhline(color = 'k', linestyle = '--')
ax3.plot(distanceFresnel, residuals, 'kx')
ax3.errorbar(distanceFresnel, residuals, yerr = pixelSize, fmt = '.',
             color = 'r', capsize = 2, alpha = 0.4)
ax3.set_xlabel('Distance (m)')
ax3.set_ylabel(f'Residuals (m)')

fig.suptitle("Comparing the Fresnel Propagator to a Gaussian Optics waist trace", size = 20)

plt.tight_layout()
plt.show()










































