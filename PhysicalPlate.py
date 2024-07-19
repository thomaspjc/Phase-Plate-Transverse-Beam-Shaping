#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:44:28 2024

Setting up the limitations in our simulation of the phase plate 

@author: thomas
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def Box(inputPhase, group):
    # Get the shape of the input array
    rows, cols = inputPhase.shape
    
    # --- Ensure the operation of grouping is possible ---
    if rows % group != 0 or cols % group != 0:
        raise IndexError("The input array is not able to be separated using the group Size")
    
    
    # Reshape the array to group the elements
    reshaped = inputPhase.reshape(rows // group, group, cols // group, group)

    # Sum within the groups
    group_sums = reshaped.sum(axis=(1, 3))
    
    # Expand the group sums back to the original shape
    expanded_sums = np.repeat(np.repeat(group_sums, group, axis=0), group, axis=1)
    
    return expanded_sums/group**2

"""
arr = np.array([[ 1,  2,  3,  4], 
                [ 5,  6,  7,  8], 
                [ 9, 10, 11, 12], 
                [13, 14, 15, 16]])

cmap_cubehelix = sns.color_palette("ch:start=0.5,rot=-0.75", as_cmap=True)

arr = np.random.rand(2**11,2**11)
plt.figure(figsize=(6, 6))
sns.heatmap(arr, annot=False, fmt=".1f", cmap=cmap_cubehelix)
plt.title('Heatmap of Summed Groups')
plt.show()
print(arr, '\n\n')
group_size = 2**5
result = Box(arr, group_size)

# Plotting the heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(result, annot=False, fmt=".1f", cmap=cmap_cubehelix)
plt.title('Heatmap of Summed Groups')
plt.show()

'''plt.imshow(result)
plt.colorbar()
plt.show()'''"""











































