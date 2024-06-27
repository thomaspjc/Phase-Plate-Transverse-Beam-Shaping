#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:18:15 2024

@author: thomas
"""

import numpy as np

def paddingOn(inputArray, size):
    """
    Returns a padded array to allow for more precise FFTs

    Parameters
    ----------
    inputArray : np.array
        input image that should be padded
    size : int
        The padded array will be (size + 1)x bigger in shape

    Raises
    ------
    ValueError
        When the input shape is odd as this causes wrong centering of the array

    Returns
    -------
    paddedOut : np.array
        input array centered with zeroes elsewhere
    """
    
    # --- Determine the shape of the input ---
    inputShape = inputArray.shape
    
    # --- Raise an error for odd shapes to make sure the input is centered ---
    #Could go around this by using NaNs and identification later but much more work
    if inputShape[0] % 2 != 0 or inputShape[1] % 2 != 0:
        raise ValueError ('The input shape has an odd number disallowing centering')
        return
        
        
    # --- Prepare the new padded array --- 
    paddedShape = (inputShape[0] * (1 + size), inputShape[1] * (1 + size))
    paddedOut = np.zeros(paddedShape, dtype=inputArray.dtype)
    
    startRow = int((paddedShape[0] - inputShape[0]) / 2)
    startCol = int((paddedShape[1] - inputShape[1]) / 2)

    # --- Center the Input Array in the padded one ---
    paddedOut[startRow:startRow + inputShape[0], startCol:startCol + inputShape[1]] = inputArray
 
    return paddedOut

def paddingOff(paddedArray, size):
    """
    Removing the padding created through the paddingOn function above
    This allow for fast computing time after FFT modifications are done

    Parameters
    ----------
    paddedArray : np.array
        array whose padding should be removed 
    size : int 
        Should be the same int as when created with paddingOn

    Returns
    -------
    unpadded : np.array
        Image before the padding was added by paddingOn
    """
    
    # --- Determine the shape of the padded array ---
    paddedShape = paddedArray.shape
    
    # --- Extract the unpadded array --- 
    extractShape = np.array(paddedShape)/(1+size)
    
    startRow = int((paddedShape[0] - extractShape[0]) / 2)
    startCol = int((paddedShape[1] - extractShape[1]) / 2)
    print(startRow, startCol)
    unpadded = paddedArray[startRow:(startRow + int(extractShape[0])), startCol:(startCol + int(extractShape[1]))]
    
    return unpadded








































