#!/usr/bin/python3

# Standard Libraries
import numpy as np
import matplotlib.pyplot as plt

# Type Hint Libraries
from typing import Optional, Tuple, Union, TypeVar, List
import numpy.typing as npt
import matplotlib.figure

# Math Libraries
from scipy.ndimage.filters import convolve

# Image Libraries
import cv2 
import skimage as ski
from skimage import io
from skimage.color import rgb2gray

# Import Functions
from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_vertical_seam import find_optimal_vertical_seam
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
from display_seam import display_seam

def reduce_width(
    img: npt.NDArray[np.uint8], 
    energy_image: npt.NDArray[np.double], 
    ) -> Tuple[npt.NDArray[np.uint8],npt.NDArray[np.double]]:
    """
    Reduces the width on an image using seam-carving for content aware image resizing
    
    Input:
        An image with dimmensions [M,N,3] of data type uint8
        A single-channel image with the result of the energy function, expected format is a numpy array with
        dimmensions [M,N] of data type double (float64)
    Output:
        An image with dimmensions [M,N-1,3] of data type uint8
        A single-channel image with the result of the energy function, expected format is a numpy array with
        dimmensions [M,N-1] of data type double (float64)
    
    Parameters
    ----------
    img : np.ndarray [shape=(M,N,3), dtype=np.uint8]
    seam_vector : np.ndarray [shape=(M,N), dtype=np.double]

    Returns
    ----------
    Tuple[npt.NDArray[np.uint8],npt.NDArray[np.double]]

    Examples
    ----------
    >>> img_input.shape
    >>> (M,N,3)
    >>> img_output, energy_image_output = reduce_width(img_input, energy_image_input)
    >>> img_output.shape
    >>> (M,N-1,3)
    """
    assert img.shape[2] == 3, 'Unexpected number of channels. Pass an image with 3 channels.'
    assert img.dtype == np.uint8, 'Unexpedted dtype. The function expects an RBG image of data type uint8.'
    
    assert len(energy_image.shape) == 2, 'Unexpected number of dimensions. Expecting a 2d numpy array.'
    assert energy_image.dtype == np.double, 'Unexpedted dtype. The function expects a 2D energy map of data type double(float64).'
    
    assert (img.shape[0] == energy_image.shape[0]) and (img.shape[1] == energy_image.shape[1]), 'Image and Energy Image sizes must match'
    
    cum_energy_map_row_traverse = cumulative_minimum_energy_map(energy_image,0)
    vertical_seam_vector = find_optimal_vertical_seam(cum_energy_map_row_traverse)
    
    num_cols = img.shape[1] 
    num_rows = img.shape[0]
    
    # Creating a 1D mask with same dimentions as image filled with True
    mask = np.ones((num_rows, num_cols), dtype=bool)
    
    # Changes values to False in each row combined with the column index stored in the vertical_seam vector
    for i in range(len(vertical_seam_vector)):
        mask[i,int(vertical_seam_vector[i])] = False
    
    # Stacks mask in 3D
    mask_3D = np.stack([mask] * 3, axis=2)

    img_out = np.zeros((num_rows,num_cols-1))
    img_out = np.stack([img_out] * 3, axis=2)
    img_out = img_out.astype(np.uint8)
    
    energy_out = np.zeros((num_rows,num_cols-1))

    # Filling the img_out with masked rows (reduced by one pixel)
    for k in range(3):
        for i in range(num_rows):
            img_out[i,:,k] = img[i,:,k][mask_3D[i,:,k]]
    
    # Filling the energy_out with masked rows (reduced by one pixel)
    for i in range(num_rows):
        energy_out[i,:] = energy_image[i,:][mask[i,:]]
    
    return img_out,energy_out