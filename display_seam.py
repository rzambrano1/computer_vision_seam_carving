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
import energy_image
import cumulative_minimum_energy_map
import find_optimal_vertical_seam
import find_optimal_horizontal_seam

def display_seam(
    img: npt.NDArray[np.uint8], 
    seam_vector:npt.NDArray[np.double], 
    seam_direction: str
    ) -> matplotlib.figure.Figure:
    """
    Display the input image and plot the seam in the selected direction on top of it
    
    Input:
        An image with dimmensions MxNx3 of data type uint8,
        A 1D numpy array of data type double (float64) containing the row or column indices of the pixels from the seam, 
        An string indicating the type of the seam: 'HORIZONTAL' or 'VERTICAL'
    Output:
        A plot of the image and the seam
    
    Parameters
    ----------
    img : np.ndarray [shape=(M,N,3), dtype=np.uint8]
    seam_vector : np.ndarray [shape=(M,), dtype=np.double] for seam_direction == 'VERTICAL'
                  np.ndarray [shape=(N,), dtype=np.double] for seam_direction == 'HORIZONTAL'
    seam_direction : 'HORIZONTAL' or 'VERTICAL'

    Returns
    ----------
    0

    Examples
    ----------
    >>> find_optimal_horizontal_seam_1(img, seam_vector,'VERTICAL')
    >>> 0
    """
    assert img.shape[2] == 3, 'Unexpected number of channels. Pass an image with 3 channels.'
    assert img.dtype == np.uint8, 'Unexpedted dtype. The function expects an RBG image of data type uint8.'
    
    assert seam_vector.dtype == np.double, 'Expecting a numpy array of the data type double (float64)'
    assert seam_direction in ['HORIZONTAL','VERTICAL'], "Unexpected seam direction. Options: ['HORIZONTAL','VERTICAL']"
    
    if seam_direction == 'HORIZONTAL':
        flag = img.shape[1] == len(seam_vector)
        if flag == False:
            ValueError("The length of the seam_vector does not match the number of columns in img") 
        
        # Creating the seam line
        xx = np.arange(0, img.shape[1], 1, dtype=np.double)
        yy = seam_vector
        
        # Display the image and plots the seam line on top of the image
        plt.imshow(img)
        plt.plot(xx, yy, color="white", linewidth=1) 
        #plt.axis('off') 
        plt.show() 
            
    elif seam_direction == 'VERTICAL':
        flag = img.shape[0] == len(seam_vector)
        if flag == False:
            ValueError("The length of the seam_vector does not match the number of rows in img")
        
        # Creating the seam line
        xx = seam_vector
        yy = np.arange(0, img.shape[0], 1, dtype=np.double)
        
        # Display the image and plots the seam line on top of the image
        plt.imshow(img)
        plt.plot(xx, yy, color="white", linewidth=1) 
        #plt.axis('off') 
        plt.show() 
    
    return 0
