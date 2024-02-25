#!/usr/bin/python3

# Standard Libraries
import numpy as np
import matplotlib.pyplot as plt

# Type Hint Libraries
from typing import Optional, Tuple, Union, TypeVar, List
import numpy.typing as npt

# Math Libraries
from scipy.ndimage.filters import convolve

# Image Libraries
import cv2 
import skimage as ski
from skimage import io
from skimage.color import rgb2gray

def cumulative_minimum_energy_map(energy_map: npt.NDArray[np.double],seam_direction: int=0) -> npt.NDArray[np.double]:
    """
    Computes the cumulative minimum energy map using dyamic programming as formulated by Avidan and Shamir in
    'Seam Carving for Content-Aware Image Resizing'
    source: http://graphics.cs.cmu.edu/courses/15-463/2012_fall/hw/proj3-seamcarving/imret.pdf
    
    Input:
        A single-channel image with the result of the energy function, expected format is a numpy array
        of data type double (float64)
        An integer indicating the direction of the desired seam. 0 traverses the image's rows
        and calculates a vertical seam. 1 traverses the image's columns and calculates an horizontal seam 
    Output:
        A 2D numpy array representing the minimum energy map. A numpy array of data type double (float64)
    
    Parameters
    ----------
    energy_map : np.ndarray [shape=(M,N)]
    direction : int

    Returns
    ----------
    cumulative_energy_map : np.ndarray [shape=(M,N)]

    Examples
    ----------
    >>> cumulative_minimum_energy_map(energy_map,0)
    >>> array([[ 2304., 26802., 14436., ..., 10188., 15308.,  8960.],
    ...        [ 3150., 24732., 26802., ..., 10548., 22246., 20324.],
    ...        [ 5454., 12978., 22290., ..., 11646., 17256., 19044.],
    ...        ...,
    ...        [25956., 28260., 25548., ..., 81786., 81786., 75342.],
    ...        [30564., 27414., 23458., ..., 46080., 33336., 46080.],
    ...        [25110., 21348., 11520., ..., 18666., 20124., 18432.]])

    >>> cumulative_minimum_energy_map(energy_map,1)
    >>> array([[ 2304., 27648., 36864., ..., 12006., 24038., 22246.],
    ...        [  846., 23274., 26568., ..., 10548., 21656., 19890.],
    ...        [ 4608., 12978., 15534., ..., 11646., 16896., 16206.],
    ...        ...,
    ...        [12978., 28260., 28260., ..., 50688., 46080., 33336.],
    ...        [17586., 25110., 21348., ..., 14058., 19278., 33480.],
    ...        [10674., 21960., 12744., ..., 12132., 18198., 17352.]])
    """
    assert len(energy_map.shape) == 2, 'Unexpected number of dimensions. Expecting a 2d numpy array.'
    assert energy_map.dtype == np.double, 'Unexpedted dtype. The function expects a 2D energy map of data type double(float64).'
    assert seam_direction in [0,1], 'ValueError: specify a valid direction for the seam.\n==> 0 for a verical seam\n==> 1 for a horizontal seam'
    
    cumulative_energy_map = np.zeros_like(energy_map)
    row_size = energy_map.shape[0]
    cols_size = energy_map.shape[1]
    
    if seam_direction == 0:
        
        # Setting the first row of the output equal to the first row of the energy map
        cumulative_energy_map[0,:] = energy_map[0,:]
        
        # Traversing the energy map across rows
        for i in range(1,row_size):
            for j in range(0,cols_size):
                if j == 0:
                    cumulative_energy_map[i,j] = energy_map[i,j] + min(energy_map[i-1,j],energy_map[i-1,j+1])
                elif j == (cols_size - 1):
                    cumulative_energy_map[i,j] = energy_map[i,j] + min(energy_map[i-1,j-1],energy_map[i-1,j])
                else:
                    cumulative_energy_map[i,j] = energy_map[i,j] + min(energy_map[i-1,j-1],energy_map[i-1,j],energy_map[i-1,j+1])
        
    if seam_direction == 1:
        
        # Setting the first column of the output equal to the first rcolumn of the energy map
        cumulative_energy_map[:,0] = energy_map[:,0]
        
        # Traversing the energy map across colums
        for j in range(1,cols_size):
            for i in range(0,row_size):
                if i == 0:
                    cumulative_energy_map[i,j] = energy_map[i,j] + min(energy_map[i,j-1],energy_map[i+1,j-1])
                elif i == (row_size - 1):
                    cumulative_energy_map[i,j] = energy_map[i,j] + min(energy_map[i,j-1],energy_map[i-1,j-1])
                else:
                    cumulative_energy_map[i,j] = energy_map[i,j] + min(energy_map[i-1,j-1],energy_map[i,j-1],energy_map[i+1,j-1])
    
    return cumulative_energy_map