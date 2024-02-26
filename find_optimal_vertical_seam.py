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

def find_optimal_vertical_seam(cum_energy_map: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """
    Find the optimal vertical seam in a 2D array representing the cumulative minimum energy map
    
    Input:
        A single-channel cumulative minimum energy map of an image, expected format is a numpy array
        of data type double (float64)
    Output:
        A 1D numpy array containing the column indices of the pixels from the seam in each row.
        The selected pixels  minimum energy map. A numpy array of data type double (float64)
    
    Parameters
    ----------
    cum_energy_map : np.ndarray [shape=(M,N)]

    Returns
    ----------
    vertical_seam : np.ndarray [shape=(M,)]

    Examples
    ----------
    >>> find_optimal_vertical_seam_1(cum_energy_map)
    >>> array([261., 261., 261., ..., 192., 191., 191.])
    """
    assert len(cum_energy_map.shape) == 2, 'Unexpected number of dimensions. Expecting a 2d numpy array.'
    assert cum_energy_map.dtype == np.double, 'Unexpedted dtype. The function expects a 2D energy map of data type double(float64).'
    
    row_size, cols_size = cum_energy_map.shape
    
    # First look to the minimum value at the last row, the end of the seam
    first_minVal_indx = np.argmin(cum_energy_map[row_size-1,:])
    
    # Record the value of the first occurence of the minimum value
    first_minVal = cum_energy_map[row_size-1,first_minVal_indx]
    
    # Checking if the value occurs in other places in the last row
    occurences_minVal = np.count_nonzero(cum_energy_map[row_size-1,:] == first_minVal)
    
    # Checking if the minimumvalue is unique or not
    unique_minVal = not bool(occurences_minVal-1)
    
    if unique_minVal:
        
        # Create a vector to store the column indexes of the seam
        vertical_seam = np.zeros((row_size,),dtype=np.double)
        
        # Assigning column index value of the pixel with the minVal in the last row of the 
        # cumulative energy map to the seam-vector 
        vertical_seam[row_size-1] = first_minVal_indx
        
        # Filling the column indexes in the vector
        for i in range(row_size-2, -1, -1):
            upleft = int(vertical_seam[i+1]-1)
            upright = int(vertical_seam[i+1]+2)
            offset = -1
            
            ## Handling slices near the edges of the image ##
            if upleft < 0:
                upleft = 0
                offset = 0 # On the upper edge the offset need to change to 0
            if upright > (cols_size-1):
                upright = cols_size-1

            if (upleft == upright) and (upleft == 0):
                upright = upright + 2
                offset = 0 # On the upper edge the offset need to change to 0
            elif (upleft == upright) and (upright == cols_size-1):
                upleft = upleft - 2
            ## --------------------------------------------- ##
            
            vertical_seam[i] = vertical_seam[i+1] + np.argmin(cum_energy_map[i,upleft:upright]) + offset
    
    else:
        
        # Create a matrix to store the potential vectors in the columns. Each candidate vector column
        # stores the indexes of the candidate seam
        vertical_seam_matrix = np.zeros((row_size,occurences_minVal),dtype=np.double)
        
        # Finding column indexes of all cuurences of the min value
        index_list = np.where(cum_energy_map[row_size-1,:] == first_minVal)[0]
        
        # Assigning the column index values of the pixels with the minimun values 
        # in the last row cumulative energy map to the seam-vector matrix
        for j in range(vertical_seam_matrix.shape[1]):
            vertical_seam_matrix[row_size-1,j] = index_list[j]
        
        # Filling the matrix with column index values of each seam candidate
        for i in range(row_size-2, -1, -1):
            for j in range(vertical_seam_matrix.shape[1]):
                upleft = int(vertical_seam_matrix[i+1,j]-1)
                upright = int(vertical_seam_matrix[i+1,j]+2)
                offset = -1
                
                ## Handling slices near the edges of the image ##
                if upleft < 0:
                    upleft = 0
                    offset = 0 # On the upper edge the offset need to change to 0
                if upright > (cols_size-1):
                    upright = cols_size-1

                if (upleft == upright) and (upleft == 0):
                    upright = upright + 2
                    offset = 0 # On the upper edge the offset need to change to 0
                elif (upleft == upright) and (upright == cols_size-1):
                    upleft = upleft - 2
                ## --------------------------------------------- ##

                # The column index has to be connected to the pixel below, thus it moves either right, none, or left 
                # of the index below. argmin will output either 0, 1, 2 for left,none, or right. Substracting 1 shifts
                # this output to the left, so if there is no change then the index below will be the same M[i+1,j] + 0
                vertical_seam_matrix[i,j] = vertical_seam_matrix[i+1,j] + np.argmin(cum_energy_map[i,upleft:upright]) + offset
    
        # Finding the column with the minimum seam cost
        energy_values = np.zeros_like(vertical_seam_matrix)
        
        for i in range(energy_values.shape[0]):
            for j in range(energy_values.shape[1]):
                energy_values[i,j] = cum_energy_map[i,int(vertical_seam_matrix[i,j])]
        
        minimum_energy_column = np.argmin(np.sum(energy_values,axis=0))
        
        # Assigning the vertical seam vector
        vertical_seam = vertical_seam_matrix[:,minimum_energy_column]
        
    return vertical_seam