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

def find_optimal_horizontal_seam(cum_energy_map:npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """
    Find the optimal horizontal seam in a 2D array representing the cumulative minimum energy map
    
    Input:
        A single-channel cumulative minimum energy map of an image, expected format is a numpy array
        of data type double (float64)
    Output:
        A 1D numpy array containing the row indices of the pixels from the seam in each column.
        The selected pixels  minimum energy map. A numpy array of data type double (float64)
    
    Parameters
    ----------
    cum_energy_map : np.ndarray [shape=(M,N)]

    Returns
    ----------
    vertical_seam : np.ndarray [shape=(N,)]

    Examples
    ----------
    >>> find_optimal_horizontal_seam_1(cum_energy_map)
    >>>
    """
    assert len(energy_map.shape) == 2, 'Unexpected number of dimensions. Expecting a 2d numpy array.'
    assert energy_map.dtype == np.double, 'Unexpedted dtype. The function expects a 2D energy map of data type double(float64).'
    
    row_size, cols_size = cum_energy_map.shape
    
    # First look to the minimum value at the last column, the end of the horizontal seam
    first_minVal_indx = np.argmin(cum_energy_map[:,cols_size-1])

    # Record the value of the first occurence of the minimum value
    first_minVal = cum_energy_map[first_minVal_indx,cols_size-1]
    
    # Checking if the value occurs in other rows in the last column
    occurences_minVal = np.count_nonzero(cum_energy_map[:,cols_size-1] == first_minVal)

    # Checking if the minimumvalue is unique or not
    unique_minVal = not bool(occurences_minVal-1)
    
    if unique_minVal:
        print('unique branch')
        # Create a vector to store the row indexes of the seam
        horizontal_seam = np.zeros((cols_size,),dtype=np.double)
        
        # Assigning row index value of the pixel with the minVal in the last column of the 
        # cumulative energy map to the seam-vector 
        horizontal_seam[cols_size-1] = first_minVal_indx
        
        # Filling the row indexes in the vector. Traverses from last column to first column
        for j in range(cols_size-2, -1, -1):
            leftup = int(horizontal_seam[j+1]-1)
            leftdown = int(horizontal_seam[j+1]+2)
            offset = -1
            
            ## Handling slices near the edges of the image ##
            if leftup < 0:
                leftup = 0
                offset = 0 # On the upper edge the offset need to change to 0
            if leftdown > (cols_size-1):
                leftdown = cols_size-1

            if (leftup == leftdown) and (leftup == 0):
                leftdown = leftdown + 2
                offset = 0 # On the upper edge the offset need to change to 0
            elif (leftup == leftdown) and (leftdown == cols_size-1):
                leftup = leftup - 2
            ## --------------------------------------------- ##
            
            horizontal_seam[j] = horizontal_seam[j+1] + np.argmin(cum_energy_map[leftup:leftdown,j]) + offset
    
    else:
        print('several mins branch')
        # Create a matrix to store the potential vectors in the rows. Each candidate vector row
        # stores the indexes of the candidate seam
        horizontal_seam_matrix = np.zeros((occurences_minVal,cols_size),dtype=np.double)
        
        # Finding row indexes of all cuurences of the min value
        index_list = np.where(cum_energy_map[:,cols_size-1] == first_minVal)[0]
        
        # Assigning the row index values of the pixels with the minimun values 
        # in the last columnof the cumulative energy map to the seam-vector matrix
        for i in range(horizontal_seam_matrix.shape[0]):
            horizontal_seam_matrix[i,cols_size-1] = index_list[i]
        
        # Filling the matrix with the row index values of each seam candidate. Traversers from last column to first
        for i in range(horizontal_seam_matrix.shape[0]):
            for j in range(cols_size-2, -1, -1):
                leftup = int(horizontal_seam_matrix[i,j+1]-1)
                leftdown = int(horizontal_seam_matrix[i,j+1]+2)
                offset = -1
                
                ## Handling slices near the edges of the image ##
                if leftup < 0:
                    leftup = 0
                    offset = 0 # On the upper edge the offset need to change to 0
                if leftdown > (cols_size-1):
                    leftdown = cols_size-1
                    
                if (leftup == leftdown) and (leftup == 0):
                    leftdown = leftdown + 2
                    offset = 0 # On the upper edge the offset need to change to 0
                elif (leftup == leftdown) and (leftdown == cols_size-1):
                    leftup = leftup - 2
                ## --------------------------------------------- ##

                # The row index has to be connected to the pixel on the left, thus it moves either up, none, or down 
                # of the index on the left. argmin will output either 0, 1, 2 for up, none, or down. Substracting offset shifts
                # this output, so if there is no change then the index to the left will be the same M[i,j+1] + 0
                horizontal_seam_matrix[i,j] = horizontal_seam_matrix[i,j+1] + np.argmin(cum_energy_map[leftup:leftdown,j]) + offset
    
        # Finding the row with the minimum seam cost
        energy_values = np.zeros_like(horizontal_seam_matrix)
        
        for i in range(energy_values.shape[0]):
            for j in range(energy_values.shape[1]):
                energy_values[i,j] = cum_energy_map[int(horizontal_seam_matrix[i,j]),j]
        
        minimum_energy_row = np.argmin(np.sum(energy_values,axis=1))
        
        # Assigning the vertical seam vector
        horizontal_seam = horizontal_seam_matrix[minimum_energy_row,:]
    
    return horizontal_seam