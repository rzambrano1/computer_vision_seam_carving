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


def energy_image(im: npt.NDArray[np.uint8]) -> npt.NDArray[np.double]:
    """
    Input Arguments: 
        An image with dimmensions MxNx3 of data type uint8
    
    Output: 
        The result of passing the image to the energy function e_1(im) = |d(im)/dx| + |d(im)/dy|
        The partial derivatives operators use the optimal 8 bit integer valued 3x3 filter 
        stemming from Scharr's theory

    Parameters
    ----------
    im : np.ndarray [shape=(M,N,3)]

    Returns
    ----------
    energy_map : np.ndarray [shape=(M,N)]

    Examples
    ----------
    >>> energy_image(im)
    >>> array([[ 2304., 26802., 14436., ..., 10188., 15308.,  8960.],
    ...        [  846., 22428., 14436., ...,  8730., 13286., 11364.],
    ...        [ 4608., 12132.,  7854., ...,  8370.,  8526.,  7680.],
    ...        ...,
    ...        [12978., 15282., 13824., ..., 35784., 35784., 22428.],
    ...        [17586., 14436., 10062., ..., 10296., 10908., 23652.],
    ...        [10674., 11286.,  1458., ...,  8370.,  9828.,  7524.]])
    """
    assert im.shape[2] == 3, 'Unexpected number of channels. Pass an image with 3 channels.'
    assert im.dtype == np.uint8, 'Unexpedted dtype. The function expects an RBG image of data type uint8.'
    
    ### Creating Partial Derivative Operators ###
    
    # 2D version of the Scharr operator
    filter_dy = np.array([
        [47.0, 162.0, 47.0],
        [0.0, 0.0, 0.0],
        [-47.0, -162.0, -47.0],
    ])

    # This converts it from a 2D filter to a 3D filter, replicating the same filter for each channel: R, G, B
    filter_dy = np.stack((filter_dy,filter_dy,filter_dy), axis=2)
    
    # 2D version of the Scharr operator
    filter_dx = np.array([
        [47.0, 0.0, -47.0],
        [162.0, 0.0, -162.0],
        [47.0, 0.0, -47.0],
    ])

    # This converts it from a 2D filter to a 3D filter, replicating the same filter for each channel: R, G, B
    filter_dx = np.stack((filter_dx,filter_dx,filter_dx), axis=2)

    ### Converting input into desired output data type ###
    img = im.astype(np.double)
    
    # Convolving the image to get the gradient on each channel
    convolved_img = np.absolute(convolve(img, filter_dx)) + np.absolute(convolve(img, filter_dy))

    # Adding the energies in the red, green, and blue channels
    energy_map = convolved_img.sum(axis=2)
    
    return energy_map