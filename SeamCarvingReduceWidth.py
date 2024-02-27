#!/usr/bin/python3

# Standard Libraries
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

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
from reduce_height import reduce_height
from reduce_width import reduce_width

def main(input_file_name,output_file_name,num_pixels_remove):

    os.chdir('..\\Zambrano_Ricardo_ASN2_py')
    print(os.getcwd())

    print('Loading the image...')

    if input_file_name[-4:] == '.jpg':
        PATH_IMG = '..\\Zambrano_Ricardo_ASN2_py\\' + input_file_name
    else:
        PATH_IMG = '..\\Zambrano_Ricardo_ASN2_py\\' + input_file_name + '.jpg'

    if output_file_name[-4:] == '.png':
        PATH_SAVE = '..\\Zambrano_Ricardo_ASN2_py\\' +  + output_file_name
    else:
        PATH_SAVE = '..\\Zambrano_Ricardo_ASN2_py\\' + output_file_name + '.png'


    img_raw = io.imread(PATH_IMG)
    img = img_raw.copy()

    # Showing image used for the excercise
    plt.imshow(img_raw)
    plt.title("Original Image")
    plt.show()

    print('The original dimension of the image are: ',img_raw.shape)

    energy_map = energy_image(img)

    for _ in range(num_pixels_remove):
        img,energy_map = reduce_width(img,energy_map)

    io.imsave(PATH_SAVE,img)

    # Showing outpur image 
    plt.imshow(img)
    plt.title("Reduced Image")
    plt.show()

    print('The dimensions of the output image are: ',img.shape)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pass a string with the name JPG file in the folder, a name for the output file, and the number of pixels to reduce from the width of the image')

    parser.add_argument('-in','--input_file_name', type=str, default=False, action='store', required=True, help="String with name of JPG file in folder")
    parser.add_argument('-out','--output_file_name', type=str, default=False, action='store', required=True, help="String with name of the output name of the PNG file")
    parser.add_argument('-pix','--num_pixels_remove', type=int, default=100, action='store', required=True, help="Number of most-informative features to show")
    
    args = parser.parse_args()
    main(str(args.input_file_name), str(args.output_file_name), int(args.num_pixels_remove))