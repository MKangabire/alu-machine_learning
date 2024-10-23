#!/usr/bin/env python3
"""a function that performs a valid convolution on grayscale img"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """a function that performs a valid convolution on grayscale img"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = h - kh + 1
    pw = w - kw + 1
    images_conv = np.zeros((m, ph, pw))
    for i in range(m):
        for j in range(ph):
            for k in range(pw):
                region = images[i, j:j + kh, k:k + kw]
                convolved_value = np.sum(region * kernel)  # Calculate convolved value
                images_conv[i, j, k] = convolved_value   # Store it in the output array
                
                # Debugging output
                print("Image {}, Position ({}, {}), Convolved Value: {}".format(i, j, k, convolved_value))
    
    return images_conv
