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
      for j in range(h):
        for k in range(w):
          region = images[i, j:j+kh, k:k+kw]
          images_conv[i, j, k] = np.sum(region * kernel)
    return images_conv