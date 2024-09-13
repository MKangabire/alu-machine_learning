#!/usr/bin/env python3
"""Add two arrays element-wise if they have the same shape."""

def add_arrays(arr1, arr2):
    """Add two arrays element-wise if they have the same shape."""
    if len(arr1) != len(arr2):
        return None  
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
