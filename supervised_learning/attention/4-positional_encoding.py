#!/usr/bin/env python3
"""Attention in RNN"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.

    Args:
        max_seq_len (int): Maximum sequence length.
        dm (int): Depth of the model.

    Returns:
        np.ndarray: Shape (max_seq_len, dm) containing the positional encoding vectors.
    """
    # Create a matrix of shape (max_seq_len, dm) with all positions and dimensions
    pos = np.arange(max_seq_len)[:, np.newaxis]  # shape (max_seq_len, 1)
    i = np.arange(dm)[np.newaxis, :]             # shape (1, dm)

    # Calculate the angle rates
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    angle_rads = pos * angle_rates  # shape (max_seq_len, dm)

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads
