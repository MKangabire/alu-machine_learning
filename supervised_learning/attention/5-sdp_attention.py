#!/usr/bin/env python3
"""Attention in RNN"""

import tensorflow as tf

def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Args:
        Q (tf.Tensor): Query tensor of shape (..., seq_len_q, dk)
        K (tf.Tensor): Key tensor of shape (..., seq_len_v, dk)
        V (tf.Tensor): Value tensor of shape (..., seq_len_v, dv)
        mask (tf.Tensor, optional): Tensor broadcastable to

    Returns:
        output (tf.Tensor): Attention output tensor of
        weights (tf.Tensor): Attention weights of shape
    """
    # Get depth of the key vectors
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Calculate the dot product between Q and K^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale the dot products
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply the mask (if any)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax to get the attention weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the attention weights by the value matrix
    output = tf.matmul(weights, V)  # (..., seq_len_q, dv)

    return output, weights
