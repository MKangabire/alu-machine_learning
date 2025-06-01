#!/usr/bin/env python3
"""Attention in RNN"""

import tensorflow as tf


sdp_attention = __import__('5-sdp_attention').sdp_attention

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Performs multi-head attention.
    """

    def __init__(self, dm, h):
        """
        Class constructor.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.

        Raises:
            AssertionError: If dm is not divisible by h.
        """
        super(MultiHeadAttention, self).__init__()
        assert dm % h == 0, "dm must be divisible by h"

        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (h, depth).

        Args:
            x (tf.Tensor): Tensor of shape (batch_size, seq_len, dm)
            batch_size (int): Batch size.

        Returns:
            tf.Tensor: Shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Applies multi-head attention.

        Args:
            Q (tf.Tensor): Query input of shape (batch, seq_len_q, dk)
            K (tf.Tensor): Key input of shape (batch, seq_len_v, dk)
            V (tf.Tensor): Value input of shape (batch, seq_len_v, dv)
            mask (tf.Tensor): Mask tensor or None

        Returns:
            output (tf.Tensor): Shape (batch, seq_len_q, dm)
            weights (tf.Tensor): Shape (batch, h, seq_len_q, seq_len_v)
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)  # (batch, seq_len_q, dm)
        K = self.Wk(K)  # (batch, seq_len_v, dm)
        V = self.Wv(V)  # (batch, seq_len_v, dm)

        Q = self.split_heads(Q, batch_size)  # (batch, h, seq_len_q, depth)
        K = self.split_heads(K, batch_size)  # (batch, h, seq_len_v, depth)
        V = self.split_heads(V, batch_size)  # (batch, h, seq_len_v, depth)

        # scaled dot-product attention
        attention_output, attention_weights = sdp_attention(Q, K, V, mask)

        # (batch, seq_len_q, h, depth)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # concatenate heads
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.dm))  # (batch, seq_len_q, dm)

        # final linear layer
        output = self.linear(concat_attention)  # (batch, seq_len_q, dm)

        return output, attention_weights
