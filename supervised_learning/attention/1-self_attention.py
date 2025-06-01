#!/usr/bin/env python3
"""Attention in RNN"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention implements the attention mechanism as
    
    Attributes:
        W (tf.keras.layers.Dense): Dense layer applied to
        U (tf.keras.layers.Dense): Dense layer applied to
        V (tf.keras.layers.Dense): Dense layer applied to 
    """

    def __init__(self, units):
        """
        Class constructor for SelfAttention.

        Args:
            units (int): Number of hidden
        """
        super(SelfAttention, self).__init__()

        # Dense layer to transform decoder hidden state
        self.W = tf.keras.layers.Dense(units)

        # Dense layer to transform encoder hidden states
        self.U = tf.keras.layers.Dense(units)

        # Dense layer to compute the attention scores
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Computes the attention weights and context vector.

        Args:
            s_prev (tf.Tensor): Previous decoder
            hidden_states (tf.Tensor): Encoder

        Returns:
            context (tf.Tensor): Context vector of
            weights (tf.Tensor): Attention weights
        """
        # Expand s_prev to shape (batch, 1, units)
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)

        # Apply W and U transformations and compute the attention scores
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        # score shape: (batch, input_seq_len, 1)

        # Compute the attention weights using softmax over the time axis
        weights = tf.nn.softmax(score, axis=1)
        # weights shape: (batch, input_seq_len, 1)

        # Compute the context vector as the weighted
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        # context shape: (batch, units)

        return context, weights
