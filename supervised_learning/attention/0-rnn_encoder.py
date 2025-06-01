#!/usr/bin/env python3
"""Attention in RNN"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder encodes a sequence of input word indices for machine translation
    using an embedding layer followed by a GRU (Gated Recurrent Unit).

    Attributes:
        batch (int): The batch size.
        units (int): The number of hidden units in the GRU cell.
        embedding (tf.keras.layers.Embedding): Embedding layer that
        gru (tf.keras.layers.GRU): GRU layer to process the sequence.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor for RNNEncoder.

        Args:
            vocab (int): The size of the input vocabulary.
            embedding (int): Dimensionality of the embedding vectors.
            units (int): Number of hidden units in the GRU cell.
            batch (int): The batch size.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units

        # Embedding layer to convert word indices to dense vectors
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        # GRU layer with return sequences and state
        self.gru = tf.keras.layers.GRU(
            units=self.units,
            return_sequences=True,      # return the full output sequence
            return_state=True,          # return the last hidden state
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden state to zeros.

        Returns:
            tf.Tensor: A tensor of shape (batch, units)
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Performs a forward pass through the encoder.

        Args:
            x (tf.Tensor): Input tensor of shape (batch,
            initial (tf.Tensor): Initial hidden state tensor

        Returns:
            outputs (tf.Tensor): Output tensor of shape (batch,)
            hidden (tf.Tensor): Final hidden state of shape (batch, units).
        """
        # Convert word indices to embeddings
        x = self.embedding(x)

        # Pass the embedded input and initial state to the GRU
        outputs, hidden = self.gru(x, initial_state=initial)

        # Return the full sequence and the last hidden state
        return outputs, hidden
