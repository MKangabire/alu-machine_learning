#!/usr/bin/env python3
"""Attention in RNN"""

import tensorflow as tf


SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder class for machine translation using GRU and attention mechanism.

    Attributes:
        embedding (tf.keras.layers.Embedding): Embedding layer to convert input tokens into dense vectors.
        gru (tf.keras.layers.GRU): GRU layer to process the embedded input and context.
        F (tf.keras.layers.Dense): Fully connected layer to project the GRU output to vocabulary size.
        attention (SelfAttention): SelfAttention layer to compute context vectors.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vectors.
            units (int): Number of hidden units in the GRU.
            batch (int): Batch size.
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Performs a forward pass through the decoder.

        Args:
            x (tf.Tensor): Tensor of shape (batch, 1) containing the previous word as an index.
            s_prev (tf.Tensor): Tensor of shape (batch, units) with the previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape (batch, input_seq_len, units) with encoder outputs.

        Returns:
            y (tf.Tensor): Tensor of shape (batch, vocab) with the output word prediction.
            s (tf.Tensor): Tensor of shape (batch, units) with the new decoder hidden state.
        """
        # Apply attention mechanism to get context vector and attention weights
        context, _ = self.attention(s_prev, hidden_states)  # context: (batch, units)

        # Embed the input word (x)
        x = self.embedding(x)  # x: (batch, 1, embedding)

        # Concatenate context with the embedded input word
        context = tf.expand_dims(context, 1)  # (batch, 1, units)
        x_combined = tf.concat([context, x], axis=-1)  # (batch, 1, units + embedding)

        # Pass through GRU
        output, state = self.gru(x_combined)  # output: (batch, 1, units), state: (batch, units)

        # Remove the sequence dimension and pass through dense layer to get predictions
        y = self.F(tf.squeeze(output, axis=1))  # y: (batch, vocab)

        return y, state
