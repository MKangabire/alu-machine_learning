#!/usr/bin/env python3
"""Attention in RNN"""
import tensorflow as tf
import numpy as np

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder consisting of multiple EncoderBlocks
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor

        Args:
            N (int): Number of encoder blocks
            dm (int): Dimensionality of the model
            h (int): Number of attention heads
            hidden (int): Number of hidden units in the feedforward layer
            input_vocab (int): Size of the input vocabulary
            max_seq_len (int): Maximum sequence length
            drop_rate (float): Dropout rate
        """
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Applies the Transformer encoder to the input sequence

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len)
            training (bool): Boolean to determine if the model is training
            mask (tf.Tensor): Optional mask tensor

        Returns:
            tf.Tensor: Output of shape (batch, input_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        # Add embedding and positional encoding
        x = self.embedding(x)  # (batch, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embedding
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x
