#!/usr/bin/env python3
"""Attention in RNN"""
import tensorflow as tf
import numpy as np


positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock

class Decoder(tf.keras.layers.Layer):
    """
    Transformer Decoder consisting of multiple DecoderBlocks
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor

        Args:
            N (int): Number of decoder blocks
            dm (int): Dimensionality of the model
            h (int): Number of attention heads
            hidden (int): Number of hidden units in the feedforward layer
            target_vocab (int): Size of the target vocabulary
            max_seq_len (int): Maximum sequence length
            drop_rate (float): Dropout rate
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Applies the Transformer decoder to the target sequence

        Args:
            x (tf.Tensor): Target input tensor of shape (batch, target_seq_len)
            encoder_output (tf.Tensor): Encoder output tensor of shape (batch, input_seq_len, dm)
            training (bool): Whether the model is training
            look_ahead_mask (tf.Tensor): Look-ahead mask for first MHA
            padding_mask (tf.Tensor): Padding mask for second MHA

        Returns:
            tf.Tensor: Decoder output of shape (batch, target_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch, target_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embeddings
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask, padding_mask)

        return x
