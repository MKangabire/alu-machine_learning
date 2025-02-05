#!/usr/bin/env python3
"""Create a layer L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a TensorFlow 1.12 layer with L2 regularization.
    
    Args:
        prev: tensor, output of the previous layer
        n: int, number of nodes in the new layer
        activation: activation function to apply
        lambtha: float, L2 regularization parameter
    
    Returns:
        Tensor, output of the new layer
    """
    l2_reg = tf.contrib.layers.l2_regularizer(scale=lambtha)

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_reg
    )

    return layer(prev)