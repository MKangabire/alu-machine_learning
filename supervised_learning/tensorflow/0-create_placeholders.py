#!/usr/bin/env python3
"""tensoflow"""

import tensorflow as tf
# tf.disable_v2_behavior()

def create_placeholders(nx, classes):
    """a function that returns 2 placeholders"""
    x = tf.placeholders(dtype=tf.float32, shape=(None, nx), name='x')
    y = tf.placeholders(dtype=tf.float32, shape=(None, classes), name='y')
    return x, y