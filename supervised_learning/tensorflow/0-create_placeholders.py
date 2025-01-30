#!/usr/bin/env python3
"""tensoflow"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """a function that returns 2 placeholders"""
    x = tf.placeholder(dtype=tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, classes), name='y')
    return x, y
