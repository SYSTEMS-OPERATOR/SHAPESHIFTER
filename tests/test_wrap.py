import numpy as np
import tensorflow as tf
from shapeshifter import TransversalWrapLayer


def test_wrap_layer_shape():
    height, width = 3, 4
    inputs = tf.zeros((1, height, width, 1))
    wrap = TransversalWrapLayer()
    outputs = wrap(inputs)
    assert outputs.shape == (1, height + 2, width + 2, 1)
