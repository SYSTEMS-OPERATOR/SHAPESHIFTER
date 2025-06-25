import os
import sys

# Ensure the repo root is in sys.path before importing the module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
from shapeshifter import TransversalWrapLayer


def test_wrap_layer_basic():
    data = tf.reshape(tf.range(9, dtype=tf.float32), (1, 3, 3, 1))
    layer = TransversalWrapLayer()
    out = layer(data)
    assert out.shape == (1, 5, 5, 1)
    out_np = out.numpy().squeeze()
    data_np = data.numpy().squeeze()
    assert out_np[0, 0] == data_np[-1, -1]
    assert out_np[0, -1] == data_np[-1, 0]
    assert out_np[-1, 0] == data_np[0, -1]
    assert out_np[-1, -1] == data_np[0, 0]
