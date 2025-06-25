import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("gradio")
pytest.importorskip("psutil")

from shapeshifter import TransversalWrapLayer
import tensorflow as tf


def test_transversal_wrap_layer():
    data = tf.reshape(tf.range(4, dtype=tf.float32), (1, 2, 2, 1))
    layer = TransversalWrapLayer()
    out = layer(data).numpy().squeeze()
    expected = np.array([[3, 2, 3, 2],
                         [1, 0, 1, 0],
                         [3, 2, 3, 2],
                         [1, 0, 1, 0]], dtype=np.float32)
    assert np.array_equal(out, expected)


def test_wrap_layer_shape():
    height, width = 3, 4
    inputs = tf.zeros((1, height, width, 1))
    wrap = TransversalWrapLayer()
    outputs = wrap(inputs)
    assert outputs.shape == (1, height + 2, width + 2, 1)

