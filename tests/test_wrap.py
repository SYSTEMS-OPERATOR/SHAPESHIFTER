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
