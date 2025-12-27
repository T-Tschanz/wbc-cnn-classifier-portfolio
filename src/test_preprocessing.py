import numpy as np
import pytest
from preprocessing import preprocess_image

def dummy_image(shape=(480,480,3)):
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)

def test_preprocess_output_shape():
    img = dummy_image()
    tensor = preprocess_image(img)
    # Should resize + batch‐dim
    assert tensor.shape == (1, 360, 363, 3)

def test_preprocess_maintains_dtype():
    img = dummy_image()
    tensor = preprocess_image(img)
    # After normalization, dtype is float32
    assert tensor.dtype == np.float32

def test_preprocess_bad_input():
    bad = "not an image"
    with pytest.raises(TypeError):
        preprocess_image(bad)
