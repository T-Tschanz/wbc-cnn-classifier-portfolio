import numpy as np
from predictor import predict, compute_saliency

def test_predict_confidence_bounds():
    dummy = np.zeros((1, 360, 363, 3), dtype=np.float32)
    label, conf = predict(dummy, model_name="mobilenet_head")
    assert 0.0 <= conf <= 1.0
    assert isinstance(label, str)

def test_saliency_shape():
    dummy = np.zeros((1, 360, 363, 3), dtype=np.float32)
    sal = compute_saliency(dummy, model_name="mobilenet_head")
    # Should match image spatial dims
    assert sal.shape == (360, 363)