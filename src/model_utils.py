import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
