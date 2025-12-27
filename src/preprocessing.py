# preprocessing.py
import numpy as np

def preprocess_image(image, width=224, height=224, normalization="0-1"):
    """
    Resize and normalize a PIL image.

    normalization:
      - "model"    : passthrough (no external scaling) — use for the Ensemble .keras
      - "0-1"      : divide by 255.0
      - "mean-std" : per-image standardization
    """
    try:
        image = image.resize((width, height))
        arr = np.array(image).astype(np.float32)

        if normalization == "0-1":
            arr /= 255.0
        elif normalization == "mean-std":
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        elif normalization == "model":
            pass  # model handles preprocessing internally
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        image_array = np.expand_dims(arr, axis=0)
        return image_array, image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None


