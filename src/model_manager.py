# model_manager.py
import os, glob, time, tempfile, shutil
from functools import lru_cache
from typing import Dict, Tuple, Optional
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")


# ---- Built-in models (pick the newest file that matches each pattern)
MODEL_PATTERNS: Dict[str, str] = {
    "Ensemble (Soft Voting)"     : os.path.join(MODELS_DIR, "wbc_ensemble*.keras"),
    "MobileNetV2 (Transfer)"    : os.path.join(MODELS_DIR, "mobilenet_v2_final*.keras"),
    "Custom CNN (AdamW)"        : os.path.join(MODELS_DIR, "custom_adamw_final*.keras"),
}


# Preprocessing modes we’ll use in preprocessing.py
#   "model"     -> model already contains Resizing+Rescaling (our Ensemble .keras)
#   "0-1"       -> expects values in [0,1] (our individual models)
#   "mobilenet" -> if a model was trained with mobilenet_v2.preprocess_input
BUILTIN_PREPROCESS: Dict[str, str] = {
    "Ensemble (Soft Voting)"  : "model",   # preprocessing embedded in model
    "MobileNetV2 (Transfer)" : "0-1",
    "Custom CNN (AdamW)"     : "0-1",
}


CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]


def _resolve_latest_path(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


@lru_cache(maxsize=8)
def _load_model_from_path(path: str):
    # compile=False so we never need custom objects (AdamW, focal loss) at app load.
    return keras.models.load_model(path, compile=False)


def _detect_preprocessing_mode(model) -> str:
    """
    Best-effort guess at preprocessing inside the model.
    - If we see a Rescaling layer near the top: return "model"
    - If we see a Lambda with mobilenet_v2: return "mobilenet"
    - Otherwise default to "0-1"
    """
    try:
        layers = model.layers
        first = layers[0]
        # For Functional, the first "real" layers follow the InputLayer
        scan = layers[:8]  # quick scan
        for lyr in scan:
            name = getattr(lyr, "name", "").lower()
            cls  = lyr.__class__.__name__.lower()
            if "rescaling" in cls or "resizing" in cls:
                return "model"
            if cls == "lambda" and "mobilenet" in name:
                return "mobilenet"
        return "0-1"
    except Exception:
        return "0-1"


class ModelManager:
    """
    Centralized model registry and loader for the WBC classifier app.

    Responsibilities:
    - Discover available production models on disk
    - Load and cache models safely for Streamlit reruns
    - Expose consistent interfaces for inference and preprocessing
    - Support optional user-uploaded models for experimentation

    This class is intentionally lightweight and framework-agnostic
    to keep deployment simple and robust.
    """

    def __init__(self):
        # Build available built-ins dynamically from disk
        self.available_models: Dict[str, str] = {}
        for name, pattern in MODEL_PATTERNS.items():
            p = _resolve_latest_path(pattern)
            if p:
                self.available_models[name] = p

        # Custom upload (filled when user uploads)
        self.custom_name: Optional[str] = None
        self.custom_path: Optional[str] = None
        self.custom_mode: Optional[str] = None

    # ---------- Query ----------
    def list_model_names(self):
        names = list(self.available_models.keys())
        if self.custom_name:
            names.append(self.custom_name)  # e.g., "Uploaded: mymodel.keras"
        return names

    def get_classes(self):
        return CLASSES

    # ---------- Loading ----------
    def get_model_and_mode(self, name: str):
        """Return (model, preprocessing_mode) for the selected name."""
        if name == self.custom_name and self.custom_path:
            model = _load_model_from_path(self.custom_path)
            return model, (self.custom_mode or "0-1")

        # built-in
        path = self.available_models.get(name)
        if not path:
            raise FileNotFoundError(f"Model '{name}' not found on disk.")
        model = _load_model_from_path(path)
        mode = BUILTIN_PREPROCESS.get(name, "0-1")
        return model, mode

    # ---------- Upload ----------
    def upload_custom_model(self, uploaded_file) -> str:
        """
        Accepts a Streamlit UploadedFile (or file-like). Saves to ./models/uploads/,
        loads it once to detect preprocessing, and exposes it as a selectable model.
        Returns the display name (e.g., 'Uploaded: my_model.keras').
        """
        # Persist to disk so it survives Streamlit reruns
        os.makedirs("./models/uploads", exist_ok=True)
        base = getattr(uploaded_file, "name", f"uploaded_{int(time.time())}.keras")
        save_path = os.path.join("./models/uploads", base)

        # If the user uploads a directory/zip SavedModel, just save the file and try loading.
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load once to detect mode & warm the cache
        model = _load_model_from_path(save_path)
        self.custom_mode = _detect_preprocessing_mode(model)
        self.custom_path = save_path
        self.custom_name = f"Uploaded: {os.path.basename(base)}"

        return self.custom_name

