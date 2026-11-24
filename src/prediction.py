import io
import threading

import numpy as np
import tensorflow as tf
from PIL import Image

from model import MODELS_DIR, load_model
from preprocessing import IMG_SIZE, preprocess_image

_lock = threading.Lock()
_cache = {}


def get_model():
    """Load the saved model once and reuse it."""
    with _lock:
        if "model" not in _cache:
            _cache["model"], _cache["classes"] = load_model(MODELS_DIR)
        return _cache["model"], _cache["classes"]


def reload_model():
    """Drop the cached model and load the one on disk."""
    with _lock:
        _cache.clear()
    return get_model()


def predict(image, model=None, class_names=None, top_k=3):
    """image: file path or raw bytes. Returns label, confidence and top-k."""
    if model is None:
        model, class_names = get_model()
    src = io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image
    img = tf.image.resize(np.asarray(Image.open(src).convert("RGB")), IMG_SIZE)
    probs = model(preprocess_image(img)[None], training=False).numpy()[0]
    top = [{"label": class_names[i], "confidence": float(probs[i])}
           for i in probs.argsort()[::-1][:top_k]]
    return {"label": top[0]["label"], "confidence": top[0]["confidence"], "top3": top}
