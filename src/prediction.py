"""Inference with the TFLite model. No TensorFlow import here (keeps the API within 512 MB)."""
import io
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # e.g. Windows, where ai-edge-litert has no wheel
    from tensorflow.lite.python.interpreter import Interpreter

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
IMG_SIZE = (224, 224)

_lock = threading.Lock()
_cache = {}


def _disk_version():
    """Model version in metadata.json (changes when any replica promotes a new model)."""
    try:
        return json.loads((MODELS_DIR / "metadata.json").read_text()).get("version", 1)
    except (OSError, ValueError):
        return None


def _load():
    interp = Interpreter(model_path=str(MODELS_DIR / "har_model.tflite"), num_threads=1)
    interp.allocate_tensors()
    classes = json.loads((MODELS_DIR / "class_names.json").read_text())
    return interp, classes


def get_model():
    """Load the tflite interpreter once; reload if metadata.json shows a new version."""
    with _lock:
        version = _disk_version()
        if "model" not in _cache or _cache.get("version") != version:
            _cache["model"], _cache["classes"] = _load()
            _cache["version"] = version
        return _cache["model"], _cache["classes"]


def get_classes():
    return get_model()[1]


def reload_model():
    with _lock:
        _cache.clear()
    return get_model()


def preprocess(src):
    img = Image.open(src).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 127.5 - 1.0)[None]


def predict(image, top_k=3):
    """image: file path or raw bytes. Returns label, confidence and top-k."""
    interp, class_names = get_model()
    src = io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image
    x = preprocess(src)
    with _lock:  # an Interpreter is not thread-safe
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        probs = interp.get_tensor(out["index"])[0]
    top = [{"label": class_names[i], "confidence": float(probs[i])}
           for i in probs.argsort()[::-1][:top_k]]
    return {"label": top[0]["label"], "confidence": top[0]["confidence"], "top3": top}
