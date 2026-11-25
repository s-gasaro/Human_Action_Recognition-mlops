"""Keras .h5 -> .tflite conversion. Imports TensorFlow lazily (only used offline / during retraining)."""
import os
from pathlib import Path


def convert(h5_path, tflite_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(h5_path)
    data = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    tflite_path = Path(tflite_path)
    tmp = tflite_path.with_suffix(".tflite.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, tflite_path)  # atomic: readers never see a half-written file
    return tflite_path
