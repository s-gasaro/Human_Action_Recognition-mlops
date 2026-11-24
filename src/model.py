import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

from preprocessing import IMG_SIZE, TRAIN_DIR, make_dataset

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "har_model.h5"


def build_model(num_classes, dropout=0.3, lr=1e-3):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze(model, n_layers=30, lr=1e-5):
    """Unfreeze the top n_layers of the base model and recompile with a low LR."""
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])


def _callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]


def train(model, train_ds, val_ds, epochs=15):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=_callbacks(), verbose=2)


def evaluate(model, ds):
    """Accuracy, loss, macro precision/recall/F1."""
    loss, acc = model.evaluate(ds, verbose=0)
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in ds])
    y_pred = model.predict(ds, verbose=0).argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "loss": float(loss),
            "precision": float(p), "recall": float(r), "f1": float(f1)}


def save_model(model, class_names, metrics=None, models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    model.save(models_dir / "har_model.h5")
    (models_dir / "class_names.json").write_text(json.dumps(list(class_names)))
    meta = {"saved_at": datetime.now().isoformat(timespec="seconds"),
            "input_size": list(IMG_SIZE), "num_classes": len(class_names),
            "metrics": metrics or {}}
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_model(models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    model = tf.keras.models.load_model(models_dir / "har_model.h5")
    class_names = json.loads((models_dir / "class_names.json").read_text())
    return model, class_names


def retrain(model_path, new_data, image_dir=TRAIN_DIR, val_df=None, epochs=5, lr=1e-5):
    """Fine-tune the saved model on new_data (DataFrame: filename,label); saves in place."""
    model_path = Path(model_path)
    models_dir = model_path.parent
    model, class_names = load_model(models_dir)

    unfreeze(model, 30, lr)

    train_ds = make_dataset(new_data, class_names, image_dir, training=True)
    if val_df is not None:
        val_ds = make_dataset(val_df, class_names, image_dir)
        train(model, train_ds, val_ds, epochs)
        metrics = evaluate(model, val_ds)
    else:
        model.fit(train_ds, epochs=epochs)
        metrics = None
    save_model(model, class_names, metrics, models_dir)
    return metrics
