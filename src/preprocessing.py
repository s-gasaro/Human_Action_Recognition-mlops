import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def make_splits(csv_path=DATA_DIR / "Training_set.csv", out_path=DATA_DIR / "splits.csv"):
    """Stratified 70/15/15 split saved to splits.csv."""
    df = pd.read_csv(csv_path)
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=42)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    out.to_csv(out_path, index=False)
    return out


def load_splits(path=DATA_DIR / "splits.csv"):
    return pd.read_csv(path)


def get_class_names(df):
    return sorted(df["label"].unique())


def augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])


def preprocess_image(img):
    """Resize-free MobileNetV2 scaling to [-1, 1]."""
    return tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img, tf.float32))


def make_dataset(df, class_names, image_dir=TRAIN_DIR, training=False, batch_size=BATCH_SIZE):
    """tf.data pipeline from a (filename, label) DataFrame."""
    paths = [str(Path(image_dir) / f) for f in df["filename"]]
    idx = {c: i for i, c in enumerate(class_names)}
    labels = [idx[l] for l in df["label"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        return img, tf.one_hot(label, len(class_names))

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch_size)
    if training:
        aug = augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_image(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


if __name__ == "__main__":
    s = make_splits()
    print(s["split"].value_counts())
