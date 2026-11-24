"""Compute dataset insights once and save them to models/insights.json (read by GET /insights)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image

from preprocessing import TRAIN_DIR, load_splits

OUT = Path(__file__).resolve().parent.parent / "models" / "insights.json"
SIZE_SAMPLE = 1000
BRIGHT_PER_CLASS = 20


def _per_class_f1(splits, class_names):
    """Per-class F1 of the current model on the test split."""
    import model as hm
    from preprocessing import make_dataset
    from sklearn.metrics import f1_score

    model, names = hm.load_model()
    test = splits[splits.split == "test"]
    ds = make_dataset(test, names, TRAIN_DIR)
    pred = np.argmax(model.predict(ds, verbose=0), axis=1)
    true = [names.index(l) for l in test["label"]]
    f1 = f1_score(true, pred, average=None, labels=range(len(names)))
    return {n: round(float(v), 4) for n, v in zip(names, f1)}


def compute():
    splits = load_splits()
    train = splits[splits.split == "train"]
    class_names = sorted(splits["label"].unique())

    dist = splits.groupby(["label", "split"]).size().unstack(fill_value=0)
    class_distribution = {c: {s: int(dist.loc[c, s]) for s in dist.columns} for c in class_names}

    sample = train.sample(min(SIZE_SAMPLE, len(train)), random_state=42)
    widths, heights = [], []
    for f in sample["filename"]:
        with Image.open(TRAIN_DIR / f) as im:
            widths.append(im.width)
            heights.append(im.height)
    widths, heights = np.array(widths), np.array(heights)
    ratio = widths / heights

    brightness = {}
    for c in class_names:
        files = train[train.label == c].sample(BRIGHT_PER_CLASS, random_state=42)["filename"]
        brightness[c] = [round(float(np.asarray(Image.open(TRAIN_DIR / f).convert("L")).mean()), 1) for f in files]

    return {
        "class_distribution": class_distribution,
        "image_size": {"width": widths.tolist(), "height": heights.tolist(),
                       "aspect_ratio": [round(float(r), 3) for r in ratio],
                       "median_width": float(np.median(widths)), "median_height": float(np.median(heights)),
                       "median_aspect_ratio": round(float(np.median(ratio)), 2),
                       "landscape_share": round(float((ratio > 1).mean()), 3)},
        "brightness": brightness,
        "per_class_f1": _per_class_f1(splits, class_names),
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(compute()))
    print(f"saved {OUT}")
