"""Copy small image samples so retraining and evaluation work without data/train.

data/eval   : 20 images per class from the test split  (evaluation of old vs new model)
data/replay : 20 images per class from the train split (replayed during retraining)
Each folder gets a labels.csv (filename,label).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import DATA_DIR, TRAIN_DIR, load_splits

PER_CLASS = 20


def make_samples(per_class=PER_CLASS):
    splits = load_splits()
    for split, folder in (("test", "eval"), ("train", "replay")):
        out = DATA_DIR / folder
        out.mkdir(exist_ok=True)
        df = splits[splits.split == split]
        sample = df.groupby("label", group_keys=False).sample(per_class, random_state=42)
        for f in sample["filename"]:
            shutil.copy2(TRAIN_DIR / f, out / f)
        sample[["filename", "label"]].to_csv(out / "labels.csv", index=False)
        print(f"{folder}: {len(sample)} images")


if __name__ == "__main__":
    make_samples()
