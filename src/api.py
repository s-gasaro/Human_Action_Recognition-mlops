import io
import os
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `uvicorn src.api:app`

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

import database as db
import prediction

# NOTE: TensorFlow (model.py / preprocessing.py) is imported lazily inside the retrain job only,
# so the API process serving /predict stays small enough for a 512 MB host.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = prediction.MODELS_DIR
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", ROOT / "uploads"))
MODEL_FILES = ("har_model.h5", "class_names.json", "metadata.json")
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
MAX_BYTES = 10 * 1024 * 1024
RETRAIN_EPOCHS = 3
RETRAIN_BATCH = 8
EVAL_DIR = DATA_DIR / "eval"
REPLAY_DIR = DATA_DIR / "replay"

app = FastAPI(title="Human Action Recognition API")
START = time.time()
stats = {"predictions": 0, "total_latency": 0.0}


@app.on_event("startup")
def startup():
    db.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db.init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _metadata():
    path = MODELS_DIR / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _model_ready():
    return (MODELS_DIR / "har_model.tflite").exists()


async def _read_image(file: UploadFile):
    """Read and validate an uploaded image; returns bytes."""
    name = file.filename or ""
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, f"{name}: only .jpg, .jpeg and .png files are accepted.")
    data = await file.read()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(400, f"{name}: file is empty or larger than 10 MB.")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, f"{name} is not a valid image.")
    return data


@app.get("/health")
def health():
    return {"status": "ok" if _model_ready() else "no_model",
            "model_version": _metadata().get("version", 1) if _model_ready() else None,
            "uptime_seconds": round(time.time() - START, 1)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _model_ready():
        raise HTTPException(503, "No trained model found in models/. Train and save a model first.")
    data = await _read_image(file)
    t = time.time()
    result = prediction.predict(data)
    stats["predictions"] += 1
    stats["total_latency"] += time.time() - t
    return result


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), label: str = Form(...)):
    if not _model_ready():
        raise HTTPException(503, "No trained model found, so valid labels are unknown.")
    class_names = prediction.get_classes()
    if label not in class_names:
        raise HTTPException(400, f"Unknown label {label}. Valid labels: {class_names}")
    images = [(f.filename, await _read_image(f)) for f in files]  # validate all before saving
    folder = UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in images:
        path = folder / f"{uuid.uuid4().hex[:8]}_{Path(name).name}"
        path.write_bytes(data)
        db.add_upload(path.name, label, path)
    return {"saved": len(images), "label": label}


def _retrain_job(run_id):
    try:
        import tensorflow as tf
        import model as hm
        from preprocessing import make_dataset
        from tflite_convert import convert
        tf.config.threading.set_intra_op_parallelism_threads(1)  # keep memory low (512 MB host)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        uploads = db.get_unused_uploads()
        new = pd.DataFrame({"filename": [u["path"] for u in uploads], "label": [u["label"] for u in uploads]})
        replay = pd.read_csv(REPLAY_DIR / "labels.csv")
        replay["filename"] = [str(REPLAY_DIR / f) for f in replay["filename"]]
        data = pd.concat([new, replay], ignore_index=True)
        eval_df = pd.read_csv(EVAL_DIR / "labels.csv")

        old_model, class_names = hm.load_model(MODELS_DIR)
        eval_ds = make_dataset(eval_df, class_names, EVAL_DIR, batch_size=RETRAIN_BATCH)
        old_f1 = hm.evaluate(old_model, eval_ds)["f1"]

        old_version = _metadata().get("version", 1)
        version = old_version + 1
        cand = MODELS_DIR / "versions" / f"v{version}"
        cand.mkdir(parents=True, exist_ok=True)
        for f in MODEL_FILES:
            if (MODELS_DIR / f).exists():
                shutil.copy2(MODELS_DIR / f, cand / f)
        db.update_run(run_id, epoch=0, epochs=RETRAIN_EPOCHS)
        hm.retrain(cand / "har_model.h5", data, epochs=RETRAIN_EPOCHS, batch_size=RETRAIN_BATCH,
                   on_epoch=lambda e, n: db.update_run(run_id, epoch=e, epochs=n))

        new_model, _ = hm.load_model(cand)
        metrics = hm.evaluate(new_model, eval_ds)
        promote = metrics["f1"] >= old_f1
        meta = json.loads((cand / "metadata.json").read_text())
        meta.update({"version": version, "metrics": metrics, "promoted": promote})
        (cand / "metadata.json").write_text(json.dumps(meta, indent=2))
        if promote:
            for f in MODEL_FILES:
                if f != "metadata.json":
                    shutil.copy2(cand / f, MODELS_DIR / f)
            convert(MODELS_DIR / "har_model.h5", MODELS_DIR / "har_model.tflite")
            shutil.copy2(cand / "metadata.json", MODELS_DIR / "metadata.json")  # last: replicas reload on version change
            prediction.reload_model()
            del old_model, new_model
            tf.keras.backend.clear_session()
        db.mark_used([u["id"] for u in uploads])
        db.update_run(run_id, status="done", finished_at=db.now(), old_f1=old_f1, new_f1=metrics["f1"],
                      promoted=int(promote), model_version=version if promote else old_version,
                      message="promoted" if promote else "new model had lower macro F1; kept old model")
    except Exception as e:
        db.update_run(run_id, status="failed", finished_at=db.now(), message=str(e))


@app.post("/retrain")
def retrain(background: BackgroundTasks):
    if not _model_ready():
        raise HTTPException(503, "No trained model found in models/ to fine-tune.")
    if db.is_retraining():
        raise HTTPException(409, "A retraining run is already in progress.")
    n = len(db.get_unused_uploads())
    if n == 0:
        raise HTTPException(400, "No unused uploads. Upload labelled images with POST /upload first.")
    run_id = db.create_run(n)
    background.add_task(_retrain_job, run_id)
    return {"run_id": run_id, "status": "running", "n_uploads": n}


@app.get("/retrain/status")
def retrain_status():
    return db.latest_run() or {"status": "never_run"}


@app.get("/insights")
def insights():
    path = MODELS_DIR / "insights.json"
    if not path.exists():
        raise HTTPException(404, "models/insights.json not found. Run `python src/insights.py` first.")
    return json.loads(path.read_text())


@app.get("/retrain/history")
def retrain_history():
    return db.list_runs()


@app.get("/metrics")
def metrics():
    n = stats["predictions"]
    meta = _metadata()
    return {"model_version": meta.get("version", 1) if meta else None,
            "model_metrics": meta.get("metrics", {}),
            "predictions_served": n,
            "avg_latency_seconds": round(stats["total_latency"] / n, 4) if n else None,
            "uploads": db.upload_counts(),
            "retrain_runs": db.run_count()}
