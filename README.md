# Human Action Recognition – MLOps

Human activity recognition from still images, with 15 activity classes (calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop). It extends my earlier wearable-sensor HAR project to computer vision and wraps the model in a full MLOps pipeline: REST API, Streamlit UI, retraining on uploaded images, Docker with load balancing, and cloud deployment.

**Use case:** activity monitoring for care (e.g. spotting a person sleeping or fighting), safety and fitness.

## Links

- **Video demo:** [Watch on YouTube](https://youtu.be/qcz05JhDoUA)
- **Live UI:** https://har-ui.onrender.com
- **API docs:** https://har-api-8g57.onrender.com/docs

> The free Render tier sleeps when idle, so the first load takes about 1 minute. Retraining is disabled in the cloud (512 MB RAM, 0.1 CPU is too small for training); it runs in the Docker deployment.

## Dataset

[Human Action Recognition (HAR) Dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) from Kaggle (meetnagadia): 12,600 labelled images, 840 per class, split 70/15/15 (train/val/test).

The full images are **not** in this repo. Download them from Kaggle into `data/train/` to retrain from scratch. `data/eval` (held-out evaluation set) and `data/replay` (replay samples mixed into retraining) are small samples committed so the app works out of the box.

## Model results

MobileNetV2 transfer learning (ImageNet weights). The best model is the fine-tuned one (top 50 layers unfrozen, lr 1e-4). Test set results:

| Accuracy | Macro F1 | Precision | Recall |
|---|---|---|---|
| 0.74 | 0.74 | 0.75 | 0.74 |

Weakest classes: **texting, sitting, listening_to_music**. They look alike in a still image (hand-near-face poses). Details are in [notebook/har_mlops.ipynb](notebook/har_mlops.ipynb).

For serving, the model is converted to TFLite (`scripts/convert_tflite.py`) so the API does not load TensorFlow.

## Setup

### Local

```bash
conda create -n har python=3.11 -y
conda activate har
pip install -r requirements.txt
pip install -r ui/requirements.txt

uvicorn src.api:app --port 8000          # API  -> http://localhost:8000/docs
streamlit run ui/app.py                  # UI   -> http://localhost:8501
```

The UI reads the API address from the `API_URL` environment variable (default `http://localhost:8000`). Local retraining needs TensorFlow (`pip install tensorflow-cpu==2.21.0`).

### Docker (API replicas behind nginx + UI)

```bash
docker compose up --build --scale api=3
```

- UI: http://localhost:8501
- API (nginx load balancer): http://localhost:8000/docs

### Retrain from the UI

1. Open **Upload & Retrain**, pick a label, upload labelled images and click **Upload**.
2. Click **Retrain** (Docker/local only). The model is fine-tuned on the uploads plus replay samples.
3. The new model is promoted only if its macro F1 on `data/eval` is not lower than the current one. Results appear in the Dashboard's retrain history.

## Flood request simulation

Locust, 50 users (spawn 10/s), 1 minute, against the local Docker stack with nginx (`scripts/run_load_tests.ps1`). Mix: `POST /predict` (most weight) and `GET /health`.

| API replicas | Requests | Failures | Avg (ms) | p50 (ms) | p95 (ms) | RPS |
|---|---|---|---|---|---|---|
| 1 | 137 | 0 | 16565 | 17000 | 26000 | 2.3 |
| 2 | 218 | 0 | 9659 | 7000 | 24000 | 3.7 |
| 3 | 188 | 0 | 11188 | 4000 | 31000 | 3.1 |

Two replicas give the best throughput and average latency. Three replicas improve the median but the tail gets worse, because all containers share a single laptop CPU, so extra replicas compete for the same cores instead of adding capacity.

## Project structure

```
├── src/
│   ├── api.py              FastAPI app (predict, upload, retrain, metrics, health)
│   ├── prediction.py       TFLite inference (no TensorFlow import)
│   ├── model.py            Model building, training, evaluation
│   ├── preprocessing.py    Splits and tf.data pipeline
│   ├── tflite_convert.py   .h5 -> .tflite conversion
│   ├── database.py         SQLite (uploads, retrain runs)
│   ├── insights.py         Dataset/model insights for the UI
│   └── make_samples.py     Builds data/eval and data/replay samples
├── ui/                     Streamlit app + Dockerfile
├── scripts/                convert_tflite.py, run_load_tests.ps1, summarize_results.py
├── models/                 Trained model (.h5, .tflite), class names, metadata
├── data/                   Split CSVs, eval and replay samples
├── notebook/               Training and evaluation notebook
├── results/                Locust CSVs and summary
├── locustfile.py           Load test
├── Dockerfile              API image
├── docker-compose.yml      API replicas + nginx + UI
├── nginx.conf              Load balancer
└── render.yaml             Render deployment blueprint
```

## Limitations

- The model knows only 15 classes. An image of something else (e.g. a person crying) is forced into the nearest class, often with high confidence.
- The free cloud tier (512 MB RAM, 0.1 CPU, sleeps when idle) is slow and cannot retrain.
