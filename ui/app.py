"""Streamlit UI for the Human Action Recognition MLOps service."""
import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CLASSES = ["calling", "clapping", "cycling", "dancing", "drinking", "eating", "fighting", "hugging",
           "laughing", "listening_to_music", "running", "sitting", "sleeping", "texting", "using_laptop"]

st.set_page_config(page_title="HAR MLOps", page_icon="🏃", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
[data-testid="stMetric"] {background:#F3F4F8; padding:14px 18px; border-radius:12px;}
h1 {font-weight:700;}
</style>""", unsafe_allow_html=True)


class ApiDown(Exception):
    pass


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=kw.pop("timeout", 30), **kw)
    except requests.RequestException:
        raise ApiDown
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


def guard(fn):
    """Run a page, showing a friendly message if the API is unreachable."""
    try:
        fn()
    except ApiDown:
        st.error(f"Cannot reach the API at {API_URL}. Start it with "
                 "`uvicorn src.api:app --port 8000` or set the API_URL environment variable.")
    except RuntimeError as e:
        st.error(str(e))


def fmt_uptime(s):
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60}m {s % 60}s"


def dashboard():
    st.title("Dashboard")
    health, metrics = api("GET", "/health"), api("GET", "/metrics")
    c = st.columns(4)
    c[0].metric("API status", "Online" if health["status"] == "ok" else "No model")
    c[1].metric("Uptime", fmt_uptime(health["uptime_seconds"]))
    c[2].metric("Model version", f"v{health['model_version']}" if health["model_version"] else "-")
    c[3].metric("Predictions served", metrics["predictions_served"])

    st.subheader("Current test metrics")
    m = metrics.get("model_metrics") or {}
    if m:
        c = st.columns(4)
        c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        c[1].metric("Macro F1", f"{m['f1']:.3f}")
        c[2].metric("Precision", f"{m['precision']:.3f}")
        c[3].metric("Recall", f"{m['recall']:.3f}")
    else:
        st.info("No metrics saved with the current model.")

    st.subheader("Retrain history")
    runs = api("GET", "/retrain/history")
    if runs:
        cols = ["id", "status", "started_at", "finished_at", "n_uploads", "old_f1", "new_f1",
                "promoted", "model_version", "message"]
        st.dataframe(pd.DataFrame(runs)[cols], use_container_width=True, hide_index=True)
    else:
        st.caption("No retraining runs yet.")


def insights():
    st.title("Insights")
    d = api("GET", "/insights")

    st.subheader("Class distribution")
    rows = [{"class": c, "split": s, "images": n} for c, v in d["class_distribution"].items() for s, n in v.items()]
    st.plotly_chart(px.bar(pd.DataFrame(rows), x="class", y="images", color="split", barmode="stack"),
                    use_container_width=True)
    st.caption("The dataset is perfectly balanced: 840 images per class, split evenly across train/val/test. "
               "Accuracy is therefore not distorted by class imbalance and no class weights are needed.")

    st.subheader("Image size and aspect ratio")
    sz = d["image_size"]
    c = st.columns(2)
    c[0].plotly_chart(px.scatter(pd.DataFrame({"width": sz["width"], "height": sz["height"]}),
                                 x="width", y="height", opacity=0.4), use_container_width=True)
    c[1].plotly_chart(px.histogram(pd.DataFrame({"aspect ratio": sz["aspect_ratio"]}), x="aspect ratio", nbins=40),
                      use_container_width=True)
    st.caption(f"Images are small (median {sz['median_width']:.0f}x{sz['median_height']:.0f}) and "
               f"{sz['landscape_share']:.0%} are landscape (median ratio {sz['median_aspect_ratio']}). "
               "Resizing to a 224x224 square squeezes the width, which may hide fine details.")

    st.subheader("Brightness per class")
    rows = [{"class": c, "brightness": b} for c, v in d["brightness"].items() for b in v]
    st.plotly_chart(px.box(pd.DataFrame(rows), x="class", y="brightness"), use_container_width=True)
    st.caption("Class medians differ, but the boxes overlap heavily: brightness varies more within a class "
               "than between classes, so it is unlikely to be a shortcut the model relies on.")

    st.subheader("Per-class F1 of the current model")
    f1 = pd.DataFrame({"class": list(d["per_class_f1"]), "f1": list(d["per_class_f1"].values())}).sort_values("f1")
    st.plotly_chart(px.bar(f1, x="f1", y="class", orientation="h", range_x=[0, 1]), use_container_width=True)
    st.caption("The lowest-F1 classes are the ones that look alike in a still image "
               "(hand-near-face poses such as texting, calling, drinking). They are the best candidates for new uploads.")


def predict():
    st.title("Predict")
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if not f:
        return
    c = st.columns(2)
    c[0].image(f, use_container_width=True)
    with st.spinner("Predicting..."):
        res = api("POST", "/predict", files={"file": (f.name, f.getvalue(), f.type)})
    with c[1]:
        st.subheader(res["label"].replace("_", " ").title())
        for p in res["top3"]:
            st.write(p["label"].replace("_", " "))
            st.progress(min(float(p["confidence"]), 1.0), text=f"{p['confidence']:.1%}")


def upload_retrain():
    st.title("Upload & Retrain")
    health = api("GET", "/health")

    st.subheader("1. Upload labelled images")
    label = st.selectbox("Label", CLASSES)
    files = st.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Upload", disabled=not files):
        res = api("POST", "/upload", data={"label": label},
                  files=[("files", (x.name, x.getvalue(), x.type)) for x in files], timeout=120)
        st.success(f"Saved {res['saved']} image(s) as '{res['label']}'.")
    st.caption(f"Uploads so far: {api('GET', '/metrics')['uploads']}")

    st.subheader("2. Retrain")
    st.caption(f"Current model: v{health['model_version']}. "
               "The new model is promoted only if its test macro F1 is not lower.")
    if st.button("Retrain", type="primary"):
        api("POST", "/retrain")
    if api("GET", "/retrain/status").get("status") == "running":
        box = st.empty()
        while True:
            run = api("GET", "/retrain/status")
            if run.get("status") != "running":
                break
            box.info(f"Retraining in progress: epoch {run.get('epoch') or 0}/{run.get('epochs') or 3}")
            time.sleep(3)
        box.empty()
    run = api("GET", "/retrain/status")
    if run.get("status") == "done":
        c = st.columns(3)
        c[0].metric("Old F1", f"{run['old_f1']:.3f}")
        c[1].metric("New F1", f"{run['new_f1']:.3f}", f"{run['new_f1'] - run['old_f1']:+.3f}")
        c[2].metric("Promoted", "Yes" if run["promoted"] else "No")
        (st.success if run["promoted"] else st.warning)(run["message"])
    elif run.get("status") == "failed":
        st.error(f"Retraining failed: {run.get('message')}")


PAGES = {"Dashboard": dashboard, "Insights": insights, "Predict": predict, "Upload & Retrain": upload_retrain}
st.sidebar.title("🏃 HAR MLOps")
choice = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
st.sidebar.caption(f"API: {API_URL}")
guard(PAGES[choice])
