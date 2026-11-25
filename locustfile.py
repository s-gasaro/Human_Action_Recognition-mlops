"""Load test: python -m locust -f locustfile.py --host http://localhost:8000"""
import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGES = [p for p in Path(__file__).parent.joinpath("data", "eval").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
DATA = [(p.name, p.read_bytes()) for p in random.sample(IMAGES, min(50, len(IMAGES)))]  # cache in memory


class HarUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(10)
    def predict(self):
        name, data = random.choice(DATA)
        self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
