"""Read results/locust_N/locust_stats.csv and write results/summary.md."""
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
rows = ["| Replicas | Requests | Failures | Avg (ms) | p50 (ms) | p95 (ms) | RPS |", "|---|---|---|---|---|---|---|"]
for d in sorted(RESULTS.glob("locust_*"), key=lambda p: int(p.name.split("_")[1])):
    f = d / "locust_stats.csv"
    if not f.exists():
        continue
    with open(f, newline="", encoding="utf-8") as fh:
        agg = next(r for r in csv.DictReader(fh) if r["Name"] == "Aggregated")
    rows.append(f"| {d.name.split('_')[1]} | {agg['Request Count']} | {agg['Failure Count']} | "
                f"{float(agg['Average Response Time']):.0f} | {agg['50%']} | {agg['95%']} | "
                f"{float(agg['Requests/s']):.1f} |")
(RESULTS / "summary.md").write_text("# Load test summary (50 users, spawn 10/s, 1 min)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
print("\n".join(rows))
