import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "app.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now():
    return datetime.now().isoformat(timespec="seconds")


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label TEXT, path TEXT,
            used INTEGER DEFAULT 0, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS retrain_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT, started_at TEXT, finished_at TEXT,
            n_uploads INTEGER, old_f1 REAL, new_f1 REAL, promoted INTEGER, model_version INTEGER,
            message TEXT)""")
        c.execute("UPDATE retrain_runs SET status='failed', message='interrupted by restart' "
                  "WHERE status='running'")


def add_upload(filename, label, path):
    with _conn() as c:
        c.execute("INSERT INTO uploads (filename, label, path, created_at) VALUES (?,?,?,?)",
                  (filename, label, str(path), now()))


def get_unused_uploads():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM uploads WHERE used=0")]


def mark_used(ids):
    with _conn() as c:
        c.executemany("UPDATE uploads SET used=1 WHERE id=?", [(i,) for i in ids])


def upload_counts():
    with _conn() as c:
        r = c.execute("SELECT COUNT(*) AS total, COALESCE(SUM(used=0),0) AS unused FROM uploads").fetchone()
        return dict(r)


def create_run(n_uploads):
    with _conn() as c:
        cur = c.execute("INSERT INTO retrain_runs (status, started_at, n_uploads) VALUES ('running',?,?)",
                        (now(), n_uploads))
        return cur.lastrowid


def update_run(run_id, **fields):
    keys = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE retrain_runs SET {keys} WHERE id=?", (*fields.values(), run_id))


def is_retraining():
    with _conn() as c:
        return c.execute("SELECT 1 FROM retrain_runs WHERE status='running'").fetchone() is not None


def latest_run():
    with _conn() as c:
        r = c.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT 1").fetchone()
        return dict(r) if r else None


def run_count():
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM retrain_runs").fetchone()[0]
