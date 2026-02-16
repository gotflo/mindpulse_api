"""
SQLite database layer for persistent storage of sessions and data points.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    start_time REAL NOT NULL,
    end_time REAL,
    activity_type TEXT NOT NULL DEFAULT 'other',
    status TEXT NOT NULL DEFAULT 'active',
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS data_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    hr REAL,
    rmssd REAL,
    sdnn REAL,
    pnn50 REAL,
    mean_rr REAL,
    lf_power REAL,
    hf_power REAL,
    lf_hf_ratio REAL,
    stress REAL,
    cognitive_load REAL,
    fatigue REAL,
    window_quality REAL,
    fatigue_slope REAL,
    fatigue_predicted REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id TEXT PRIMARY KEY,
    duration_sec REAL,
    avg_hr REAL,
    avg_rmssd REAL,
    avg_stress REAL,
    avg_cognitive_load REAL,
    avg_fatigue REAL,
    max_stress REAL,
    max_cognitive_load REAL,
    max_fatigue REAL,
    time_overload_pct REAL,
    time_recovery_pct REAL,
    feedback TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_data_points_session
    ON data_points(session_id);
CREATE INDEX IF NOT EXISTS idx_data_points_timestamp
    ON data_points(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_start
    ON sessions(start_time);
"""


class Database:
    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            logger.info("Database initialized at %s", self._db_path)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Sessions ---

    def create_session(self, session_id: str, start_time: float, activity_type: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, start_time, activity_type, status) VALUES (?, ?, ?, 'active')",
                (session_id, start_time, activity_type),
            )

    def end_session(self, session_id: str, end_time: float):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET end_time = ?, status = 'completed' WHERE id = ?",
                (end_time, session_id),
            )

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_sessions_for_date(self, date_str: str) -> list[dict]:
        """Get sessions for a specific date (YYYY-MM-DD)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM sessions
                   WHERE date(start_time, 'unixepoch', 'localtime') = ?
                   ORDER BY start_time""",
                (date_str,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM data_points WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # --- Data Points ---

    def insert_data_point(self, session_id: str, data: dict):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO data_points
                   (session_id, timestamp, hr, rmssd, sdnn, pnn50, mean_rr,
                    lf_power, hf_power, lf_hf_ratio,
                    stress, cognitive_load, fatigue,
                    window_quality, fatigue_slope, fatigue_predicted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    data.get("timestamp", 0),
                    data.get("hr"),
                    data.get("rmssd"),
                    data.get("sdnn"),
                    data.get("pnn50"),
                    data.get("mean_rr"),
                    data.get("lf_power"),
                    data.get("hf_power"),
                    data.get("lf_hf_ratio"),
                    data.get("stress"),
                    data.get("cognitive_load"),
                    data.get("fatigue"),
                    data.get("window_quality"),
                    data.get("fatigue_slope"),
                    data.get("fatigue_predicted"),
                ),
            )

    def get_session_data(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM data_points WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Summaries ---

    def save_summary(self, session_id: str, summary: dict):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO session_summaries
                   (session_id, duration_sec, avg_hr, avg_rmssd,
                    avg_stress, avg_cognitive_load, avg_fatigue,
                    max_stress, max_cognitive_load, max_fatigue,
                    time_overload_pct, time_recovery_pct, feedback)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    summary.get("duration_sec"),
                    summary.get("avg_hr"),
                    summary.get("avg_rmssd"),
                    summary.get("avg_stress"),
                    summary.get("avg_cognitive_load"),
                    summary.get("avg_fatigue"),
                    summary.get("max_stress"),
                    summary.get("max_cognitive_load"),
                    summary.get("max_fatigue"),
                    summary.get("time_overload_pct"),
                    summary.get("time_recovery_pct"),
                    summary.get("feedback"),
                ),
            )

    def get_summary(self, session_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    # --- Analytics ---

    def get_daily_averages(self, date_str: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT
                       AVG(stress) as avg_stress,
                       AVG(cognitive_load) as avg_cognitive_load,
                       AVG(fatigue) as avg_fatigue,
                       AVG(hr) as avg_hr,
                       AVG(rmssd) as avg_rmssd,
                       COUNT(*) as data_point_count
                   FROM data_points dp
                   JOIN sessions s ON dp.session_id = s.id
                   WHERE date(s.start_time, 'unixepoch', 'localtime') = ?""",
                (date_str,),
            ).fetchone()
            return dict(row) if row and row["data_point_count"] > 0 else None

    def get_weekly_stats(self, end_date: str) -> list[dict]:
        """Get daily averages for the 7 days ending at end_date."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT
                       date(s.start_time, 'unixepoch', 'localtime') as day,
                       AVG(dp.stress) as avg_stress,
                       AVG(dp.cognitive_load) as avg_cognitive_load,
                       AVG(dp.fatigue) as avg_fatigue,
                       AVG(dp.hr) as avg_hr,
                       SUM(CASE WHEN dp.cognitive_load > 70 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as overload_pct
                   FROM data_points dp
                   JOIN sessions s ON dp.session_id = s.id
                   WHERE date(s.start_time, 'unixepoch', 'localtime')
                         BETWEEN date(?, '-6 days') AND ?
                   GROUP BY day
                   ORDER BY day""",
                (end_date, end_date),
            ).fetchall()
            return [dict(r) for r in rows]
