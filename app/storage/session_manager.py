"""
Session lifecycle manager.

Creates, manages, and finalizes recording sessions. Handles
data persistence and CSV/summary export.
"""

import csv
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.config.settings import StorageConfig
from app.storage.database import Database

logger = logging.getLogger(__name__)

ACTIVITY_TYPES = ["travail", "etude", "repos", "autre"]


@dataclass
class SessionInfo:
    id: str
    start_time: float
    end_time: Optional[float] = None
    activity_type: str = "autre"
    status: str = "active"
    data_point_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "activity_type": self.activity_type,
            "status": self.status,
            "data_point_count": self.data_point_count,
            "duration_sec": (self.end_time or time.time()) - self.start_time,
        }


class SessionManager:
    def __init__(self, config: StorageConfig, db: Database):
        self._config = config
        self._db = db
        self._active_session: Optional[SessionInfo] = None
        os.makedirs(config.sessions_dir, exist_ok=True)
        os.makedirs(config.exports_dir, exist_ok=True)

    @property
    def active_session(self) -> Optional[SessionInfo]:
        return self._active_session

    @property
    def is_recording(self) -> bool:
        return self._active_session is not None

    def start_session(self, activity_type: str = "autre") -> SessionInfo:
        if self._active_session is not None:
            raise RuntimeError("A session is already active")

        if activity_type not in ACTIVITY_TYPES:
            activity_type = "autre"

        session_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        self._db.create_session(session_id, start_time, activity_type)

        self._active_session = SessionInfo(
            id=session_id,
            start_time=start_time,
            activity_type=activity_type,
        )
        logger.info("Session started: %s [%s]", session_id, activity_type)
        return self._active_session

    def record_data_point(self, data: dict):
        if self._active_session is None:
            return
        self._db.insert_data_point(self._active_session.id, data)
        self._active_session.data_point_count += 1

    def stop_session(self) -> Optional[dict]:
        if self._active_session is None:
            return None

        end_time = time.time()
        session_id = self._active_session.id
        self._db.end_session(session_id, end_time)
        self._active_session.end_time = end_time
        self._active_session.status = "completed"

        summary = self._compute_summary(session_id)
        self._db.save_summary(session_id, summary)

        session_dict = self._active_session.to_dict()
        session_dict["summary"] = summary

        logger.info("Session stopped: %s (%.0fs)", session_id, end_time - self._active_session.start_time)
        self._active_session = None
        return session_dict

    def _compute_summary(self, session_id: str) -> dict:
        data_points = self._db.get_session_data(session_id)
        if not data_points:
            return {"feedback": "Aucune donnée enregistrée."}

        session = self._db.get_session(session_id)
        duration = (session["end_time"] or time.time()) - session["start_time"]

        def avg(key):
            vals = [dp[key] for dp in data_points if dp.get(key) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        def max_val(key):
            vals = [dp[key] for dp in data_points if dp.get(key) is not None]
            return max(vals) if vals else 0.0

        n = len(data_points)
        overload_count = sum(
            1 for dp in data_points
            if dp.get("cognitive_load") is not None and dp["cognitive_load"] > 70
        )
        recovery_count = sum(
            1 for dp in data_points
            if dp.get("stress") is not None and dp["stress"] < 30
            and dp.get("fatigue") is not None and dp["fatigue"] < 30
        )

        overload_pct = round(overload_count / n * 100, 1) if n > 0 else 0
        recovery_pct = round(recovery_count / n * 100, 1) if n > 0 else 0

        # Generate feedback
        feedback_parts = []
        avg_load = avg("cognitive_load")
        if overload_pct > 40:
            feedback_parts.append(
                f"Charge cognitive élevée pendant {overload_pct}% de la session."
            )
        if avg("fatigue") > 60:
            feedback_parts.append("Fatigue mentale importante détectée.")
        if recovery_pct > 30:
            feedback_parts.append("Bons moments de récupération observés.")
        if avg("stress") > 60:
            feedback_parts.append("Niveau de stress élevé durant la session.")
        if not feedback_parts:
            feedback_parts.append("Session dans les normes. Bon état cognitif général.")

        return {
            "duration_sec": round(duration, 1),
            "avg_hr": round(avg("hr"), 1),
            "avg_rmssd": round(avg("rmssd"), 2),
            "avg_stress": round(avg("stress"), 1),
            "avg_cognitive_load": round(avg("cognitive_load"), 1),
            "avg_fatigue": round(avg("fatigue"), 1),
            "max_stress": round(max_val("stress"), 1),
            "max_cognitive_load": round(max_val("cognitive_load"), 1),
            "max_fatigue": round(max_val("fatigue"), 1),
            "time_overload_pct": overload_pct,
            "time_recovery_pct": recovery_pct,
            "feedback": " ".join(feedback_parts),
        }

    def export_csv(self, session_id: str) -> str:
        data_points = self._db.get_session_data(session_id)
        if not data_points:
            raise ValueError(f"No data for session {session_id}")

        filename = f"session_{session_id}.csv"
        filepath = os.path.join(self._config.exports_dir, filename)

        fieldnames = [
            "timestamp", "hr", "rmssd", "sdnn", "pnn50", "mean_rr",
            "lf_power", "hf_power", "lf_hf_ratio",
            "stress", "cognitive_load", "fatigue",
            "window_quality", "fatigue_slope", "fatigue_predicted",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for dp in data_points:
                writer.writerow({k: dp.get(k) for k in fieldnames})

        logger.info("Exported CSV: %s (%d rows)", filepath, len(data_points))
        return filepath

    def export_summary(self, session_id: str) -> dict:
        session = self._db.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        summary = self._db.get_summary(session_id)
        return {
            "session": session,
            "summary": summary or {},
        }
