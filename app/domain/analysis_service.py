"""
Analysis service for historical data.

Provides weekly stats, overload detection, recovery period
identification, and personalized recommendations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from app.storage.database import Database

logger = logging.getLogger(__name__)

OVERLOAD_THRESHOLD = 70
RECOVERY_STRESS_THRESHOLD = 30
RECOVERY_FATIGUE_THRESHOLD = 30
HIGH_STRESS_THRESHOLD = 60
HIGH_FATIGUE_THRESHOLD = 60


@dataclass
class CriticalPeriod:
    start_timestamp: float
    end_timestamp: float
    period_type: str  # "overload" | "recovery" | "prolonged_fatigue"
    avg_score: float
    duration_sec: float


@dataclass
class DailyDigest:
    date: str
    avg_stress: float
    avg_cognitive_load: float
    avg_fatigue: float
    avg_hr: float
    overload_pct: float
    session_count: int


class AnalysisService:
    def __init__(self, db: Database):
        self._db = db

    def get_daily_digest(self, date_str: str) -> Optional[DailyDigest]:
        averages = self._db.get_daily_averages(date_str)
        if not averages:
            return None

        sessions = self._db.get_sessions_for_date(date_str)

        return DailyDigest(
            date=date_str,
            avg_stress=round(averages.get("avg_stress", 0) or 0, 1),
            avg_cognitive_load=round(averages.get("avg_cognitive_load", 0) or 0, 1),
            avg_fatigue=round(averages.get("avg_fatigue", 0) or 0, 1),
            avg_hr=round(averages.get("avg_hr", 0) or 0, 1),
            overload_pct=0.0,
            session_count=len(sessions),
        )

    def get_weekly_evolution(self, end_date: Optional[str] = None) -> list[dict]:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        return self._db.get_weekly_stats(end_date)

    def detect_critical_periods(self, session_id: str) -> list[dict]:
        data_points = self._db.get_session_data(session_id)
        if not data_points:
            return []

        periods = []

        # Detect overload periods (cognitive_load > 70)
        periods.extend(
            self._detect_periods(data_points, "cognitive_load", OVERLOAD_THRESHOLD, "overload", above=True)
        )

        # Detect recovery periods (low stress AND low fatigue)
        periods.extend(
            self._detect_recovery_periods(data_points)
        )

        # Detect prolonged fatigue (fatigue > 60 for extended time)
        periods.extend(
            self._detect_periods(data_points, "fatigue", HIGH_FATIGUE_THRESHOLD, "prolonged_fatigue", above=True)
        )

        periods.sort(key=lambda p: p["start_timestamp"])
        return periods

    def _detect_periods(
        self, data_points: list[dict], key: str, threshold: float,
        period_type: str, above: bool, min_duration_sec: float = 30.0
    ) -> list[dict]:
        periods = []
        in_period = False
        start_ts = 0.0
        values = []

        for dp in data_points:
            val = dp.get(key)
            if val is None:
                continue

            is_active = (val > threshold) if above else (val < threshold)

            if is_active and not in_period:
                in_period = True
                start_ts = dp["timestamp"]
                values = [val]
            elif is_active and in_period:
                values.append(val)
            elif not is_active and in_period:
                duration = dp["timestamp"] - start_ts
                if duration >= min_duration_sec and values:
                    periods.append({
                        "start_timestamp": start_ts,
                        "end_timestamp": dp["timestamp"],
                        "period_type": period_type,
                        "avg_score": round(sum(values) / len(values), 1),
                        "duration_sec": round(duration, 1),
                    })
                in_period = False
                values = []

        # Close open period
        if in_period and values and data_points:
            duration = data_points[-1]["timestamp"] - start_ts
            if duration >= min_duration_sec:
                periods.append({
                    "start_timestamp": start_ts,
                    "end_timestamp": data_points[-1]["timestamp"],
                    "period_type": period_type,
                    "avg_score": round(sum(values) / len(values), 1),
                    "duration_sec": round(duration, 1),
                })

        return periods

    def _detect_recovery_periods(self, data_points: list[dict]) -> list[dict]:
        periods = []
        in_recovery = False
        start_ts = 0.0

        for dp in data_points:
            stress = dp.get("stress")
            fatigue = dp.get("fatigue")
            if stress is None or fatigue is None:
                continue

            is_recovery = stress < RECOVERY_STRESS_THRESHOLD and fatigue < RECOVERY_FATIGUE_THRESHOLD

            if is_recovery and not in_recovery:
                in_recovery = True
                start_ts = dp["timestamp"]
            elif not is_recovery and in_recovery:
                duration = dp["timestamp"] - start_ts
                if duration >= 30.0:
                    periods.append({
                        "start_timestamp": start_ts,
                        "end_timestamp": dp["timestamp"],
                        "period_type": "recovery",
                        "avg_score": 0.0,
                        "duration_sec": round(duration, 1),
                    })
                in_recovery = False

        return periods

    def generate_recommendations(self, session_id: str) -> list[str]:
        summary = self._db.get_summary(session_id)
        if not summary:
            return []

        recs = []

        if (summary.get("avg_stress") or 0) > HIGH_STRESS_THRESHOLD:
            recs.append("Essayez une respiration profonde (cohérence cardiaque 5-5-5) pendant 5 minutes.")

        if (summary.get("avg_cognitive_load") or 0) > OVERLOAD_THRESHOLD:
            recs.append("Pensez à fractionner vos périodes de travail intense (technique Pomodoro).")

        if (summary.get("time_overload_pct") or 0) > 50:
            recs.append("Plus de 50% de la session en surcharge cognitive. Prévoyez des pauses plus fréquentes.")

        if (summary.get("avg_fatigue") or 0) > HIGH_FATIGUE_THRESHOLD:
            recs.append("Niveau de fatigue élevé. Envisagez une pause longue ou un changement d'activité.")

        if (summary.get("time_recovery_pct") or 0) < 10:
            recs.append("Très peu de temps de récupération. Intégrez des micro-pauses régulières.")

        if not recs:
            recs.append("Bon équilibre cognitif. Continuez à maintenir ce rythme.")

        return recs

    def get_history_days(self, n_days: int = 30) -> list[dict]:
        """Get list of days with their summary scores."""
        days = []
        today = datetime.now()
        for i in range(n_days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            digest = self.get_daily_digest(date_str)
            if digest:
                days.append({
                    "date": digest.date,
                    "avg_stress": digest.avg_stress,
                    "avg_cognitive_load": digest.avg_cognitive_load,
                    "avg_fatigue": digest.avg_fatigue,
                    "avg_hr": digest.avg_hr,
                    "session_count": digest.session_count,
                })
        return days
