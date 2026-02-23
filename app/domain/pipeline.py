"""
Cloud processing pipeline.

Receives raw PPI/HR data from the mobile app via Socket.IO,
then orchestrates: windowing -> cleaning -> feature extraction ->
inference -> storage -> WebSocket emission.
"""

import logging
import time
from typing import Callable, Optional

from app.config.settings import AppConfig
from app.features.hrv_features import HRVFeatureExtractor, HRVFeatures
from app.ml.inference import CognitiveInference, InferenceResult
from app.signal.ppi_cleaning import PPICleaner
from app.signal.windowing import SlidingWindow, WindowData
from app.storage.session_manager import SessionManager

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """Wires together signal processing -> features -> ML -> storage.

    Data is received from the mobile app (which handles BLE acquisition)
    via receive_ppi_data() and receive_hr_data().
    """

    def __init__(self, config: AppConfig, session_manager: SessionManager):
        self._config = config
        self._session_manager = session_manager

        # Signal processing
        self._cleaner = PPICleaner(config.signal)
        self._window = SlidingWindow(config.signal)
        self._feature_extractor = HRVFeatureExtractor()

        # Inference
        self._inference = CognitiveInference(
            config.ml, self._cleaner, self._feature_extractor
        )

        # Callbacks
        self._on_inference: Optional[Callable[[InferenceResult], None]] = None
        self._on_hr_update: Optional[Callable[[int, float], None]] = None

        # Current HR tracking
        self._current_hr: int = 0
        self._last_hr_time: float = 0.0
        self._early_inference_sent: bool = False
        self._last_early_time: float = 0.0

        # Wire internal callback
        self._window.on_window(self._handle_window)

    @property
    def current_hr(self) -> int:
        return self._current_hr

    def on_inference(self, callback: Callable[[InferenceResult], None]):
        self._on_inference = callback

    def on_hr_update(self, callback: Callable[[int, float], None]):
        self._on_hr_update = callback

    # ─── Data reception from mobile app ───

    def receive_ppi_data(self, ppi_ms: list[int], timestamp: float):
        """Receive raw PPI samples from the mobile app via Socket.IO."""
        if ppi_ms:
            self._window.add_samples(ppi_ms, timestamp)

    def receive_hr_data(self, hr: int, timestamp: float):
        """Receive HR value from the mobile app via Socket.IO."""
        if hr > 0:
            self._current_hr = hr
            self._last_hr_time = timestamp
            if self._on_hr_update:
                self._on_hr_update(hr, timestamp)

            # Emit HR-only inference periodically when no PPI data is flowing.
            # This ensures scores still update even if PPI stream fails.
            if self._window.sample_count == 0:
                now = timestamp
                if not self._early_inference_sent or (now - self._last_early_time > 3.0):
                    self._emit_early_hr_inference(hr, timestamp)
                    self._last_early_time = now

    # ─── Session management ───

    def start_session(self, activity_type: str = "autre"):
        """Start a new recording session (called when mobile starts monitoring)."""
        self._window.reset()
        self._inference.reset()
        self._current_hr = 0
        self._early_inference_sent = False
        self._last_early_time = 0.0
        session = self._session_manager.start_session(activity_type)
        logger.info("Session started — %s", session.id)
        return session

    def stop_session(self) -> Optional[dict]:
        """Stop the current session (called when mobile stops monitoring)."""
        summary = self._session_manager.stop_session()
        self._window.reset()
        self._inference.reset()
        self._current_hr = 0
        self._early_inference_sent = False
        self._last_early_time = 0.0
        logger.info("Session stopped")
        return summary

    def force_stop_session(self) -> Optional[dict]:
        """Stop the active session (for unexpected disconnects from mobile)."""
        summary = self._session_manager.stop_session()
        self._window.reset()
        self._inference.reset()
        self._current_hr = 0
        self._early_inference_sent = False
        self._last_early_time = 0.0
        return summary

    # ─── Internal processing ───

    def _handle_window(self, window: WindowData):
        logger.info("Window received — %d samples, span=%.1fs",
                     window.sample_count, window.window_end - window.window_start)
        try:
            result = self._inference.process_window(window)

            # Store data point if session active
            if self._session_manager.is_recording:
                data_point = {
                    "timestamp": result.timestamp,
                    "hr": result.features.mean_hr,
                    "rmssd": result.features.rmssd,
                    "sdnn": result.features.sdnn,
                    "pnn50": result.features.pnn50,
                    "mean_rr": result.features.mean_rr,
                    "lf_power": result.features.lf_power,
                    "hf_power": result.features.hf_power,
                    "lf_hf_ratio": result.features.lf_hf_ratio,
                    "stress": result.scores.stress,
                    "cognitive_load": result.scores.cognitive_load,
                    "fatigue": result.scores.fatigue,
                    "window_quality": result.window_quality,
                    "fatigue_slope": result.fatigue_trend.slope,
                    "fatigue_predicted": result.fatigue_trend.predicted_fatigue_10min,
                }
                self._session_manager.record_data_point(data_point)

            # Emit to WebSocket
            if self._on_inference:
                self._on_inference(result)

        except Exception as e:
            logger.error("Pipeline processing error: %s", e, exc_info=True)

    def _emit_early_hr_inference(self, hr: int, timestamp: float):
        """Emit inference based on HR only (when PPI data is unavailable).

        Produces periodically updated scores so the UI always reflects
        the current HR, even if PPI streaming fails.
        """
        from app.ml.inference import InferenceResult, FatigueTrend
        from app.ml.model import CognitiveScores

        self._early_inference_sent = True
        mean_rr = 60000.0 / hr if hr > 0 else 800.0

        # Rough estimates from HR (60-80 bpm = resting range)
        stress_estimate = max(0.0, min(100.0, (hr - 60) * 1.5))
        load_estimate = max(0.0, min(100.0, (hr - 55) * 0.8))

        # Basic fatigue: grows slowly with time since session start
        session = self._session_manager.active_session
        elapsed_min = 0.0
        if session:
            elapsed_min = (timestamp - session.start_time) / 60.0
        fatigue_estimate = max(0.0, min(100.0, elapsed_min * 1.5 + (hr - 65) * 0.3))

        # Apply smoothing against previous scores
        raw = CognitiveScores(
            stress=stress_estimate, cognitive_load=load_estimate,
            fatigue=fatigue_estimate, timestamp=timestamp,
        )
        scores = self._inference._smooth(raw)

        features = HRVFeatures(
            mean_hr=float(hr), mean_rr=mean_rr,
            sdnn=0, rmssd=0, pnn50=0, sdsd=0, cv_rr=0,
            lf_power=0, hf_power=0, lf_hf_ratio=0, total_power=0,
            sd1=0, sd2=0, sd_ratio=0,
            quality_ratio=0, sample_count=0,
        )

        # Track fatigue trend
        self._inference._fatigue_history.append((timestamp, scores.fatigue))
        fatigue_trend = self._inference._compute_fatigue_trend()

        result = InferenceResult(
            scores=scores, features=features,
            fatigue_trend=fatigue_trend,
            timestamp=timestamp, window_quality=0,
        )

        # Store data point if session active
        if self._session_manager.is_recording:
            self._session_manager.record_data_point({
                "timestamp": timestamp,
                "hr": float(hr),
                "rmssd": 0, "sdnn": 0, "pnn50": 0, "mean_rr": mean_rr,
                "lf_power": 0, "hf_power": 0, "lf_hf_ratio": 0,
                "stress": scores.stress,
                "cognitive_load": scores.cognitive_load,
                "fatigue": scores.fatigue,
                "window_quality": 0, "fatigue_slope": fatigue_trend.slope,
                "fatigue_predicted": fatigue_trend.predicted_fatigue_10min,
            })

        logger.info("HR-only inference — hr=%d stress=%.1f load=%.1f fatigue=%.1f",
                     hr, scores.stress, scores.cognitive_load, scores.fatigue)

        if self._on_inference:
            self._on_inference(result)
