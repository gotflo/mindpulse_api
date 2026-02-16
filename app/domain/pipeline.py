"""
Real-time processing pipeline.

Orchestrates the full flow: PPI samples → windowing → cleaning →
feature extraction → inference → storage → WebSocket emission.
"""

import logging
import time
from typing import Callable, Optional

from app.acquisition.polar_client import PolarClient, PolarSample, ConnectionState
from app.config.settings import AppConfig
from app.features.hrv_features import HRVFeatureExtractor, HRVFeatures
from app.ml.inference import CognitiveInference, InferenceResult
from app.signal.ppi_cleaning import PPICleaner
from app.signal.windowing import SlidingWindow, WindowData
from app.storage.session_manager import SessionManager

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """Wires together acquisition → signal → features → ML → storage."""

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

        # BLE client
        self._polar = PolarClient(config.ble)

        # Callbacks
        self._on_inference: Optional[Callable[[InferenceResult], None]] = None
        self._on_hr_update: Optional[Callable[[int, float], None]] = None
        self._on_state_change: Optional[Callable] = None

        # Current HR tracking
        self._current_hr: int = 0
        self._last_hr_time: float = 0.0
        self._early_inference_sent: bool = False

        # Wire internal callbacks
        self._polar.on_sample(self._handle_sample)
        self._window.on_window(self._handle_window)

    @property
    def polar_client(self) -> PolarClient:
        return self._polar

    @property
    def current_hr(self) -> int:
        return self._current_hr

    def on_inference(self, callback: Callable[[InferenceResult], None]):
        self._on_inference = callback

    def on_hr_update(self, callback: Callable[[int, float], None]):
        self._on_hr_update = callback

    def on_state_change(self, callback: Callable):
        self._on_state_change = callback
        self._polar.on_state_change(callback)

    def on_unexpected_disconnect(self, callback: Callable):
        """Register callback for when the sensor disconnects unexpectedly."""
        self._polar.on_unexpected_disconnect(callback)

    def _handle_sample(self, sample: PolarSample):
        # Update HR
        if sample.hr > 0:
            self._current_hr = sample.hr
            self._last_hr_time = sample.timestamp
            if self._on_hr_update:
                self._on_hr_update(sample.hr, sample.timestamp)

            # Emit early HR-only inference if no PPI window yet
            if self._window.sample_count == 0 and not self._early_inference_sent:
                self._emit_early_hr_inference(sample.hr, sample.timestamp)

        # Feed PPI to sliding window
        if sample.ppi_ms:
            self._window.add_samples(sample.ppi_ms, sample.timestamp)

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
        """Emit a preliminary inference based on HR only, before PPI arrives.

        Gives immediate visual feedback (~2s after connect) while waiting for
        the PPI buffer to fill up.
        """
        from app.ml.inference import InferenceResult, FatigueTrend
        from app.ml.model import CognitiveScores

        self._early_inference_sent = True

        # Rough estimate from HR alone (normal resting: 60-80 bpm)
        stress_estimate = max(0.0, min(100.0, (hr - 60) * 1.5))
        load_estimate = max(0.0, min(100.0, (hr - 55) * 0.8))
        mean_rr = 60000.0 / hr if hr > 0 else 800.0

        scores = CognitiveScores(
            stress=stress_estimate, cognitive_load=load_estimate,
            fatigue=0.0, timestamp=timestamp,
        )

        features = HRVFeatures(
            mean_hr=float(hr), mean_rr=mean_rr,
            sdnn=0, rmssd=0, pnn50=0, sdsd=0, cv_rr=0,
            lf_power=0, hf_power=0, lf_hf_ratio=0, total_power=0,
            sd1=0, sd2=0, sd_ratio=0,
            quality_ratio=0, sample_count=0,
        )

        result = InferenceResult(
            scores=scores, features=features,
            fatigue_trend=FatigueTrend(slope=0, predicted_fatigue_10min=0, confidence=0),
            timestamp=timestamp, window_quality=0,
        )

        logger.info("Early HR-only inference — hr=%d, stress_est=%.1f", hr, stress_estimate)

        if self._on_inference:
            self._on_inference(result)

    async def start_monitoring(self, on_progress: Optional[Callable[[str], None]] = None):
        """Full auto-start: scan → connect → stream → create session.

        Args:
            on_progress: Optional callback called with status strings
                         ("scanning", "connecting", "streaming").

        Returns:
            SessionInfo for the newly created session.
        """
        # Reset pipeline state
        self._window.reset()
        self._inference.reset()
        self._current_hr = 0
        self._early_inference_sent = False

        # Scan
        if on_progress:
            on_progress("scanning")
        device = await self._polar.scan()
        if device is None:
            raise RuntimeError("Polar device not found")

        # Connect
        if on_progress:
            on_progress("connecting")
        connected = await self._polar.connect()
        if not connected:
            raise RuntimeError("Failed to connect to Polar device")

        # Start streaming
        await self._polar.start_streaming()

        # Create session
        session = self._session_manager.start_session("autre")

        if on_progress:
            on_progress("streaming")

        logger.info("Monitoring started — session %s", session.id)
        return session

    async def stop_monitoring(self) -> Optional[dict]:
        """Full auto-stop: stop streaming → stop session → disconnect.

        Returns:
            Session summary dict, or None if no active session.
        """
        # Stop session first to capture all data
        summary = self._session_manager.stop_session()

        # Stop streaming & disconnect
        try:
            await self._polar.stop_streaming()
        except Exception as e:
            logger.warning("Error stopping streaming: %s", e)
        try:
            await self._polar.disconnect()
        except Exception as e:
            logger.warning("Error disconnecting: %s", e)

        # Reset pipeline
        self._window.reset()
        self._inference.reset()
        self._current_hr = 0
        self._early_inference_sent = False

        logger.info("Monitoring stopped")
        return summary

    def force_stop_session(self) -> Optional[dict]:
        """Stop the active session without touching BLE (for unexpected disconnects)."""
        summary = self._session_manager.stop_session()
        self._window.reset()
        self._inference.reset()
        self._current_hr = 0
        self._early_inference_sent = False
        return summary
