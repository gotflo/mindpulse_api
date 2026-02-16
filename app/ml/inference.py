"""
Real-time inference pipeline.

Orchestrates feature extraction → prediction → score smoothing → fatigue trend.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.config.settings import MLConfig
from app.features.hrv_features import HRVFeatureExtractor, HRVFeatures
from app.ml.model import CognitiveModel, CognitiveScores
from app.signal.ppi_cleaning import CleanedPPI, PPICleaner
from app.signal.windowing import WindowData

logger = logging.getLogger(__name__)


@dataclass
class FatigueTrend:
    slope: float  # positive = increasing fatigue
    predicted_fatigue_10min: float
    confidence: float  # 0–1


@dataclass
class InferenceResult:
    scores: CognitiveScores
    features: HRVFeatures
    fatigue_trend: FatigueTrend
    timestamp: float
    window_quality: float

    def to_dict(self) -> dict:
        return {
            "scores": self.scores.to_dict(),
            "features": self.features.to_dict(),
            "fatigue_trend": {
                "slope": round(self.fatigue_trend.slope, 3),
                "predicted_fatigue_10min": round(
                    self.fatigue_trend.predicted_fatigue_10min, 1
                ),
                "confidence": round(self.fatigue_trend.confidence, 2),
            },
            "timestamp": self.timestamp,
            "window_quality": round(self.window_quality, 3),
        }


class CognitiveInference:
    def __init__(
        self,
        ml_config: MLConfig,
        cleaner: PPICleaner,
        feature_extractor: HRVFeatureExtractor,
    ):
        self._config = ml_config
        self._cleaner = cleaner
        self._extractor = feature_extractor
        self._model = CognitiveModel(ml_config.model_path, ml_config.scaler_path)

        # Smoothing state
        self._alpha = ml_config.score_smoothing_alpha
        self._prev_scores: CognitiveScores | None = None

        # Fatigue trend tracking
        self._fatigue_history: deque[tuple[float, float]] = deque(maxlen=120)

    def process_window(self, window: WindowData) -> InferenceResult:
        # 1. Clean PPI
        cleaned = self._cleaner.clean(
            ppi_ms=window.ppi_ms.astype(int).tolist(),
            timestamps=window.timestamps.tolist(),
        )

        # 2. Interpolate if needed
        rr_clean = self._cleaner.interpolate(cleaned)

        # 3. Extract features
        features = self._extractor.extract(rr_clean, cleaned.quality_ratio)

        # 4. Predict
        raw_scores = self._model.predict(features.to_feature_vector())
        raw_scores.timestamp = time.time()

        # 5. Smooth scores
        scores = self._smooth(raw_scores)

        # 6. Track fatigue trend
        self._fatigue_history.append((scores.timestamp, scores.fatigue))
        fatigue_trend = self._compute_fatigue_trend()

        return InferenceResult(
            scores=scores,
            features=features,
            fatigue_trend=fatigue_trend,
            timestamp=scores.timestamp,
            window_quality=cleaned.quality_ratio,
        )

    def _smooth(self, raw: CognitiveScores) -> CognitiveScores:
        if self._prev_scores is None:
            self._prev_scores = raw
            return raw

        a = self._alpha
        smoothed = CognitiveScores(
            stress=a * raw.stress + (1 - a) * self._prev_scores.stress,
            cognitive_load=a * raw.cognitive_load + (1 - a) * self._prev_scores.cognitive_load,
            fatigue=a * raw.fatigue + (1 - a) * self._prev_scores.fatigue,
            timestamp=raw.timestamp,
        )
        self._prev_scores = smoothed
        return smoothed

    def _compute_fatigue_trend(self) -> FatigueTrend:
        if len(self._fatigue_history) < 6:
            return FatigueTrend(slope=0.0, predicted_fatigue_10min=0.0, confidence=0.0)

        times = np.array([t for t, _ in self._fatigue_history])
        values = np.array([v for _, v in self._fatigue_history])

        # Normalize time to minutes
        t_min = (times - times[0]) / 60.0

        # Linear regression
        coeffs = np.polyfit(t_min, values, 1)
        slope = float(coeffs[0])  # fatigue points per minute

        # Project 10 minutes ahead
        horizon = self._config.fatigue_horizon_min
        current = values[-1]
        predicted = float(np.clip(current + slope * horizon, 0, 100))

        # Confidence based on R² and data span
        predicted_line = np.polyval(coeffs, t_min)
        ss_res = np.sum((values - predicted_line) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        data_span_min = t_min[-1]
        span_factor = min(data_span_min / 5.0, 1.0)
        confidence = float(np.clip(r_squared * span_factor, 0, 1))

        return FatigueTrend(
            slope=slope,
            predicted_fatigue_10min=predicted,
            confidence=confidence,
        )

    def reset(self):
        self._prev_scores = None
        self._fatigue_history.clear()
