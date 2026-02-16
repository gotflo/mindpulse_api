"""
Cognitive state prediction model.

Wraps the ML model (scikit-learn or equivalent) for predicting
stress, cognitive load, and mental fatigue from HRV features.

When no trained model is available, uses a rule-based heuristic
based on established HRV-cognition relationships from literature.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CognitiveScores:
    stress: float         # 0–100
    cognitive_load: float  # 0–100
    fatigue: float         # 0–100
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stress": round(self.stress, 1),
            "cognitive_load": round(self.cognitive_load, 1),
            "fatigue": round(self.fatigue, 1),
            "timestamp": self.timestamp,
        }


class CognitiveModel:
    """
    Dual-mode model: loads a trained sklearn model if available,
    otherwise falls back to physiologically-grounded heuristics.
    """

    def __init__(self, model_path: str, scaler_path: str):
        self._model = None
        self._scaler = None
        self._use_heuristic = True

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                import joblib
                self._model = joblib.load(model_path)
                self._scaler = joblib.load(scaler_path)
                self._use_heuristic = False
                logger.info("Loaded trained model from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load model, using heuristic: %s", e)
        else:
            logger.info("No trained model found, using heuristic mode")

    @property
    def is_heuristic(self) -> bool:
        return self._use_heuristic

    def predict(self, features: np.ndarray) -> CognitiveScores:
        if self._use_heuristic:
            return self._heuristic_predict(features)
        return self._model_predict(features)

    def _model_predict(self, features: np.ndarray) -> CognitiveScores:
        try:
            X = features.reshape(1, -1)
            X_scaled = self._scaler.transform(X)
            predictions = self._model.predict(X_scaled)

            # Model outputs [stress, cognitive_load, fatigue]
            scores = np.clip(predictions[0], 0, 100)
            return CognitiveScores(
                stress=float(scores[0]),
                cognitive_load=float(scores[1]),
                fatigue=float(scores[2]),
            )
        except Exception as e:
            logger.error("Model prediction failed: %s", e)
            return self._heuristic_predict(features)

    def _heuristic_predict(self, features: np.ndarray) -> CognitiveScores:
        """
        Rule-based estimation grounded in HRV-cognition literature:
        - High LF/HF ratio + low RMSSD → high stress
        - Low HRV (SDNN) + high HR → high cognitive load
        - Decreasing RMSSD + low pNN50 over time → fatigue
        """
        # Feature vector order: [mean_hr, mean_rr, sdnn, rmssd, pnn50, sdsd,
        #   cv_rr, lf_power, hf_power, lf_hf_ratio, total_power, sd1, sd2, sd_ratio]
        mean_hr = features[0]
        sdnn = features[2]
        rmssd = features[3]
        pnn50 = features[4]
        lf_hf = features[9]
        sd1 = features[11]

        # Stress: driven by sympathetic activation
        # High LF/HF (>2.0) and low RMSSD (<30ms) indicate stress
        stress_lf = np.clip((lf_hf - 0.5) / 4.0 * 100, 0, 100)
        stress_rmssd = np.clip((1 - rmssd / 80.0) * 100, 0, 100)
        stress_hr = np.clip((mean_hr - 60) / 50.0 * 60, 0, 100)
        stress = 0.4 * stress_lf + 0.4 * stress_rmssd + 0.2 * stress_hr

        # Cognitive load: reduced HRV + elevated HR
        load_sdnn = np.clip((1 - sdnn / 100.0) * 100, 0, 100)
        load_hr = np.clip((mean_hr - 55) / 55.0 * 80, 0, 100)
        load_sd1 = np.clip((1 - sd1 / 50.0) * 100, 0, 100)
        cognitive_load = 0.35 * load_sdnn + 0.35 * load_hr + 0.3 * load_sd1

        # Fatigue: parasympathetic withdrawal pattern
        fatigue_rmssd = np.clip((1 - rmssd / 60.0) * 80, 0, 100)
        fatigue_pnn50 = np.clip((1 - pnn50 / 30.0) * 80, 0, 100)
        fatigue_hr = np.clip((mean_hr - 65) / 40.0 * 50, 0, 100)
        fatigue = 0.4 * fatigue_rmssd + 0.35 * fatigue_pnn50 + 0.25 * fatigue_hr

        return CognitiveScores(
            stress=float(np.clip(stress, 0, 100)),
            cognitive_load=float(np.clip(cognitive_load, 0, 100)),
            fatigue=float(np.clip(fatigue, 0, 100)),
        )
