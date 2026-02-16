"""
PPI (pulse-to-pulse interval) cleaning and artifact removal.

Applies physiological range filtering and successive difference checks
to remove ectopic beats and motion artifacts.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config.settings import SignalConfig

logger = logging.getLogger(__name__)


@dataclass
class CleanedPPI:
    timestamps: np.ndarray
    intervals_ms: np.ndarray
    mask_valid: np.ndarray
    quality_ratio: float
    n_original: int
    n_removed: int


class PPICleaner:
    def __init__(self, config: SignalConfig):
        self._config = config

    def clean(
        self, ppi_ms: list[int], timestamps: list[float]
    ) -> CleanedPPI:
        if len(ppi_ms) == 0:
            return CleanedPPI(
                timestamps=np.array([]),
                intervals_ms=np.array([]),
                mask_valid=np.array([], dtype=bool),
                quality_ratio=0.0,
                n_original=0,
                n_removed=0,
            )

        ppi = np.array(ppi_ms, dtype=np.float64)
        ts = np.array(timestamps, dtype=np.float64)
        n_original = len(ppi)

        # 1. Physiological range filter
        mask = (ppi >= self._config.min_ppi_ms) & (ppi <= self._config.max_ppi_ms)

        # 2. Successive difference filter (ectopic beat detection)
        if len(ppi) > 1:
            diff_ratio = np.abs(np.diff(ppi)) / ppi[:-1]
            succ_mask = np.ones(len(ppi), dtype=bool)
            bad_indices = np.where(diff_ratio > self._config.max_ppi_diff_ratio)[0]
            for idx in bad_indices:
                succ_mask[idx] = False
                if idx + 1 < len(succ_mask):
                    succ_mask[idx + 1] = False
            mask = mask & succ_mask

        n_removed = n_original - int(np.sum(mask))
        quality_ratio = float(np.sum(mask)) / n_original if n_original > 0 else 0.0

        if quality_ratio < self._config.min_quality_ratio:
            logger.warning(
                "Low quality segment: %.1f%% valid (%d/%d)",
                quality_ratio * 100,
                n_original - n_removed,
                n_original,
            )

        return CleanedPPI(
            timestamps=ts,
            intervals_ms=ppi,
            mask_valid=mask,
            quality_ratio=quality_ratio,
            n_original=n_original,
            n_removed=n_removed,
        )

    def interpolate(self, cleaned: CleanedPPI) -> np.ndarray:
        """Interpolate removed samples using cubic interpolation."""
        if cleaned.n_original == 0 or np.all(cleaned.mask_valid):
            return cleaned.intervals_ms.copy()

        valid_idx = np.where(cleaned.mask_valid)[0]
        invalid_idx = np.where(~cleaned.mask_valid)[0]

        if len(valid_idx) < 2:
            return cleaned.intervals_ms.copy()

        result = cleaned.intervals_ms.copy()
        result[invalid_idx] = np.interp(
            invalid_idx, valid_idx, cleaned.intervals_ms[valid_idx]
        )
        return result
