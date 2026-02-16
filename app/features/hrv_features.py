"""
HRV feature extraction from cleaned PPI intervals.

Computes time-domain, frequency-domain, and nonlinear HRV features
used as input to the cognitive state prediction models.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import interpolate, signal as sp_signal

logger = logging.getLogger(__name__)


@dataclass
class HRVFeatures:
    # Time-domain
    mean_hr: float
    mean_rr: float
    sdnn: float
    rmssd: float
    pnn50: float
    sdsd: float
    cv_rr: float  # coefficient of variation

    # Frequency-domain
    lf_power: float   # 0.04–0.15 Hz
    hf_power: float   # 0.15–0.40 Hz
    lf_hf_ratio: float
    total_power: float

    # Nonlinear
    sd1: float   # Poincaré SD1
    sd2: float   # Poincaré SD2
    sd_ratio: float

    # Quality
    quality_ratio: float
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "mean_hr": round(self.mean_hr, 1),
            "mean_rr": round(self.mean_rr, 1),
            "sdnn": round(self.sdnn, 2),
            "rmssd": round(self.rmssd, 2),
            "pnn50": round(self.pnn50, 2),
            "sdsd": round(self.sdsd, 2),
            "cv_rr": round(self.cv_rr, 4),
            "lf_power": round(self.lf_power, 2),
            "hf_power": round(self.hf_power, 2),
            "lf_hf_ratio": round(self.lf_hf_ratio, 3),
            "total_power": round(self.total_power, 2),
            "sd1": round(self.sd1, 2),
            "sd2": round(self.sd2, 2),
            "sd_ratio": round(self.sd_ratio, 3),
            "quality_ratio": round(self.quality_ratio, 3),
            "sample_count": self.sample_count,
        }

    def to_feature_vector(self) -> np.ndarray:
        """Return ordered feature vector for ML model input."""
        return np.array([
            self.mean_hr,
            self.mean_rr,
            self.sdnn,
            self.rmssd,
            self.pnn50,
            self.sdsd,
            self.cv_rr,
            self.lf_power,
            self.hf_power,
            self.lf_hf_ratio,
            self.total_power,
            self.sd1,
            self.sd2,
            self.sd_ratio,
        ])


FEATURE_NAMES = [
    "mean_hr", "mean_rr", "sdnn", "rmssd", "pnn50", "sdsd", "cv_rr",
    "lf_power", "hf_power", "lf_hf_ratio", "total_power",
    "sd1", "sd2", "sd_ratio",
]


class HRVFeatureExtractor:
    def extract(
        self, rr_intervals_ms: np.ndarray, quality_ratio: float = 1.0
    ) -> HRVFeatures:
        if len(rr_intervals_ms) < 4:
            return self._empty_features(quality_ratio, len(rr_intervals_ms))

        rr = rr_intervals_ms.astype(np.float64)

        time_features = self._time_domain(rr)
        freq_features = self._frequency_domain(rr)
        nonlinear = self._nonlinear(rr)

        return HRVFeatures(
            **time_features,
            **freq_features,
            **nonlinear,
            quality_ratio=quality_ratio,
            sample_count=len(rr),
        )

    def _time_domain(self, rr: np.ndarray) -> dict:
        mean_rr = float(np.mean(rr))
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
        sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0

        diffs = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
        sdsd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

        nn50 = int(np.sum(np.abs(diffs) > 50))
        pnn50 = float(nn50) / len(diffs) * 100.0 if len(diffs) > 0 else 0.0

        cv_rr = sdnn / mean_rr if mean_rr > 0 else 0.0

        return {
            "mean_hr": mean_hr,
            "mean_rr": mean_rr,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnn50,
            "sdsd": sdsd,
            "cv_rr": cv_rr,
        }

    def _frequency_domain(self, rr: np.ndarray) -> dict:
        """Compute LF/HF power via Welch's method on interpolated RR series."""
        try:
            # Build cumulative time axis in seconds
            t_rr = np.cumsum(rr) / 1000.0
            t_rr -= t_rr[0]

            if t_rr[-1] < 10.0:
                return {"lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0, "total_power": 0.0}

            # Resample at 4 Hz
            fs = 4.0
            t_uniform = np.arange(0, t_rr[-1], 1.0 / fs)
            f_interp = interpolate.interp1d(
                t_rr, rr, kind="cubic", fill_value="extrapolate"
            )
            rr_uniform = f_interp(t_uniform)

            # Detrend
            rr_uniform = rr_uniform - np.mean(rr_uniform)

            # Welch PSD
            nperseg = min(256, len(rr_uniform))
            freqs, psd = sp_signal.welch(
                rr_uniform, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
            )

            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

            lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
            hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
            total_power = lf_power + hf_power
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

            return {
                "lf_power": lf_power,
                "hf_power": hf_power,
                "lf_hf_ratio": lf_hf_ratio,
                "total_power": total_power,
            }
        except Exception as e:
            logger.warning("Frequency domain extraction failed: %s", e)
            return {"lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0, "total_power": 0.0}

    def _nonlinear(self, rr: np.ndarray) -> dict:
        """Poincaré plot analysis."""
        if len(rr) < 4:
            return {"sd1": 0.0, "sd2": 0.0, "sd_ratio": 0.0}

        rr_n = rr[:-1]
        rr_n1 = rr[1:]
        diff = rr_n1 - rr_n
        summ = rr_n1 + rr_n

        sd1 = float(np.std(diff, ddof=1) / np.sqrt(2))
        sd2 = float(np.std(summ, ddof=1) / np.sqrt(2))
        sd_ratio = sd1 / sd2 if sd2 > 0 else 0.0

        return {"sd1": sd1, "sd2": sd2, "sd_ratio": sd_ratio}

    def _empty_features(self, quality_ratio: float, count: int) -> HRVFeatures:
        return HRVFeatures(
            mean_hr=0, mean_rr=0, sdnn=0, rmssd=0, pnn50=0, sdsd=0, cv_rr=0,
            lf_power=0, hf_power=0, lf_hf_ratio=0, total_power=0,
            sd1=0, sd2=0, sd_ratio=0,
            quality_ratio=quality_ratio, sample_count=count,
        )
