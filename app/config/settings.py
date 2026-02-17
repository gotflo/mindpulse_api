import os
from dataclasses import dataclass, field


@dataclass
class SignalConfig:
    window_size_sec: float = 15.0
    window_step_sec: float = 1.0
    min_ppi_ms: int = 300
    max_ppi_ms: int = 2000
    max_ppi_diff_ratio: float = 0.20
    min_quality_ratio: float = 0.80
    interpolation_method: str = "cubic"


@dataclass
class MLConfig:
    model_path: str = os.path.join(
        os.path.dirname(__file__), "..", "ml", "models", "cognitive_model.joblib"
    )
    scaler_path: str = os.path.join(
        os.path.dirname(__file__), "..", "ml", "models", "scaler.joblib"
    )
    prediction_interval_sec: float = 1.0
    fatigue_horizon_min: float = 10.0
    score_smoothing_alpha: float = 0.3


@dataclass
class StorageConfig:
    db_path: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "cognitive.db"
    )
    sessions_dir: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "sessions"
    )
    exports_dir: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "exports"
    )


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    cors_origins: str = "*"
    socketio_async_mode: str = "threading"


@dataclass
class AppConfig:
    signal: SignalConfig = field(default_factory=SignalConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config() -> AppConfig:
    config = AppConfig()
    config.server.host = os.getenv("HOST", config.server.host)
    config.server.port = int(os.getenv("PORT", config.server.port))
    config.server.debug = os.getenv("DEBUG", "false").lower() == "true"
    return config
