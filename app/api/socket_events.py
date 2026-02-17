"""
WebSocket (Socket.IO) event handlers.

Handles real-time bidirectional communication with the Flutter app:
- Receives raw PPI/HR data from the mobile BLE acquisition
- Processes data through the pipeline (cleaning, features, inference)
- Emits cognitive scores and HR updates back to the mobile app
- Manages session lifecycle (start/stop)

Data flow:
  Mobile sends "ppi_data" / "hr_data" continuously
    -> pipeline processes -> emits "inference" + "hr_update"

  Mobile sends "start_monitoring" -> creates session
  Mobile sends "stop_monitoring" -> stops session with summary
"""

import logging
import time

from flask_socketio import SocketIO, emit

from app.domain.pipeline import RealtimePipeline
from app.ml.inference import InferenceResult
from app.storage.session_manager import SessionManager

logger = logging.getLogger(__name__)


def register_socket_events(
    socketio: SocketIO,
    pipeline: RealtimePipeline,
    session_manager: SessionManager,
):
    """Register all Socket.IO event handlers."""

    # ─── Pipeline callbacks → Socket.IO emissions ───

    def _on_inference(result: InferenceResult):
        data = result.to_dict()
        logger.info("Emitting inference — stress=%.1f, load=%.1f, fatigue=%.1f",
                     result.scores.stress, result.scores.cognitive_load, result.scores.fatigue)
        socketio.emit("inference", data)

    def _on_hr_update(hr: int, timestamp: float):
        socketio.emit("hr_update", {"hr": hr, "timestamp": timestamp})

    pipeline.on_inference(_on_inference)
    pipeline.on_hr_update(_on_hr_update)

    # ─── Data reception from mobile BLE ───

    @socketio.on("ppi_data")
    def handle_ppi_data(data):
        """Receive raw PPI samples from mobile BLE acquisition."""
        ppi_ms = data.get("ppi_ms", [])
        timestamp = data.get("timestamp", time.time())
        pipeline.receive_ppi_data(ppi_ms, timestamp)

    @socketio.on("hr_data")
    def handle_hr_data(data):
        """Receive HR value from mobile BLE acquisition."""
        hr = data.get("hr", 0)
        timestamp = data.get("timestamp", time.time())
        pipeline.receive_hr_data(hr, timestamp)

    # ─── Client lifecycle events ───

    @socketio.on("connect")
    def handle_connect():
        logger.info("Client connected")
        # Emit current monitoring status
        active = session_manager.active_session
        emit("monitoring_status", {
            "status": "streaming" if active else "stopped",
            "session": active.to_dict() if active else None,
            "summary": None,
        })

    @socketio.on("disconnect")
    def handle_disconnect():
        logger.info("Client disconnected")

    # ─── Session control ───

    @socketio.on("start_monitoring")
    def handle_start_monitoring():
        """Create a new session. BLE acquisition is handled by the mobile app."""
        logger.info("Start monitoring requested")

        if session_manager.is_recording:
            emit("error", {"message": "Monitoring already active"})
            return

        try:
            session = pipeline.start_session("autre")
            emit("monitoring_status", {
                "status": "streaming",
                "session": session.to_dict(),
                "summary": None,
            })
        except Exception as e:
            logger.error("Start monitoring failed: %s", e, exc_info=True)
            emit("monitoring_status", {
                "status": "stopped",
                "session": None,
                "summary": None,
            })
            emit("error", {"message": f"Start monitoring failed: {e}"})

    @socketio.on("stop_monitoring")
    def handle_stop_monitoring():
        """Stop the current session. BLE disconnection is handled by the mobile app."""
        logger.info("Stop monitoring requested")

        if not session_manager.is_recording:
            emit("error", {"message": "No active monitoring"})
            return

        try:
            summary = pipeline.stop_session()
            emit("monitoring_status", {
                "status": "stopped",
                "reason": "user_stopped",
                "session": None,
                "summary": summary,
            })
        except Exception as e:
            logger.error("Stop monitoring failed: %s", e, exc_info=True)
            emit("error", {"message": f"Stop monitoring failed: {e}"})

    @socketio.on("force_stop")
    def handle_force_stop():
        """Force-stop session (e.g., unexpected BLE disconnect on mobile)."""
        logger.warning("Force stop requested (mobile BLE disconnect)")
        summary = pipeline.force_stop_session()
        emit("monitoring_status", {
            "status": "stopped",
            "reason": "device_disconnected",
            "session": None,
            "summary": summary,
        })
