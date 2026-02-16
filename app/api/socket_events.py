"""
WebSocket (Socket.IO) event handlers.

Handles real-time bidirectional communication with the Flutter app:
- Emits live HR, HRV features, cognitive scores, and fatigue trends
- Receives control commands (start_monitoring / stop_monitoring)

Auto-monitoring flow:
  Client sends "start_monitoring"
    → scan → connect → start streaming → create session
    → emits monitoring_status at each step
    → emits inference + hr_update continuously

  Client sends "stop_monitoring"
    → stop session → stop streaming → disconnect
    → emits monitoring_status with summary

IMPORTANT: bleak (BLE library) requires a persistent asyncio event loop to
deliver notification callbacks (especially PMD PPI data). We run a dedicated
background thread with an always-running event loop for all async BLE work.
"""

import asyncio
import logging
import threading
from typing import Optional

from flask_socketio import SocketIO, emit

from app.acquisition.polar_client import ConnectionState, DeviceInfo
from app.domain.pipeline import RealtimePipeline
from app.ml.inference import InferenceResult
from app.storage.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _start_background_loop(loop: asyncio.AbstractEventLoop):
    """Run an asyncio event loop forever in a background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def register_socket_events(
    socketio: SocketIO,
    pipeline: RealtimePipeline,
    session_manager: SessionManager,
):
    """Register all Socket.IO event handlers."""

    # Create a persistent event loop in a background thread.
    # This keeps bleak's BLE notification callbacks alive after start_monitoring returns.
    _loop = asyncio.new_event_loop()
    _thread = threading.Thread(target=_start_background_loop, args=(_loop,), daemon=True)
    _thread.start()
    logger.info("Background asyncio event loop started for BLE operations")

    def _run_async(coro):
        """Submit a coroutine to the persistent background loop and wait for result."""
        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        return future.result(timeout=60)

    # ─── Pipeline callbacks → Socket.IO emissions ───

    def _on_inference(result: InferenceResult):
        data = result.to_dict()
        logger.info("Emitting inference — stress=%.1f, load=%.1f, fatigue=%.1f",
                     result.scores.stress, result.scores.cognitive_load, result.scores.fatigue)
        socketio.emit("inference", data)

    def _on_hr_update(hr: int, timestamp: float):
        socketio.emit("hr_update", {"hr": hr, "timestamp": timestamp})

    def _on_state_change(info: DeviceInfo):
        socketio.emit("device_state", {
            "connection_state": info.connection_state.value,
            "name": info.name,
            "address": info.address,
            "battery_level": info.battery_level,
            "signal_quality": round(info.signal_quality, 3),
        })

    def _on_unexpected_disconnect():
        """Handle unexpected BLE disconnection: auto-stop session."""
        logger.warning("Unexpected disconnect — auto-stopping session")
        summary = pipeline.force_stop_session()
        socketio.emit("monitoring_status", {
            "status": "stopped",
            "reason": "device_disconnected",
            "session": None,
            "summary": summary,
        })

    pipeline.on_inference(_on_inference)
    pipeline.on_hr_update(_on_hr_update)
    pipeline.on_state_change(_on_state_change)
    pipeline.on_unexpected_disconnect(_on_unexpected_disconnect)

    # ─── Client events ───

    @socketio.on("connect")
    def handle_connect():
        logger.info("Client connected")
        info = pipeline.polar_client.info
        emit("device_state", {
            "connection_state": info.connection_state.value,
            "name": info.name,
            "address": info.address,
            "battery_level": info.battery_level,
            "signal_quality": round(info.signal_quality, 3),
        })
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

    @socketio.on("start_monitoring")
    def handle_start_monitoring():
        """Auto-start: scan → connect → stream → create session."""
        logger.info("Start monitoring requested")

        if session_manager.is_recording:
            emit("error", {"message": "Monitoring already active"})
            return

        def on_progress(status: str):
            socketio.emit("monitoring_status", {
                "status": status,
                "session": None,
                "summary": None,
            })

        try:
            session = _run_async(
                pipeline.start_monitoring(on_progress=on_progress)
            )
            logger.info("Monitoring started successfully, session=%s", session.id)
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
        """Auto-stop: stop session → stop streaming → disconnect."""
        logger.info("Stop monitoring requested")

        if not session_manager.is_recording:
            emit("error", {"message": "No active monitoring"})
            return

        try:
            summary = _run_async(pipeline.stop_monitoring())
            emit("monitoring_status", {
                "status": "stopped",
                "reason": "user_stopped",
                "session": None,
                "summary": summary,
            })
        except Exception as e:
            logger.error("Stop monitoring failed: %s", e, exc_info=True)
            emit("error", {"message": f"Stop monitoring failed: {e}"})
