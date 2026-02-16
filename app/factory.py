"""
Application factory.

Wires together all modules and returns a configured Flask app + SocketIO instance.
"""

import logging

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from app.api.routes import register_routes
from app.api.socket_events import register_socket_events
from app.config.settings import AppConfig, load_config
from app.domain.analysis_service import AnalysisService
from app.domain.pipeline import RealtimePipeline
from app.storage.database import Database
from app.storage.session_manager import SessionManager

logger = logging.getLogger(__name__)


def create_app(config: AppConfig = None) -> tuple[Flask, SocketIO]:
    if config is None:
        config = load_config()

    # Flask
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "cognitive-api-secret"
    CORS(app, resources={r"/api/*": {"origins": config.server.cors_origins}})

    # SocketIO
    socketio = SocketIO(
        app,
        cors_allowed_origins=config.server.cors_origins,
        async_mode=config.server.socketio_async_mode,
        logger=False,
        engineio_logger=False,
    )

    # Storage
    db = Database(config.storage.db_path)
    session_manager = SessionManager(config.storage, db)

    # Domain
    pipeline = RealtimePipeline(config, session_manager)
    analysis_service = AnalysisService(db)

    # API
    register_routes(app, session_manager, analysis_service, pipeline)
    register_socket_events(socketio, pipeline, session_manager)

    # Store references for testing
    app.extensions["pipeline"] = pipeline
    app.extensions["session_manager"] = session_manager
    app.extensions["analysis_service"] = analysis_service
    app.extensions["db"] = db

    logger.info("Application created successfully")
    return app, socketio
