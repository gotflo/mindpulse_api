"""
Entry point for the Cognitive API server.

Usage:
    python run.py
    python run.py --host 0.0.0.0 --port 5000 --debug
"""

import argparse
import logging
import sys

from app.config.settings import load_config
from app.factory import create_app


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Silence noisy libraries
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Cognitive State API Server")
    parser.add_argument("--host", default=None, help="Host address")
    parser.add_argument("--port", type=int, default=None, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    config = load_config()
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.debug:
        config.server.debug = True

    setup_logging(config.server.debug)

    app, socketio = create_app(config)

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting server on %s:%d (debug=%s)",
        config.server.host,
        config.server.port,
        config.server.debug,
    )

    socketio.run(
        app,
        host=config.server.host,
        port=config.server.port,
        debug=config.server.debug,
        use_reloader=config.server.debug,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
