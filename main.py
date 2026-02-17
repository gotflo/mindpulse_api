"""Entry point for the application."""

from app.factory import create_app

app, socketio = create_app()

if __name__ == "__main__":
    from app.config.settings import load_config

    config = load_config()
    socketio.run(app, host=config.server.host, port=config.server.port)
