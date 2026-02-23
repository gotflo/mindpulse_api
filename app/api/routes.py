"""
REST API routes.

Provides endpoints for session management, history, analysis,
export, and device status.
"""

import logging
from flask import Blueprint, jsonify, request, send_file

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def register_routes(app, session_manager, analysis_service, pipeline):
    """Register all REST routes on the Flask app."""

    # ─── Health ───

    @api_bp.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    # ─── Sessions ───

    @api_bp.route("/sessions", methods=["GET"])
    def list_sessions():
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        sessions = session_manager._db.list_sessions(limit=limit, offset=offset)
        # Attach summary to each session for the Flutter client
        for s in sessions:
            summary = session_manager._db.get_summary(s["id"])
            s["summary"] = summary
        return jsonify({"sessions": sessions})

    @api_bp.route("/sessions/active", methods=["GET"])
    def active_session():
        session = session_manager.active_session
        if session is None:
            return jsonify({"active": False})
        return jsonify({"active": True, "session": session.to_dict()})

    @api_bp.route("/monitoring/status", methods=["GET"])
    def monitoring_status():
        active = session_manager.active_session
        return jsonify({
            "is_monitoring": active is not None,
            "session": active.to_dict() if active else None,
        })

    @api_bp.route("/sessions/<session_id>", methods=["GET"])
    def get_session(session_id):
        session = session_manager._db.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        summary = session_manager._db.get_summary(session_id)
        return jsonify({"session": session, "summary": summary})

    @api_bp.route("/sessions/<session_id>/data", methods=["GET"])
    def get_session_data(session_id):
        data = session_manager._db.get_session_data(session_id)
        return jsonify({"data_points": data})

    @api_bp.route("/sessions/<session_id>/critical-periods", methods=["GET"])
    def get_critical_periods(session_id):
        periods = analysis_service.detect_critical_periods(session_id)
        return jsonify({"critical_periods": periods})

    @api_bp.route("/sessions/<session_id>/recommendations", methods=["GET"])
    def get_recommendations(session_id):
        recs = analysis_service.generate_recommendations(session_id)
        return jsonify({"recommendations": recs})

    # ─── Export ───

    @api_bp.route("/sessions/<session_id>/export/csv", methods=["GET"])
    def export_csv(session_id):
        try:
            filepath = session_manager.export_csv(session_id)
            return send_file(filepath, as_attachment=True)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    @api_bp.route("/sessions/<session_id>/export/summary", methods=["GET"])
    def export_summary(session_id):
        try:
            result = session_manager.export_summary(session_id)
            return jsonify(result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    # ─── History & Analysis ───

    @api_bp.route("/history/days", methods=["GET"])
    def history_days():
        n_days = request.args.get("n", 30, type=int)
        days = analysis_service.get_history_days(n_days)
        return jsonify({"days": days})

    @api_bp.route("/history/<date_str>", methods=["GET"])
    def daily_digest(date_str):
        digest = analysis_service.get_daily_digest(date_str)
        if not digest:
            return jsonify({"error": "No data for this date"}), 404
        return jsonify({
            "date": digest.date,
            "avg_stress": digest.avg_stress,
            "avg_cognitive_load": digest.avg_cognitive_load,
            "avg_fatigue": digest.avg_fatigue,
            "avg_hr": digest.avg_hr,
            "session_count": digest.session_count,
        })

    @api_bp.route("/analysis/weekly", methods=["GET"])
    def weekly_analysis():
        end_date = request.args.get("end_date", None)
        stats = analysis_service.get_weekly_evolution(end_date)
        return jsonify({"days": stats})

    # ─── Settings ───

    @api_bp.route("/settings/window", methods=["GET"])
    def get_window_settings():
        return jsonify({
            "window_size_sec": pipeline._config.signal.window_size_sec,
            "window_step_sec": pipeline._config.signal.window_step_sec,
        })

    @api_bp.route("/settings/window", methods=["PUT"])
    def update_window_settings():
        data = request.get_json(silent=True) or {}
        if "window_size_sec" in data:
            pipeline._config.signal.window_size_sec = float(data["window_size_sec"])
        if "window_step_sec" in data:
            pipeline._config.signal.window_step_sec = float(data["window_step_sec"])
        return jsonify({"status": "updated"})

    app.register_blueprint(api_bp)
