"""Flask API and production static host for the React/Leaflet explorer."""
from __future__ import annotations

import argparse
from pathlib import Path

from flask import Blueprint, Flask, current_app, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from .explorer_service import AnalysisChanged, AnalysisNotFound, ExplorerService, REPO_ROOT


def explorer_blueprint(service: ExplorerService) -> Blueprint:
    api = Blueprint("explorer", __name__, url_prefix="/api")

    @api.get("/catalog")
    def catalog():
        return jsonify(service.catalog())

    @api.get("/cities/<city>/analysis")
    def analysis(city):
        return jsonify(service.analysis(city, request.args.get("scenario", "plus_50cm")))

    @api.get("/cities/<city>/components/<int:component_id>")
    def inspect(city, component_id):
        return jsonify(service.inspect(city, request.args.get("scenario", "plus_50cm"),
                                       component_id, request.args.get("version")))

    @api.get("/cities/<city>/evidence")
    def evidence(city):
        return jsonify(service.evidence(city))

    @api.post("/cities/<city>/refresh")
    def refresh(city):
        body = request.get_json(silent=True) or {}
        return jsonify(service.refresh(city, body.get("scenario", "plus_50cm"), int(body.get("horizon", 6))))

    @api.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, AnalysisNotFound):
            status = 404
        elif isinstance(exc, AnalysisChanged):
            status = 409
        elif isinstance(exc, (ValueError, TypeError)):
            status = 400
        elif isinstance(exc, HTTPException):
            status = exc.code
        else:
            status = 503
            current_app.logger.exception("Explorer request failed")
        return jsonify({"error": str(exc) if status < 500 else "Analysis unavailable. Check the API log and try again."}), status

    return api


def create_app(root: Path = REPO_ROOT, live_provider=None) -> Flask:
    app = Flask(__name__)
    service = ExplorerService(root, live_provider=live_provider)
    app.register_blueprint(explorer_blueprint(service))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "app": "coastal-flood-explorer"})

    dist = Path(root) / "Frontend/react-app/dist"

    @app.get("/")
    @app.get("/<path:filename>")
    def frontend(filename="index.html"):
        if filename.startswith("api/"):
            return jsonify({"error": "Endpoint not found."}), 404
        if not dist.exists():
            return jsonify({"error": "Run npm run build in Frontend/react-app or open the Vite development server."}), 404
        target = (dist / filename).resolve()
        if target.is_relative_to(dist.resolve()) and target.is_file():
            return send_from_directory(dist, filename)
        return send_from_directory(dist, "index.html")

    return app


def create_runtime_app() -> Flask:
    provider = None

    def predict(**kwargs):
        nonlocal provider
        if provider is None:
            from .realtime_api import RealtimeService
            provider = RealtimeService(
                model_path=str(REPO_ROOT / "Backend/sea_level_risk/outputs/sea_level_axial_lstm.keras"),
                metadata_path=str(REPO_ROOT / "Backend/sea_level_risk/outputs/metadata.json"),
                models_root=str(REPO_ROOT / "Backend/sea_level_risk/outputs/models"),
            )
        return provider.predict(**kwargs)

    return create_app(live_provider=predict)


def main():
    parser = argparse.ArgumentParser(description="Serve the interactive flood explorer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    create_runtime_app().run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
