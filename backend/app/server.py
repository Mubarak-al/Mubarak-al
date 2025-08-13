from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from backend.model.inference import predict_css, MODEL_PATH


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ResponsiveCSS/0.1"

    def _set_headers(self, status: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):  # noqa: N802
        self._set_headers(204)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            body = {
                "status": "ok",
                "model_present": MODEL_PATH.exists(),
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(body).encode("utf-8"))
            return

        if parsed.path == "/predict":
            query = parse_qs(parsed.query)
            try:
                width_px = int(query.get("width_px", [None])[0])
                height_px = int(query.get("height_px", [None])[0])
                diagonal_in = float(query.get("diagonal_in", [None])[0])
                refresh_hz = int(query.get("refresh_hz", [60])[0])
            except Exception:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Invalid or missing parameters"}).encode("utf-8"))
                return

            try:
                scale, tokens, css = predict_css(width_px, height_px, diagonal_in, refresh_hz)
                self._set_headers(200)
                self.wfile.write(json.dumps({"scale": scale, "tokens": tokens, "css": css}).encode("utf-8"))
            except Exception as e:  # pragma: no cover
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            data = json.loads(raw)
            width_px = int(data["width_px"])  # may raise
            height_px = int(data["height_px"])  # may raise
            diagonal_in = float(data["diagonal_in"])  # may raise
            refresh_hz = int(data.get("refresh_hz", 60))
        except Exception:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode("utf-8"))
            return

        try:
            scale, tokens, css = predict_css(width_px, height_px, diagonal_in, refresh_hz)
            self._set_headers(200)
            self.wfile.write(json.dumps({"scale": scale, "tokens": tokens, "css": css}).encode("utf-8"))
        except Exception as e:  # pragma: no cover
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    httpd = HTTPServer((host, port), RequestHandler)
    print(f"Serving on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()