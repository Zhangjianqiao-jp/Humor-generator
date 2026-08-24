#!/usr/bin/env python3
"""Serve the blind human-validation dashboard without exposing its answer key."""

from __future__ import annotations

import argparse
import http.server
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_ROOT = Path(
    "results/preference_diagnostics/human_validation_v1"
)
BLOCKED_NAMES = {"blind_key.json"}


class BlindDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def list_directory(self, path: str):  # noqa: ANN001
        self.send_error(403, "Directory listing is disabled")
        return None

    def do_GET(self) -> None:  # noqa: N802
        requested_name = Path(urlsplit(self.path).path).name
        if requested_name in BLOCKED_NAMES:
            self.send_error(403, "Blind answer key is not served")
            return
        if urlsplit(self.path).path == "/":
            self.send_response(302)
            self.send_header("Location", "/preference_pairs_quick_gate_blind.html")
            self.end_headers()
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dashboard directory does not exist: {root}")

    handler = lambda *a, **kw: BlindDashboardHandler(  # noqa: E731
        *a, directory=str(root), **kw
    )
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving blind dashboard from {root}")
    print(f"Open http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
