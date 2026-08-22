"""A scripted OpenAI-compatible endpoint, so the demo and CI are identical everywhere.

It answers by matching a phrase in the system prompt, which means the demo exercises the
real HTTP client, the real dialect handling and the real parsing — everything except the
part that would make the output vary from machine to machine.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = {
    "deserves to become a permanent memory": json.dumps([
        {"topic": "archive-on-slow-disk", "kind": "project",
         "why": "a decision about where the archive lives",
         "quotes": ["[USER] put the archive on the slow disk"]},
    ]),
    "walked past": "[]",
    "actually NEW": "NEW\nthe store has nothing about disks",
    "INDEX of everything remembered": '["archive-on-slow-disk"]',
    "You write the final memory": (
        "SLUG: archive-on-slow-disk\n"
        "TITLE: Archive on the slow disk\n"
        "DESC: the archive lives on the slow disk; the fast one stays scratch\n"
        "BODY:\n"
        "The archive goes on the slow disk. The fast disk is scratch space.\n\n"
        "**Why:** the other way round burns write endurance for nothing.\n"
        "**How to apply:** check which disk a target directory is on before writing.\n"),
    "draw the last line": "POUR\nreason: inside its evidence and useful later",
}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._json({"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        system = body["messages"][0]["content"]
        answer = next((v for k, v in SCRIPT.items() if k in system), "[]")
        self._json({"choices": [{"message": {"content": answer}}]})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18099
    ThreadingHTTPServer(("127.0.0.1", port), Fake).serve_forever()
