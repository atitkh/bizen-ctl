#!/usr/bin/env python3
"""Fake Kev server for testing bizen-ctl: GET /v1/models, POST /v1/systemone (choice questions).
Picks the choice whose description shares a word with the state; 'UNSURE' in the state gives a flat answer."""
import json
import re
import sys
import time
import http.server
import socketserver

port = int(sys.argv[1])
t0 = time.time()
import os
print("mock kev starting on", port, "model", sys.argv[2] if len(sys.argv) > 2 else "?",
      "dtype", os.environ.get("KEV_DTYPE"), "graphs", os.environ.get("KEV_CUDA_GRAPHS"), flush=True)


def words(s):
    return set(re.findall(r"[a-z]{4,}", s.lower()))


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models" and time.time() - t0 > 1.5:
            return self.send(200, {"models": [{"name": "kev-latest"}]})
        self.send(503, {"detail": "loading"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state = req["state"] if isinstance(req["state"], str) else json.dumps(req["state"])
        answers = {}
        for qid, q in req["questions"].items():
            keys = list(q["criteria"])
            if "UNSURE" in state:
                probs = {k: round(1 / len(keys), 3) for k in keys}
                pick = keys[0]
            else:
                sw = words(state)
                score = {k: len(sw & words(str(q["criteria"][k]))) for k in keys}
                pick = max(keys, key=lambda k: score[k])
                rest = 0.2 / max(1, len(keys) - 1)
                probs = {k: (0.8 if k == pick else round(rest, 3)) for k in keys}
            answers[qid] = {"type": "choice", "choice": pick, "confidence": probs[pick], "probabilities": probs}
        self.send(200, {"model": req["model"], "answers": answers, "usage": {}, "latency_ms": 5})


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


S(("127.0.0.1", port), H).serve_forever()
