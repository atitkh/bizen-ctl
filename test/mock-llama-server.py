#!/usr/bin/env python3
"""Fake llama-server for testing bizen-ctl: same CLI shape, /health, chat (plain + SSE)."""
import json, sys, time, http.server, socketserver

args = sys.argv[1:]
opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
port, key = int(opt("--port")), opt("--api-key")
print("mock llama-server args:", " ".join(args), flush=True)
t0 = time.time()


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def j(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            return self.j(200 if time.time() - t0 > 2 else 503, {"status": "ok"})
        if self.headers.get("Authorization") != f"Bearer {key}":
            return self.j(401, {"error": "bad key"})
        self.j(200, {"data": [{"id": opt("--alias")}]})
    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {key}":
            return self.j(401, {"error": "bad key"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        print("prompt eval time =      50.00 ms /    20 tokens (    2.50 ms per token,   400.00 tokens per second)", flush=True)
        print("       eval time =    1000.00 ms /    32 tokens (   31.25 ms per token,    32.00 tokens per second)", flush=True)
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            for w in ["Hello", " from", " mock"]:
                d = f"data: {json.dumps({'choices': [{'delta': {'content': w}}]})}\n\n".encode()
                self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n"); self.wfile.flush(); time.sleep(0.3)
            d = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n0\r\n\r\n"); return
        self.j(200, {"choices": [{"message": {"content": "Hello from mock"}}],
                     "timings": {"predicted_per_second": 32.0, "prompt_per_second": 400.0, "predicted_n": 3}})


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
S(("127.0.0.1", port), H).serve_forever()
