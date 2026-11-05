#!/usr/bin/env python3
"""Tiny mock of the Hugging Face endpoints bizen-ctl uses (test-only, stdlib). Usage: mock-hf.py HF_PORT CDN_PORT
HF_PORT: /api/models (search), /api/models/<id>, /api/models/<id>/tree/main (paginated via Link), /<id>/raw/main/README.md,
         /<id>/resolve/main/<path> (302 to the CDN port). CDN_PORT: the file bytes, Range / If-Range / ETag.
Control (either port): GET /_log -> requests seen [{port, path, auth, range}], POST /_reset, POST /_slow?s=0.2
(seconds slept per 64 KiB chunk of file data, for pause/cancel tests). The gated repo needs 'Bearer hf_testtoken'."""
import hashlib, json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HF_PORT, CDN_PORT = int(sys.argv[1]), int(sys.argv[2])
TOKEN = "hf_testtoken"
STATE = {"slow": float(os.environ.get("MOCK_HF_SLOW", "0")), "log": []}
LOCK = threading.Lock()
CHUNK = 65536
MB = 1024 * 1024

def shards(prefix, n, size):
    return [(f"{prefix}-{i:05d}-of-{n:05d}.gguf", size + i * 1000) for i in range(1, n + 1)]

REPOS = {
    "acme/Tiny-GGUF": {"gated": False, "files": [
        ("Tiny-Q4_K_M.gguf", MB), ("Tiny-Q8_0.gguf", 2 * MB), ("mmproj-F16.gguf", 300000),
        *[("UD-Q3_K_XL/" + n, s) for n, s in shards("Tiny-UD-Q3_K_XL", 3, 2 * MB)],
        ("README.md", 900), (".gitattributes", 200)]},
    "acme/Gated-GGUF": {"gated": "manual", "files": [("Gated-Q4_K_M.gguf", MB)]},
    "acme/Liar-GGUF": {"gated": False, "files": [("Liar-Q4_K_M.gguf", MB)], "tree_extra": 10},
    "acme/NoRange-GGUF": {"gated": False, "files": [("NoRange-Q4_K_M.gguf", 1500000)], "ignore_range": True},
}
MODELS = [  # search index
    {"id": "acme/Tiny-GGUF", "author": "acme", "downloads": 500, "likes": 9, "lastModified": "2026-10-01T00:00:00.000Z", "tags": ["gguf", "license:apache-2.0"], "private": False, "gated": False},
    {"id": "acme/Gated-GGUF", "author": "acme", "downloads": 900, "likes": 3, "lastModified": "2026-09-01T00:00:00.000Z", "tags": ["gguf"], "private": False, "gated": "manual"},
    {"id": "other/Tiny-Chat-GGUF", "author": "other", "downloads": 100, "likes": 40, "lastModified": "2026-10-20T00:00:00.000Z", "tags": ["gguf"], "private": False, "gated": False},
    {"id": "bad id/with space", "downloads": 1},
]
CARD = "---\nlicense: apache-2.0\nbase_model: acme/Tiny\n---\n# Tiny GGUF\n\nA <b>tiny</b> test model. " + "x" * 30000
PAGE = 4

def content(path, off, n):
    pat = hashlib.sha256(path.encode()).digest()
    start = off % 32
    return (pat * (n // 32 + 2))[start:start + n]

def file_size(repo, path):
    for p, s in REPOS[repo]["files"]:
        if p == path:
            return s
    return None

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def reply(self, code, body=b"", ctype="application/json", extra=()):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/_reset":
            with LOCK: STATE["log"].clear()
            return self.reply(200, {"ok": True})
        if u.path == "/_slow":
            STATE["slow"] = float((urllib.parse.parse_qs(u.query).get("s") or ["0"])[0])
            return self.reply(200, {"slow": STATE["slow"]})
        self.reply(404, {"error": "no"})

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        port = self.server.server_address[1]
        if u.path == "/_log":
            return self.reply(200, STATE["log"])
        with LOCK:
            STATE["log"].append({"port": port, "path": u.path, "auth": self.headers.get("Authorization"),
                                 "range": self.headers.get("Range"), "if_range": self.headers.get("If-Range")})
        q = urllib.parse.parse_qs(u.query)
        if port == CDN_PORT:
            return self.cdn(u, q)
        p = urllib.parse.unquote(u.path)
        if p == "/api/models":
            term = (q.get("search") or [""])[0].lower()
            rows = [m for m in MODELS if term in m["id"].lower()]
            sort = (q.get("sort") or ["downloads"])[0]
            if sort not in ("downloads", "likes", "lastModified"):
                return self.reply(400, {"error": "bad sort"})
            rows.sort(key=lambda m: m.get(sort) or 0, reverse=True)
            return self.reply(200, rows[:int((q.get("limit") or ["20"])[0])])
        parts = p.strip("/").split("/")
        if p.startswith("/api/models/") and len(parts) >= 4:
            repo = parts[2] + "/" + parts[3]
            rest = parts[4:]
            return self.api_repo(repo, rest, q)
        if len(parts) >= 4 and parts[2] == "raw" and parts[-1] == "README.md":
            repo = parts[0] + "/" + parts[1]
            if not self.allowed(repo): return
            return self.reply(200, CARD.encode(), "text/plain")
        if len(parts) >= 5 and parts[2] == "resolve":
            repo = parts[0] + "/" + parts[1]
            if not self.allowed(repo): return
            path = "/".join(parts[4:])
            if file_size(repo, path) is None:
                return self.reply(404, {"error": "Entry not found"})
            loc = f"http://127.0.0.1:{CDN_PORT}/blob/{repo}/{path}?sig=abc"
            return self.reply(302, b"", extra=[("Location", loc)])
        self.reply(404, {"error": "Repository not found"})

    def allowed(self, repo):
        r = REPOS.get(repo)
        if r is None:
            self.reply(404, {"error": "Repository not found"})
            return False
        if r["gated"] and self.headers.get("Authorization") != "Bearer " + TOKEN:
            self.reply(401, {"error": "Invalid username or password."})
            return False
        return True

    def api_repo(self, repo, rest, q):
        if not self.allowed(repo): return
        r = REPOS[repo]
        if not rest:
            return self.reply(200, {"id": repo, "author": repo.split("/")[0], "downloads": 500, "likes": 9,
                                    "lastModified": "2026-10-01T00:00:00.000Z", "private": False, "gated": r["gated"],
                                    "tags": ["gguf", "license:apache-2.0"], "cardData": {"license": "apache-2.0", "base_model": "acme/Tiny"},
                                    "gguf": {"architecture": "llama", "total": 123}})
        if rest[0] == "tree":
            ents = [{"type": "directory", "path": "UD-Q3_K_XL"}] if repo == "acme/Tiny-GGUF" else []
            for pth, s in r["files"]:
                e = {"type": "file", "path": pth, "size": 134 if pth.endswith(".gguf") else s}
                if pth.endswith(".gguf"):
                    e["lfs"] = {"oid": "0" * 64, "size": s + r.get("tree_extra", 0), "pointerSize": 134}
                ents.append(e)
            cur = int((q.get("cursor") or ["0"])[0])
            page = ents[cur:cur + PAGE]
            extra = []
            if cur + PAGE < len(ents):
                extra.append(("Link", f'<http://127.0.0.1:{HF_PORT}/api/models/{repo}/tree/main?recursive=1&cursor={cur + PAGE}>; rel="next"'))
            return self.reply(200, page, extra=extra)
        self.reply(404, {"error": "no"})

    def cdn(self, u, q):
        parts = urllib.parse.unquote(u.path).strip("/").split("/")   # blob/<owner>/<name>/<path...>
        repo = parts[1] + "/" + parts[2]
        path = "/".join(parts[3:])
        size = file_size(repo, path)
        if size is None:
            return self.reply(404, {"error": "no"})
        etag = '"' + hashlib.md5(path.encode()).hexdigest() + '"'
        start, code = 0, 200
        rng = self.headers.get("Range")
        if rng and not REPOS[repo].get("ignore_range"):
            ir = self.headers.get("If-Range")
            if not ir or ir == etag:
                try:
                    start = int(rng.split("=")[1].split("-")[0])
                except (ValueError, IndexError):
                    start = 0
                if start >= size:
                    return self.reply(416, b"", extra=[("Content-Range", f"bytes */{size}")])
                code = 206
        n = size - start
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(n))
        self.send_header("ETag", etag)
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        self.end_headers()
        sent = 0
        try:
            while sent < n:
                k = min(CHUNK, n - sent)
                self.wfile.write(content(path, start + sent, k))
                self.wfile.flush()
                sent += k
                if STATE["slow"]:
                    time.sleep(STATE["slow"])
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

class S(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    a = S(("127.0.0.1", HF_PORT), H)
    b = S(("127.0.0.1", CDN_PORT), H)
    threading.Thread(target=b.serve_forever, daemon=True).start()
    a.serve_forever()
