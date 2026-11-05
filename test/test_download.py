#!/usr/bin/env python3
"""Hugging Face browse + downloader: validation, free-space refusal, Range resume, bearer token never leaves the
endpoint's origin, size mismatch, queue limit, persistence, token never in files/log/API output, remove confinement,
plus a fake-DOM render of the Browse / Downloads views. Talks only to test/mock-hf.py on local ports."""
import contextlib, hashlib, io, json, os, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def wait_for(pred, timeout=20.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return False

T = tempfile.mkdtemp(prefix="bizendl.")
HF, CDN = free_port(), free_port()
mock = subprocess.Popen([sys.executable, os.path.join(HERE, "mock-hf.py"), str(HF), str(CDN)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
EP = f"http://127.0.0.1:{HF}"
def mget(path, port=HF):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5))
def mpost(path):
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{HF}{path}", data=b"", method="POST"), timeout=5))
assert wait_for(lambda: socket.socket().connect_ex(("127.0.0.1", HF)) == 0 and socket.socket().connect_ex(("127.0.0.1", CDN)) == 0, 10), "mock did not start"

def expected(path, size):
    pat = hashlib.sha256(path.encode()).digest()
    return (pat * (size // 32 + 2))[:size]

class Cfg:
    def __init__(self, d, path): self.data, self.path = d, path
    def get(self, k): return self.data.get(k)

TOKEN = "hf_testtoken"
def mkcfg(name, token="", **kw):
    root = os.path.join(T, name); dl = os.path.join(root, "dl"); os.makedirs(dl)
    data = {"download_dir": dl, "model_dirs": [dl], "hf_endpoint": EP, "hf_token": token, "min_free_gib_for_download": 0, "profiles": []}
    data.update(kw)
    return Cfg(data, os.path.join(root, "config.json")), dl

def sizes(cfg, repo):
    return {f["path"]: f["bytes"] for f in b.hf_repo_raw(cfg.data, repo)["files"]}

def plan(cfg, repo, paths):
    return b.dl_plan(cfg.data, repo, paths, sizes(cfg, repo))

def refused(fn, code=None, text=None):
    try:
        fn()
    except b.DlError as e:
        return (code is None or e.code == code) and (text is None or text in e.msg)
    return False

def get(dl, did):
    return next((x for x in dl.list() if x["id"] == did), None)

def finish(dl, did, timeout=30):
    wait_for(lambda: get(dl, did)["state"] in ("done", "error", "canceled", "paused"), timeout)
    return get(dl, did)

try:
    # ---- validation
    check("repo id: accepts owner/name", all(b.hf_valid_repo(r) for r in ("acme/Tiny-GGUF", "a/b.c_d", "Org1/Model-7B.v2")))
    check("repo id: rejects traversal, extra slashes, dots-only parts, newline, spaces",
          not any(b.hf_valid_repo(r) for r in ("../x", "a/..", "a/.", "a/b/c", "/a/b", "a b/c", ".a/b", "a/", "a/b\n", "a/b c", "", None, "a/b?x=1", "a/b#h")))
    check("file name: accepts relative .gguf paths", all(b.hf_valid_file(p) for p in ("x.gguf", "UD-Q3/x-00001-of-00003.gguf", "a/b/c.GGUF")))
    check("file name: rejects traversal, absolute, backslash, double slash, wrong suffix, newline",
          not any(b.hf_valid_file(p) for p in ("../x.gguf", "/etc/x.gguf", "a/../b.gguf", "x.txt", "a\\b.gguf", "a//b.gguf", "x.gguf\n", "x.gguf.part", "..gguf", "", None, "a/./b.gguf", "~/x.gguf", "x y.gguf")))

    # ---- plan: destination layout and set rules
    cfg, dl = mkcfg("plan")
    p = plan(cfg, "acme/Tiny-GGUF", ["UD-Q3_K_XL/Tiny-UD-Q3_K_XL-0000%d-of-00003.gguf" % i for i in (1, 2, 3)] + ["mmproj-F16.gguf"])
    check("plan: lands in download_dir/<repo name>/<quant folder>, mmproj beside it", p["dest_dir"] == os.path.join(os.path.realpath(dl), "Tiny-GGUF", "UD-Q3_K_XL")
          and p["name"] == "Tiny-UD-Q3_K_XL" and len(p["files"]) == 4 and p["files"][3]["path"] == "mmproj-F16.gguf")
    p = plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"])
    check("plan: root files go to 'files'", p["dest_dir"].endswith(os.path.join("Tiny-GGUF", "files")) and p["files"][0]["bytes"] == 2 * 1024 * 1024)
    sh = ["UD-Q3_K_XL/Tiny-UD-Q3_K_XL-0000%d-of-00003.gguf" % i for i in (1, 2, 3)]
    check("plan: refuses a partial shard set", refused(lambda: plan(cfg, "acme/Tiny-GGUF", sh[:2]), 400, "one quant set"))
    check("plan: refuses two different quants", refused(lambda: plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf", "Tiny-Q4_K_M.gguf"]), 400))
    check("plan: refuses traversal / absolute / non-gguf file names", all(refused(lambda x=x: plan(cfg, "acme/Tiny-GGUF", [x]), 400) for x in ("../Tiny-Q8_0.gguf", "/etc/passwd.gguf", "README.md", "UD-Q3_K_XL/../Tiny-Q8_0.gguf")))
    check("plan: refuses a file that is not in the repo (404) and a bad repo id (400)", refused(lambda: plan(cfg, "acme/Tiny-GGUF", ["Nope-Q4.gguf"]), 404)
          and refused(lambda: b.dl_plan(cfg.data, "../etc/x", ["a.gguf"], {"a.gguf": 1}), 400) and refused(lambda: b.dl_plan(cfg.data, "a/..", ["a.gguf"], {"a.gguf": 1}), 400))
    check("plan: refuses mmproj-only and empty requests", refused(lambda: plan(cfg, "acme/Tiny-GGUF", ["mmproj-F16.gguf"]), 400) and refused(lambda: plan(cfg, "acme/Tiny-GGUF", []), 400))
    out = os.path.join(T, "outside"); os.makedirs(out)
    os.symlink(out, os.path.join(dl, "Tiny-GGUF"))
    check("plan: a symlinked repo folder that points outside download_dir is refused", refused(lambda: plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"]), 400, "leave"))
    os.unlink(os.path.join(dl, "Tiny-GGUF"))
    cfg_empty, _ = mkcfg("noroot", download_dir="", model_dirs=[])
    check("plan: no download directory configured -> 400", refused(lambda: b.dl_plan(cfg_empty.data, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"], {"Tiny-Q8_0.gguf": 5}), 400, "download directory"))

    # ---- free space and writability
    cfg, dl = mkcfg("space", min_free_gib_for_download=20)
    d = b.Downloader(cfg)
    real_free = b._free_bytes
    b._free_bytes = lambda path: 5 * 2**30
    check("free space: refuses with 409 'not enough free space: need X GiB, have Y GiB'",
          refused(lambda: d.add(plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"])), 409, "not enough free space: need 20.0 GiB, have 5.0 GiB"))
    b._free_bytes = real_free
    cfg.data["min_free_gib_for_download"] = 0
    if os.geteuid() != 0:
        ro = os.path.join(T, "ro"); os.makedirs(ro); os.chmod(ro, 0o500)
        cfg_ro, _ = mkcfg("rodir", download_dir=ro)
        check("non-writable download dir -> 400", refused(lambda: b.Downloader(cfg_ro).add(plan(cfg_ro, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"])), 400, "not writable"))
        os.chmod(ro, 0o700)
    d.shutdown()

    # ---- Range resume continuing a hand-made .part, with the redirect to the CDN port
    mpost("/_reset"); mpost("/_slow?s=0")
    cfg, dl = mkcfg("resume")
    dest = os.path.join(os.path.realpath(dl), "Tiny-GGUF", "files"); os.makedirs(dest)
    full = expected("Tiny-Q8_0.gguf", 2 * 1024 * 1024)
    with open(os.path.join(dest, "Tiny-Q8_0.gguf.part"), "wb") as f:
        f.write(full[:700000])
    d = b.Downloader(cfg)
    it = d.add(plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"]))
    check("add: an existing .part is counted as already done", it["bytes_done"] == 700000)
    it = finish(d, it["id"])
    final = os.path.join(dest, "Tiny-Q8_0.gguf")
    check("resume: finishes and the file is byte-identical", it["state"] == "done" and os.path.isfile(final) and open(final, "rb").read() == full and not os.path.exists(final + ".part"))
    log = mget("/_log")
    cdn = [e for e in log if e["port"] == CDN and "Q8_0" in e["path"]]
    check("resume: the CDN request carried Range: bytes=700000-", cdn and cdn[0]["range"] == "bytes=700000-")
    check("resume: finished entry has bytes_done == bytes_total and 100%", it["bytes_done"] == it["bytes_total"] == 2 * 1024 * 1024 and it["pct"] == 100.0)
    check("add: refuses a set that is already on disk (409)", refused(lambda: d.add(plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"])), 409, "already downloaded"))
    d.shutdown()

    # ---- server ignores Range -> restart from zero
    cfg, dl = mkcfg("norange")
    dest = os.path.join(os.path.realpath(dl), "NoRange-GGUF", "files"); os.makedirs(dest)
    with open(os.path.join(dest, "NoRange-Q4_K_M.gguf.part"), "wb") as f:
        f.write(b"garbage" * 1000)
    d = b.Downloader(cfg)
    it = finish(d, d.add(plan(cfg, "acme/NoRange-GGUF", ["NoRange-Q4_K_M.gguf"]))["id"])
    check("Range ignored by the server: restarts and still ends byte-identical", it["state"] == "done" and open(os.path.join(dest, "NoRange-Q4_K_M.gguf"), "rb").read() == expected("NoRange-Q4_K_M.gguf", 1500000))
    d.shutdown()

    # ---- pause / resume with a validator, then a changed validator
    mpost("/_reset"); mpost("/_slow?s=0.1")
    cfg, dl = mkcfg("pause")
    d = b.Downloader(cfg)
    it = d.add(plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"]))
    check("pause: transfer starts and reports progress", wait_for(lambda: get(d, it["id"])["bytes_done"] > 200000, 15) and get(d, it["id"])["state"] == "running")
    check("status: downloads_active is set while running", (d.active() or {}).get("id") == it["id"] and set(d.active()) == {"id", "name", "pct", "mbps", "eta_s"})
    t0 = time.time(); r = d.pause(it["id"]); took = time.time() - t0
    part = os.path.join(os.path.realpath(dl), "Tiny-GGUF", "files", "Tiny-Q8_0.gguf.part")
    check("pause: stops quickly, keeps the .part, state paused", r["state"] == "paused" and took < 4 and os.path.getsize(part) > 100000 and d.active() is None)
    before = os.path.getsize(part)
    d.items[0].validators["Tiny-Q8_0.gguf"] = '"stale-etag"'          # server will answer 200 to If-Range: restart
    mpost("/_reset"); mpost("/_slow?s=0")
    d.resume(it["id"])
    it2 = finish(d, it["id"])
    c = [e for e in mget("/_log") if e["port"] == CDN and "Q8_0" in e["path"]]
    check("resume: Range + If-Range sent; a changed validator makes the server resend everything and we restart",
          c and c[0]["range"] == f"bytes={before}-" and c[0]["if_range"] == '"stale-etag"' and it2["state"] == "done"
          and open(os.path.join(os.path.dirname(part), "Tiny-Q8_0.gguf"), "rb").read() == full)
    d.shutdown()

    # ---- pause + resume with the right validator continues (206)
    mpost("/_reset"); mpost("/_slow?s=0.1")
    cfg, dl = mkcfg("pause2")
    d = b.Downloader(cfg)
    it = d.add(plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"]))
    wait_for(lambda: get(d, it["id"])["bytes_done"] > 200000, 15)
    d.pause(it["id"]); mpost("/_slow?s=0"); mpost("/_reset")
    d.resume(it["id"]); it2 = finish(d, it["id"])
    c = [e for e in mget("/_log") if e["port"] == CDN and "Q8_0" in e["path"]]
    fin = os.path.join(os.path.realpath(dl), "Tiny-GGUF", "files", "Tiny-Q8_0.gguf")
    check("resume with the stored ETag continues with a 206 and ends byte-identical", it2["state"] == "done" and c and c[0]["if_range"] and open(fin, "rb").read() == full)
    d.shutdown()

    # ---- bearer token: sent to the endpoint's origin, never to the CDN
    mpost("/_reset")
    cfg, dl = mkcfg("gated", token=TOKEN)
    d = b.Downloader(cfg)
    it = finish(d, d.add(plan(cfg, "acme/Gated-GGUF", ["Gated-Q4_K_M.gguf"]))["id"])
    lg = mget("/_log")
    hf_auth = [e["auth"] for e in lg if e["port"] == HF and "resolve" in e["path"]]
    cdn_auth = [e["auth"] for e in lg if e["port"] == CDN]
    check("gated download works with the token", it["state"] == "done")
    check("token went to the endpoint host on the resolve request", hf_auth and all(a == "Bearer " + TOKEN for a in hf_auth))
    check("token was stripped on the redirect to the other host (CDN)", cdn_auth and all(a is None for a in cdn_auth))
    # without a token the gated download errors readably and keeps nothing
    cfg_nt, dl_nt = mkcfg("gated-notoken")
    d2 = b.Downloader(cfg_nt)
    try:
        d2.add(plan(cfg_nt, "acme/Gated-GGUF", ["Gated-Q4_K_M.gguf"])); sized = True
    except b.HFError as e:
        sized = False; msg = e.msg
    check("gated repo without a token: readable 'set a token in System' error", not sized and "set a token in System" in msg)
    cfg_bad, _ = mkcfg("gated-bad", token="wrong-token-xyz")
    try:
        b.hf_repo_raw(cfg_bad.data, "acme/Gated-GGUF"); msg2 = ""
    except b.HFError as e:
        msg2 = e.msg
    check("gated repo with a wrong token: 'token rejected', token text not echoed", "token rejected" in msg2 and "wrong-token-xyz" not in msg2)
    d2.shutdown()

    # ---- size mismatch -> error, .part kept
    cfg, dl = mkcfg("liar")
    d = b.Downloader(cfg)
    it = finish(d, d.add(plan(cfg, "acme/Liar-GGUF", ["Liar-Q4_K_M.gguf"]))["id"])
    part = os.path.join(os.path.realpath(dl), "Liar-GGUF", "files", "Liar-Q4_K_M.gguf.part")
    check("size mismatch: state error with a readable message, final file absent, .part kept",
          it["state"] == "error" and "size mismatch" in it["error"] and os.path.isfile(part) and not os.path.exists(part[:-5]))
    # ---- remove with delete_partial
    r = d.remove(it["id"], True)
    check("remove + delete_partial: deletes the .part and the emptied folders", r["removed_partial"] == 1 and not os.path.exists(part) and not os.path.exists(os.path.dirname(os.path.dirname(part))) and os.path.isdir(dl))
    check("remove: unknown id -> 404", refused(lambda: d.remove(it["id"]), 404))
    d.shutdown()

    # ---- queue limit (1 running + 3 queued, the 5th is refused), cancel keeps .part, FIFO order
    mpost("/_reset"); mpost("/_slow?s=0.3")
    cfg, dl = mkcfg("queue")
    d = b.Downloader(cfg)
    a = d.add(plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q8_0.gguf"]))
    wait_for(lambda: get(d, a["id"])["state"] == "running", 10)
    q = [d.add(plan(cfg, "acme/Tiny-GGUF", x)) for x in (["Tiny-Q4_K_M.gguf"], sh)]
    q.append(d.add(plan(cfg, "acme/NoRange-GGUF", ["NoRange-Q4_K_M.gguf"])))
    check("queue: one running, three waiting in FIFO order", [x["state"] for x in d.list()] == ["running", "queued", "queued", "queued"])
    check("queue: the fourth waiting download is refused (409)", refused(lambda: d.add(plan(cfg, "acme/Liar-GGUF", ["Liar-Q4_K_M.gguf"])), 409, "queue is full"))
    check("queue: the same download twice is refused (409)", refused(lambda: d.add(plan(cfg, "acme/Tiny-GGUF", ["Tiny-Q4_K_M.gguf"])), 409, "already in the list"))
    check("remove: refuses a queued/running entry (409)", refused(lambda: d.remove(a["id"]), 409))
    r = d.cancel(a["id"])
    part = os.path.join(os.path.realpath(dl), "Tiny-GGUF", "files", "Tiny-Q8_0.gguf.part")
    check("cancel: running download stops, state canceled, .part kept", r["state"] == "canceled" and os.path.isfile(part))
    check("queue: next in line (FIFO) starts after the cancel", wait_for(lambda: get(d, q[0]["id"])["state"] == "running", 10) and get(d, q[1]["id"])["state"] == "queued")
    mpost("/_slow?s=0")
    # persistence: a second Downloader reading the same downloads.json sees running/queued as paused and does not start them
    snap = json.load(open(d.path))
    d_re = b.Downloader(cfg)
    st = {x["id"]: x["state"] for x in d_re.list()}
    check("persistence: downloads.json written, running and queued entries come back paused",
          any(x["state"] == "running" for x in snap["items"]) and st.get(q[0]["id"]) == "paused" and st.get(q[1]["id"]) == "paused" and st.get(a["id"]) == "canceled")
    time.sleep(0.8)
    check("persistence: nothing auto-starts after a reload", d_re.cur is None and d_re.active() is None)
    check("persistence: a reloaded paused entry carries its partial progress and a note",
          next(x for x in d_re.list() if x["id"] == a["id"])["bytes_done"] == os.path.getsize(part))
    d_re.shutdown(); d.shutdown()

    # ---- token never in downloads.json / log / API output
    blob = open(os.path.join(T, "gated", "downloads.json")).read() + json.dumps(b.Downloader.list(d)) + json.dumps(it)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg_t, _ = mkcfg("leak", token=TOKEN)
        dt = b.Downloader(cfg_t)
        try:
            blob += json.dumps(finish(dt, dt.add(plan(cfg_t, "acme/Gated-GGUF", ["Gated-Q4_K_M.gguf"]))["id"]))
            blob += json.dumps(b.hf_search(cfg_t.data, "tiny", "downloads", 5)) + json.dumps(b.hf_repo_view(cfg_t.data, "acme/Gated-GGUF"))
            blob += open(dt.path).read()
        finally:
            dt.shutdown()
    check("token never appears in downloads.json, list output, repo/search output or the log", TOKEN not in blob and TOKEN not in buf.getvalue() and "Bearer" not in blob)
    check("public_config / public_settings still hide the token", TOKEN not in json.dumps(b.public_config(cfg_t.data)) and TOKEN not in json.dumps(b.public_settings(cfg_t.data)))

    # ---- remove confinement: a tampered downloads.json cannot make remove delete outside download_dir
    cfg, dl = mkcfg("confine")
    outside = os.path.join(T, "victim"); os.makedirs(outside)
    victim = os.path.join(outside, "x.gguf.part"); open(victim, "w").write("precious")
    secret = os.path.join(T, "secret.txt"); open(secret, "w").write("precious")
    linkdir = os.path.join(dl, "linkdir"); os.symlink(outside, linkdir)
    realdir = os.path.join(dl, "R", "files"); os.makedirs(realdir); os.symlink(secret, os.path.join(realdir, "y.gguf.part"))
    rows = [{"id": "aaaaaaa1", "repo": "acme/Tiny-GGUF", "name": "x", "state": "canceled", "files": [{"path": "x.gguf", "bytes": 5, "done": 0}], "dest_dir": outside},
            {"id": "aaaaaaa2", "repo": "acme/Tiny-GGUF", "name": "x", "state": "canceled", "files": [{"path": "x.gguf", "bytes": 5, "done": 0}], "dest_dir": linkdir},
            {"id": "aaaaaaa3", "repo": "acme/Tiny-GGUF", "name": "y", "state": "canceled", "files": [{"path": "y.gguf", "bytes": 5, "done": 0}], "dest_dir": realdir},
            {"id": "aaaaaaa4", "repo": "acme/Tiny-GGUF", "name": "z", "state": "canceled", "files": [{"path": "../../z.gguf", "bytes": 5, "done": 0}], "dest_dir": realdir}]
    json.dump({"version": 1, "items": rows}, open(os.path.join(os.path.dirname(cfg.path), "downloads.json"), "w"))
    d = b.Downloader(cfg)
    check("persistence: an entry with a traversal file name in downloads.json is dropped on load", sorted(x["id"] for x in d.list()) == ["aaaaaaa1", "aaaaaaa2", "aaaaaaa3"])
    for i in ("aaaaaaa1", "aaaaaaa2", "aaaaaaa3"):
        d.remove(i, True)
    check("remove delete_partial: never touches files outside download_dir (outside dir, symlinked dir)", open(victim).read() == "precious")
    check("remove delete_partial: a .part that is a symlink is left alone, target intact", os.path.islink(os.path.join(realdir, "y.gguf.part")) and open(secret).read() == "precious")
    d.shutdown()

    # ---- search / repo shape and untrusted rows
    cfg, dl = mkcfg("shape")
    r = b.hf_search(cfg.data, "tiny", "downloads", 20)
    check("search: rows have the documented fields, invalid repo ids are dropped",
          [m["id"] for m in r["results"]] == ["acme/Tiny-GGUF", "other/Tiny-Chat-GGUF"] and set(r["results"][0]) == {"id", "author", "downloads", "likes", "last_modified", "tags", "private", "gated"})
    check("search: second identical call is served from the 60 s cache", b.hf_search(cfg.data, "tiny", "downloads", 20)["cached"] is True)
    v = b.hf_repo_view(cfg.data, "acme/Tiny-GGUF")
    s = {x["name"]: x for x in v["sets"]}
    check("repo: three quant sets, shards collapsed, sizes from lfs.size, mmproj attached",
          sorted(s) == ["Tiny-Q4_K_M", "Tiny-Q8_0", "Tiny-UD-Q3_K_XL"] and s["Tiny-UD-Q3_K_XL"]["shards"] == 3 and s["Tiny-UD-Q3_K_XL"]["bytes"] == 6297456
          and s["Tiny-Q4_K_M"]["bytes"] == 1048576 and s["Tiny-Q4_K_M"]["mmproj"]["path"] == "mmproj-F16.gguf" and s["Tiny-UD-Q3_K_XL"]["quant"] == "UD-Q3_K_XL")
    check("repo: paginated tree was followed (all files present)", len(v["sets"]) == 3 and all(len(x["files"]) == x["shards"] for x in v["sets"]))
    check("repo: license / base model / architecture hints, card stripped of front matter and cut at 20000 chars",
          v["license"] == "apache-2.0" and v["base_model"] == ["acme/Tiny"] and v["architecture"] == "llama" and v["card"].startswith("# Tiny GGUF") and len(v["card"]) == 20000 and v["card_truncated"])
    check("repo: every set carries the library's fit verdict and already_downloaded", all("verdict" in x["fit"] and x["already_downloaded"] is False for x in v["sets"]))
    errs = {}
    for repo in ("acme/Nope-GGUF", "acme/Gated-GGUF"):
        try:
            b.hf_repo_view(cfg.data, repo)
        except b.HFError as e:
            errs[repo] = (e.code, e.msg)
    check("repo: 404 and gated 401 both say 'repo not found or gated (set a token in System)'", all("set a token in System" in m for _, m in errs.values()) and len(errs) == 2)
    cfg_down = Cfg(dict(cfg.data, hf_endpoint=f"http://127.0.0.1:{free_port()}"), cfg.path)
    try:
        b.hf_search(cfg_down.data, "x", "likes", 5); e2 = None
    except b.HFError as e:
        e2 = e
    check("network failure -> readable HFError (502), no traceback", e2 is not None and e2.code == 502 and "could not reach" in e2.msg)
    already, _ = mkcfg("already")
    dest = os.path.join(os.path.realpath(already.data["download_dir"]), "Tiny-GGUF", "files"); os.makedirs(dest)
    open(os.path.join(dest, "Tiny-Q4_K_M.gguf"), "wb").write(b"x" * 1000)
    v2 = b.hf_repo_view(already.data, "acme/Tiny-GGUF")
    check("repo: already_downloaded true for a set whose name is in the library scan", {x["name"]: x["already_downloaded"] for x in v2["sets"]}["Tiny-Q4_K_M"] is True)
    check("group sets: incomplete shard list is flagged, not collapsed away",
          not b.hf_group_sets([{"path": "m-00001-of-00003.gguf", "bytes": 1}, {"path": "m-00003-of-00003.gguf", "bytes": 1}])[0]["complete"])

    # ---- UI: fake-DOM render of Browse + Downloads, untrusted text stays text
    if shutil.which("node"):
        html = open(os.path.join(HERE, "..", "ui.html")).read()
        js = html[html.index("<script>") + 8:html.rindex("</script>")]
        sub = js[js.index("let libSub = 'installed'"):js.index("/* ---- Library tab")]
        harness = r'''
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.className = ''; this._text = ''; this.style = {}; this.dataset = {}; this.disabled = false; this.title = ''; this.parent = null;
    this.classList = { toggle() {}, add() {}, remove() {} }; }
  set textContent(v) { this._text = String(v); this.children = []; } get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(v) { this._html = v; this.children = []; this._text = ''; } get innerHTML() { return this._html || ''; }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  replaceChild(n, o) { const i = this.children.indexOf(o); this.children[i] = n; n.parent = this; return o; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); }
  querySelectorAll() { return []; }
}
const els = {}; const $ = id => els[id] || (els[id] = new El('#' + id));
const document = { createElement: t => new El(t), querySelectorAll: () => [], hidden: false };
const tree = e => [e].concat(...e.children.map(tree));
const toasts = []; const toast = (m, e) => toasts.push([m, !!e]); const location = { hash: '' };
const fmtDur = s => s == null ? '–' : s + 's'; let curTab = 'library', sysInfo = { settings: { has_hf_token: false }, disks: [] };
const LIB_FIT = { 'fits-vram': ['fits VRAM', 'ok'], 'needs-ram': ['needs RAM', 'warn'], 'too-big': ['too big', 'bad'] };
let libData = null; const loadLibrary = () => {}; const libAddProfile = async id => { toasts.push(['profile ' + id, false]); }; const setInterval = () => 1;
let calls = [], reply = async () => ({}); const api = async (p, o) => { calls.push([p, o && o.body ? JSON.parse(o.body) : null]); return reply(p, o); };
const confirm = () => true;
''' + sub + r'''
(async () => {
const out = {}; const evil = '<img src=x onerror=alert(1)>';
hfRes = { results: [{ id: 'acme/Tiny-GGUF', downloads: 15300, likes: 9, last_modified: '2026-10-01T00:00:00Z', gated: 'manual', private: false, tags: [] }, { id: 'o/Plain', downloads: 3, likes: 0, last_modified: null, gated: false, private: false, tags: [] }], cached: false };
hfRender();
out.cards = tree($('hfResults')).filter(e => e.className.startsWith('hfcard')).length; out.gated = tree($('hfResults')).filter(e => e.className === 'chip warn').map(e => e.textContent);
out.dl = tree($('hfResults')).map(e => e.textContent).join('|').includes('15.3k');
hfRepo = { id: 'acme/Tiny-GGUF', downloads: 1, likes: 2, license: evil, base_model: ['b/' + evil], architecture: 'llama', gated: 'manual', tree_truncated: false,
  card: evil + '\n<script>alert(2)</script>', card_truncated: true,
  sets: [{ name: 'Tiny-Q4_K_M', quant: 'Q4_K_M', files: [{ path: 'Tiny-Q4_K_M.gguf', bytes: 1048576 }], bytes: 1048576, shards: 1, shards_expected: 1, complete: true, mmproj: { path: 'mmproj-F16.gguf', bytes: 5 }, fit: { verdict: 'fits-vram', text: 'tip' }, already_downloaded: false },
         { name: 'Tiny-UD', quant: 'UD-Q3_K_XL', files: [{ path: 'a/x-00001-of-00003.gguf', bytes: 3 }, { path: 'a/x-00002-of-00003.gguf', bytes: 3 }], bytes: 6e9, shards: 2, shards_expected: 3, complete: false, mmproj: null, fit: { verdict: 'too-big', text: 'big' }, already_downloaded: false },
         { name: 'Have', quant: 'Q8_0', files: [{ path: 'Have-Q8_0.gguf', bytes: 9 }], bytes: 9, shards: 1, shards_expected: 1, complete: true, mmproj: null, fit: { verdict: 'needs-ram', text: 'r' }, already_downloaded: true }] };
hfRenderRepo();
const all = tree($('hfSets')); const btns = all.filter(e => e.tag === 'button');
out.btns = btns.map(b => b.disabled); out.texts = all.map(e => e.textContent).join('|');
out.cardText = $('hfCard').textContent; out.cardHtml = $('hfCard').innerHTML; out.cardTags = tree($('hfCard')).length;
out.chipTexts = $('hfChips').children.map(c => c.textContent); out.chipHtml = $('hfChips').innerHTML; out.noImgEl = tree($('hfChips')).concat(all).every(e => e.tag !== 'img' && e.tag !== 'script');
calls = []; await hfDownload(0, btns[0]);
out.post = calls[0]; out.hash = location.hash;
dlItems = [{ id: 'aaaaaaa1', repo: 'a/b', name: 'M-Q4', state: 'running', files: [{}], bytes_total: 4 * 2 ** 30, bytes_done: 2 ** 30, pct: 25, mbps: 11.4, eta_s: 280, error: null, note: null, dest_dir: '/dl/b/files' },
  { id: 'aaaaaaa2', repo: 'a/c', name: evil, state: 'error', files: [{}, {}], bytes_total: 10, bytes_done: 5, pct: 50, mbps: 0, eta_s: null, error: 'size mismatch ' + evil, note: null, dest_dir: '/dl' },
  { id: 'aaaaaaa3', repo: 'a/d', name: 'Done', state: 'done', files: [{}], bytes_total: 10, bytes_done: 10, pct: 100, mbps: 0, eta_s: null, error: null, note: null, dest_dir: '/dl/d' }];
dlRender();
const cards = $('dlList').children;
const labels = c => tree(c).filter(e => e.tag === 'button').map(e => e.textContent);
out.dlLabels = cards.map(labels); out.num = cards[0].children[2].textContent; out.fill = tree(cards[0]).find(e => e.tag === 'i').style.width;
out.errText = cards[1].children[3].textContent; out.errNoEl = tree(cards[1]).every(e => e.tag !== 'img');
dlItems[0] = Object.assign({}, dlItems[0], { bytes_done: 2 * 2 ** 30, pct: 50 }); const first = cards[0]; dlRender();
out.sameEl = $('dlList').children[0] === first; out.fill2 = tree($('dlList').children[0]).find(e => e.tag === 'i').style.width;
dlItems[0] = Object.assign({}, dlItems[0], { state: 'paused' }); dlRender(); out.afterPause = labels($('dlList').children[0]);
dlItems = []; dlRender(); out.empty = $('dlList').innerHTML.includes('No downloads yet');
reply = async (p) => p === '/api/library' ? { models: [{ id: 'm1', name: 'Done', complete: true, dir: '/dl/d' }, { id: 'm2', name: 'Done', complete: true, dir: '/other' }] } : {};
await dlAddProfile({ name: 'Done', dest_dir: '/dl/d/' }); out.addProf = toasts[toasts.length - 1][0];
console.log(JSON.stringify(out));
})().catch(e => { console.log(JSON.stringify({ error: String(e && e.stack || e) })); });
'''
        hp = os.path.join(T, "h.js"); open(hp, "w").write(harness)
        r = subprocess.run(["node", hp], capture_output=True, text=True, timeout=60)
        try:
            o = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            o = None
            print(r.stderr[-600:])
        check("ui: browse/downloads views render without throwing", o is not None and "error" not in o)
        if o and "error" in o:
            print(o["error"][:700])
        if o and "error" not in o:
            check("ui: result cards with counts, a gated badge only on the gated one", o["cards"] == 2 and o["gated"] == ["gated"] and o["dl"])
            check("ui: quant table: Download disabled for incomplete and already-downloaded sets", o["btns"] == [False, True, True])
            check("ui: fit chips, mmproj chip, already-downloaded badge, partial-shard note", "fits VRAM" in o["texts"] and "too big" in o["texts"] and "+ mmproj" in o["texts"] and "already downloaded" in o["texts"] and "2 of 3 shards" in o["texts"])
            check("ui: model card goes in as text (no markup interpreted, no innerHTML)", o["cardText"].startswith("<img src=x onerror=alert(1)>") and "<script>alert(2)</script>" in o["cardText"] and o["cardHtml"] == "" and o["cardTags"] == 1)
            check("ui: license / base model chips from the network are text only", any("<img" in c for c in o["chipTexts"]) and o["chipHtml"] == "" and o["noImgEl"])
            check("ui: Download posts the repo and the set's files plus its mmproj, then opens Downloads",
                  o["post"][0] == "/api/downloads" and o["post"][1] == {"repo": "acme/Tiny-GGUF", "files": ["Tiny-Q4_K_M.gguf", "mmproj-F16.gguf"]} and o["hash"] == "#library/downloads")
            check("ui: running card shows bytes, percent, MB/s, ETA, and Pause / Cancel", "1.00 GiB / 4.00 GiB" in o["num"] and "25.0%" in o["num"] and "11.4 MB/s" in o["num"] and "ETA 280s" in o["num"] and o["dlLabels"][0] == ["Pause", "Cancel"] and o["fill"] == "25%")
            check("ui: error card: Retry / Remove / delete-partial buttons; error text is text", o["dlLabels"][1] == ["Retry", "Remove", "Remove + delete partial files"] and "size mismatch <img" in o["errText"] and o["errNoEl"])
            check("ui: finished card offers Add as profile and Remove", o["dlLabels"][2] == ["Add as profile", "Remove"])
            check("ui: progress updates in place (same element), buttons rebuilt on a state change", o["sameEl"] and o["fill2"] == "50%" and o["afterPause"] == ["Resume", "Cancel"])
            check("ui: empty state and Add as profile picks the library entry in that folder", o["empty"] and o["addProf"] == "profile m1")
    else:
        print("SKIP ui render (node missing)")
finally:
    mock.terminate()
    try:
        mock.wait(5)
    except Exception:  # noqa: BLE001
        mock.kill()
    for root, dirs, files in os.walk(T):
        for n in dirs:
            try: os.chmod(os.path.join(root, n), 0o700)
            except OSError: pass
    shutil.rmtree(T, ignore_errors=True)
print("FAILED: " + ", ".join(fails) if fails else "ALL download checks passed")
sys.exit(1 if fails else 0)
