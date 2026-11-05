#!/usr/bin/env python3
"""In-app update: Updater.check/apply/info against a temp bare 'origin' plus a clone. No network, no systemd."""
import io, json, os, re, subprocess, sys, tempfile, threading, time, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

T = tempfile.mkdtemp(prefix="bizenupd.")
ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
           GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")

def git(cwd, *a):
    r = subprocess.run(["git", "-C", cwd] + list(a), capture_output=True, text=True, env=ENV, timeout=30)
    assert r.returncode == 0, (a, r.stderr)
    return r.stdout.strip()

def commit(repo, path, text, msg):
    with open(os.path.join(repo, path), "w") as f:
        f.write(text)
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", msg)

class Sup:
    def __init__(self, state="stopped"): self.state = state

# origin (bare) + a working clone 'dev' used to publish commits + the 'box' clone under test
origin = os.path.join(T, "origin.git")
subprocess.run(["git", "init", "-q", "--bare", "-b", "main", origin], check=True, env=ENV)
dev = os.path.join(T, "dev")
subprocess.run(["git", "clone", "-q", origin, dev], check=True, env=ENV, capture_output=True)
git(dev, "checkout", "-q", "-B", "main")
commit(dev, "bizenctl.py", "v1\n", "init"); commit(dev, "ui.html", "u1\n", "ui 1")
git(dev, "push", "-q", "-u", "origin", "main")

def fresh_box(name):
    box = os.path.join(T, name)
    subprocess.run(["git", "clone", "-q", origin, box], check=True, env=ENV, capture_output=True)
    return box

def upd(box, **kw):
    # git env for the child processes: identity for merges is not needed (ff only), but keep config isolated
    os.environ.update({k: v for k, v in ENV.items() if k.startswith("GIT_")})
    return b.Updater(box, kw.pop("sup", Sup()), **kw)

box = fresh_box("box")
u = upd(box)

# 1 up to date
info = u.info()
check("info: is_git, short+full commit, branch, upstream", info["is_git"] and len(info["commit"]) == 7 and len(info["commit_full"]) == 40
      and info["branch"] == "main" and info["upstream"] == "origin/main" and info["dirty"] is False)
check("info: remote_host is a host only (local path -> None or no creds)", info["remote_host"] in (None, ""))
r = u.check()
check("up to date: ok, behind 0, no commits", r["ok"] and r["behind"] == 0 and r["ahead"] == 0 and r["commits"] == [] and not r["restart_needed"])
check("info after fetch has last_fetch ts", isinstance(u.info()["last_fetch"], float))

# 2 behind with ui-only change
commit(dev, "ui.html", "u2\n", "ui 2"); commit(dev, "README.md", "doc\n", "docs"); git(dev, "push", "-q")
r = u.check()
check("behind ui-only: ok, 2 commits, no restart", r["ok"] and r["behind"] == 2 and len(r["commits"]) == 2 and not r["restart_needed"]
      and sorted(r["files"]) == ["README.md", "ui.html"] and not r["would_unload"])

# 3 behind with bizenctl.py change; would_unload depends on the supervisor
commit(dev, "bizenctl.py", "v2\n", "py change"); git(dev, "push", "-q")
r = u.check()
check("behind py: restart_needed, model not loaded -> no unload", r["ok"] and r["restart_needed"] and r["model_loaded"] is False and r["would_unload"] is False)
for st, want in (("running", True), ("starting", True), ("stopped", False)):
    rr = upd(box, sup=Sup(st)).check()
    check(f"would_unload with model state {st}: {want}", rr["would_unload"] is want and rr["model_loaded"] is want)

# 4 apply: fast-forward, new hash, restart hook injectable
fired = threading.Event()
u2 = upd(box, sup=Sup("running"), restart_hook=fired.set, delay=0.05)
code, body = u2.apply(None)
check("apply without confirm:true refused (400)", code == 400 and not body["ok"])
before = u2.info()["commit"]
code, body = u2.apply(True)
after = u2.info()["commit"]
check("apply: ff pulled, returns from/to hash", code == 200 and body["ok"] and body["from"] == before and body["to"] == after and before != after)
check("apply: HEAD equals origin/main", git(box, "rev-parse", "HEAD") == git(dev, "rev-parse", "HEAD"))
check("apply: restart_needed + restarting reported", body["restart_needed"] and body["restarting"] and u2.restarting)
check("apply: hook not fired before schedule_restart", not fired.is_set())
u2.schedule_restart()
check("schedule_restart fires the injected hook (no process exit)", fired.wait(3))
code, body = u2.apply(True)
check("apply while restarting refused 409", code == 409 and not body["ok"])

# 5 apply with ui-only change: no restart
box2 = fresh_box("box2")
commit(dev, "ui.html", "u3\n", "ui 3"); git(dev, "push", "-q")
fired2 = threading.Event()
u3 = upd(box2, restart_hook=fired2.set, delay=0.05)
code, body = u3.apply(True)
check("apply ui-only: ok, restart_needed False, restarting False", code == 200 and body["ok"] and not body["restart_needed"] and not body["restarting"] and not u3.restarting)
code, body = u3.apply(True)
check("apply when already up to date: ok no-op", code == 200 and body["ok"] and body["from"] == body["to"])

# untracked runtime files (downloads.json next to config.json) must not count as local changes
box_u = fresh_box("untracked")
open(os.path.join(box_u, "downloads.json"), "w").write("{}\n")
uu = upd(box_u)
check("untracked file is not 'dirty'", uu.info()["dirty"] is False)
check("check ok with an untracked runtime file", uu.check()["ok"] or "local changes" not in (uu.check().get("reason") or ""))

# 6 refusals
box3 = fresh_box("dirty")
commit(dev, "ui.html", "u4\n", "ui 4"); git(dev, "push", "-q")
with open(os.path.join(box3, "ui.html"), "a") as f: f.write("local edit\n")
ud = upd(box3)
r = ud.check()
check("dirty tree refused with readable reason", not r["ok"] and "local changes" in r["reason"])
code, body = ud.apply(True)
check("apply on dirty tree refused 409, tree untouched", code == 409 and not body["ok"] and "local edit" in open(os.path.join(box3, "ui.html")).read())

box4 = fresh_box("diverged")
commit(box4, "local.txt", "x\n", "local commit")
commit(dev, "ui.html", "u5\n", "ui 5"); git(dev, "push", "-q")
ug = upd(box4)
r = ug.check()
check("diverged refused (ahead and behind)", not r["ok"] and "diverged" in r["reason"] and r["ahead"] == 1 and r["behind"] >= 1)
code, body = ug.apply(True)
check("apply on diverged refused 409, HEAD unchanged", code == 409 and git(box4, "log", "-1", "--format=%s") == "local commit")

box5 = fresh_box("ahead")
commit(box5, "local.txt", "x\n", "local only")
r = upd(box5).check()
check("ahead-only refused (nothing to pull)", not r["ok"] and r["ahead"] == 1 and r["behind"] == 0)

box6 = fresh_box("noup")
git(box6, "branch", "--unset-upstream")
r = upd(box6).check()
check("no upstream refused", not r["ok"] and "upstream" in r["reason"])
check("no upstream: info.upstream None", upd(box6).info()["upstream"] is None)

nogit = os.path.join(T, "nogit"); os.makedirs(nogit)
r = upd(nogit).check()
check("not a git dir refused", not r["ok"] and "not a git checkout" in r["reason"])
i = upd(nogit).info()
check("not a git dir: info is_git False, other fields None", i["is_git"] is False and i["commit"] is None and i["upstream"] is None)
sub = os.path.join(box, "subdir"); os.makedirs(sub)
check("subdir of a repo is not treated as the checkout", upd(sub).info()["is_git"] is False)
check("missing dir does not raise", upd(os.path.join(T, "nope")).check()["ok"] is False)

# 7 credentials never leak
box7 = fresh_box("creds")
git(box7, "remote", "set-url", "origin", "https://alice:s3cr3tTOKEN@127.0.0.1:9/nope/repo.git")
buf = io.StringIO()
uc = upd(box7)
with contextlib.redirect_stdout(buf):
    i = uc.info(); r = uc.check(); code, body = uc.apply(True)
blob = json.dumps([i, r, body]) + buf.getvalue()
check("credentials never appear in info/check/apply/log", "s3cr3tTOKEN" not in blob and "alice" not in blob)
check("remote_host is the bare host", i["remote_host"] == "127.0.0.1")
check("unreachable remote: check refused with a git fetch reason", not r["ok"] and "fetch" in r["reason"])
check("redact() strips user:pass@ and user@", b.redact("fatal: https://u:p@h.example/x.git and ssh://git@h/y") == "fatal: https://h.example/x.git and ssh://h/y")
check("host_of: scp style and https with creds", b.Updater.host_of("git@github.com:o/r.git") == "github.com" and b.Updater.host_of("https://u:p@gitea.lan:3000/a/b") == "gitea.lan")

# 8 lock: a second apply during the first is refused
box8 = fresh_box("lock")
commit(dev, "ui.html", "u6\n", "ui 6"); git(dev, "push", "-q")
ul = upd(box8)
gate, started = threading.Event(), threading.Event()
orig = ul.git
def slow_git(*a, **k):
    if a and a[0] == "pull":
        started.set(); gate.wait(10)
    return orig(*a, **k)
ul.git = slow_git
res = {}
t = threading.Thread(target=lambda: res.update(first=ul.apply(True)))
t.start()
check("first apply reached git pull", started.wait(10))
code2, body2 = ul.apply(True)
check("concurrent apply refused 409 (lock)", code2 == 409 and "already running" in body2["reason"])
check("info.busy true while applying", ul.info()["busy"] is True)
gate.set(); t.join(15)
check("first apply still completes", res["first"][0] == 200 and res["first"][1]["ok"])
check("lock released afterwards", not ul.lock.locked())

# 9 git never hangs on a prompt and has a timeout
ut = upd(box)
t0 = time.time()
rc, out, err = ut.git("fetch", "--quiet", timeout=0.001)
check("git timeout returns rc 124 instead of raising", rc in (124, 0) and time.time() - t0 < 5)

# 10 metrics: host_free_mib from node data
b._numa_save = b.numa_meminfo
b.numa_meminfo = lambda: [{"node": 0, "total_mib": 1000, "free_mib": 100, "used_mib": 900, "cached_mib": 300, "avail_mib": 400},
                          {"node": 1, "total_mib": 1000, "free_mib": 50, "used_mib": 950, "cached_mib": 10, "avail_mib": 60}]
check("meminfo.host_free_mib = sum of node avail_mib", b.meminfo().get("host_free_mib") == 460)
b.numa_meminfo = lambda: []
check("meminfo.host_free_mib None without node data", b.meminfo().get("host_free_mib") is None)
b.numa_meminfo = b._numa_save

print("ALL update checks passed" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
