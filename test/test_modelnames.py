#!/usr/bin/env python3
"""Unit checks: model-name resolution (pure), switch decision (pure), the busy-wait loop with a fake supervisor,
and the two new config keys (defaults, validation)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

N = ["alpha", "Beta-2"]
R = lambda m, router=False, strict=True, alias="auto": b.resolve_model(m, N, alias, router, strict)
check("profile name", R("alpha") == ("profile", "alpha"))
check("profile name wins with router on", R("Beta-2", router=True) == ("profile", "Beta-2"))
check("case sensitive: ALPHA is unknown (strict)", R("ALPHA") == ("unknown", "ALPHA"))
check("case sensitive: AUTO is unknown (strict)", R("AUTO") == ("unknown", "AUTO"))
check("alias, router on -> router", R("auto", router=True) == ("router", None))
check("alias, router off -> auto", R("auto", router=False) == ("auto", None))
check("custom alias honoured", R("smart", router=True, alias="smart") == ("router", None) and R("auto", alias="smart") == ("unknown", "auto"))
for label, m in (("missing", None), ("empty", ""), ("int", 5), ("list", ["alpha"]), ("dict", {})):
    check(f"{label} model -> auto (never the router)", R(m, router=True) == ("auto", None) and R(m) == ("auto", None))
check("unknown, strict -> unknown", R("gpt-4") == ("unknown", "gpt-4"))
check("unknown, lenient -> auto", R("gpt-4", strict=False) == ("auto", None))
check("lenient still resolves profile names", R("alpha", strict=False) == ("profile", "alpha"))
check("whitespace is not trimmed", R(" alpha") == ("unknown", " alpha"))

D = b.switch_decision
check("nothing loaded -> start", D(False, None, "stopped", 0, "alpha") == "start")
check("failed -> start", D(True, "beta", "failed", 0, "alpha") == "start")
check("already loaded -> serve", D(True, "alpha", "running", 5, "alpha") == "serve")
check("loading the same one -> serve", D(True, "alpha", "starting", 0, "alpha") == "serve")
check("other loaded, idle -> switch", D(True, "beta", "running", 0, "alpha") == "switch")
check("other loaded, busy -> busy", D(True, "beta", "running", 2, "alpha") == "busy")
check("other is loading for someone -> busy (even with 0 others)", D(True, "beta", "starting", 0, "alpha") == "busy")
check("other is stopping -> busy", D(True, "beta", "stopping", 0, "alpha") == "busy")


class Clock:
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, s): self.t += s


class Fake:
    """claim_named scripted by a list of outcomes; records waiter marks."""
    def __init__(self, outcomes):
        self.outcomes, self.calls, self.marks, self.waiters = list(outcomes), [], [], 0
    def claim_named(self, profile, waiting=False):
        self.calls.append((profile, waiting))
        return self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
    def wait_mark(self, d):
        self.marks.append(d); self.waiters += d


c = Clock(); f = Fake([("ok", 0)])
check("idle: switches at once, no waiting", b.acquire_profile(f, "alpha", 120, 0.5, c.now, c.sleep) == ("ok", 0) and c.t == 0 and f.marks == [])
c = Clock(); f = Fake([("busy", 1), ("busy", 1), ("ok", 0)])
check("busy then idle: switches after the others finish", b.acquire_profile(f, "alpha", 120, 0.5, c.now, c.sleep) == ("ok", 0) and c.t == 1.0)
check("waiter is marked while parked and unmarked after", f.marks == [1, -1] and f.waiters == 0 and f.calls[0][1] is False and f.calls[1][1] is True)
c = Clock(); f = Fake([("busy", 2)])
r = b.acquire_profile(f, "alpha", 3, 0.5, c.now, c.sleep)
check("busy past switch_wait_s -> busy with the count", r == ("busy", 2) and 3 <= c.t <= 3.5)
check("waiter unmarked after giving up", f.waiters == 0 and f.marks == [1, -1])
c = Clock(); f = Fake([("busy", 1)])
check("switch_wait_s=0: one attempt, no sleeping", b.acquire_profile(f, "alpha", 0, 0.5, c.now, c.sleep) == ("busy", 1) and c.t == 0 and len(f.calls) == 1)
c = Clock(); f = Fake([("busy", 1), ("busy", 1)])
class Boom(Fake):
    def claim_named(self, p, w=False):
        if self.calls: raise RuntimeError("x")
        self.calls.append(1); return ("busy", 1)
f = Boom([]); 
try:
    b.acquire_profile(f, "alpha", 5, 0.5, c.now, c.sleep)
except RuntimeError:
    pass
check("waiter unmarked even when claim raises", f.waiters == 0)


# the real Supervisor.claim_named against a stubbed process: never stops a busy or loading profile
class P:  # fake Popen
    def poll(self): return None
class Sup(b.Supervisor):
    def __init__(self):  # no threads, no config
        import threading
        self.lock = threading.RLock(); self.switch_lock = threading.Lock()
        self.proc = P(); self.state = "running"; self.running_profile = {"name": "beta"}
        self.inflight = 1; self.sw_waiters = 0; self.log = []
    def stop(self, r="manual"): self.log.append(("stop", r)); self.proc = None; self.state = "stopped"
    def start(self, p=None, via_router=False): self.log.append(("start", p)); self.proc = P(); self.state = "starting"; self.running_profile = {"name": p}

s = Sup(); s.inflight = 1   # just me
check("real claim: idle other profile -> stop + start", s.claim_named("alpha") == ("ok", 0) and s.log == [("stop", "model switch to alpha"), ("start", "alpha")])
s = Sup(); s.inflight = 3
r = s.claim_named("alpha")
check("real claim: 2 others running -> busy 2, nothing stopped", r == ("busy", 2) and s.log == [])
s = Sup(); s.inflight = 3; s.sw_waiters = 3   # me + two other parked waiters, no real work
check("real claim: parked waiters are not work", s.claim_named("alpha", waiting=True) == ("ok", 0) and s.log[-1] == ("start", "alpha"))
s = Sup(); s.inflight = 3; s.sw_waiters = 1
check("real claim: a parked waiter does not hide real work", s.claim_named("alpha") == ("busy", 1) and s.log == [])
s = Sup(); s.inflight = 1; s.state = "starting"
check("real claim: other profile loading for someone -> busy, load untouched", s.claim_named("alpha")[0] == "busy" and s.log == [])
s = Sup(); s.running_profile = {"name": "alpha"}; s.inflight = 5
check("real claim: already loaded -> ok, untouched", s.claim_named("alpha") == ("ok", 0) and s.log == [])
s = Sup(); s.proc = None; s.state = "stopped"
check("real claim: stopped -> start", s.claim_named("alpha") == ("ok", 0) and s.log == [("start", "alpha")])

# config keys
ex = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.example.json")))
ex.update(api_keys=["k"], admin_password="p")
d = b.Config.normalize(ex)
check("defaults: switch_wait_s 120, strict_model_names true", d["switch_wait_s"] == 120 and d["strict_model_names"] is True)
old = {k: v for k, v in ex.items() if k not in ("switch_wait_s", "strict_model_names")}
try:
    b.Config.validate(b.Config.normalize(old)); okold = True
except Exception:
    okold = False
check("a config.json without the new keys still validates", okold)
def v(**kw):
    x = b.Config.normalize(dict(ex, **kw))
    try:
        b.Config.validate(x); return True
    except ValueError:
        return False
check("switch_wait_s 0 and 1800 accepted", v(switch_wait_s=0) and v(switch_wait_s=1800))
check("switch_wait_s -1, 1801, 'x', 1.5, true, null rejected", not any(v(switch_wait_s=x) for x in (-1, 1801, "x", 1.5, True, None)))
check("strict_model_names bool accepted, others rejected", v(strict_model_names=False) and not any(v(strict_model_names=x) for x in ("yes", 1, None)))

# profile_contexts: per-profile ctx + 'auto' = the ctx of the profile auto would serve
class _Cfg(dict):
    pass
cfgc = _Cfg(profiles=[{"name": "a", "ctx": 32768}, {"name": "b", "ctx": 16384}, {"name": "c", "ctx": None}, {"name": "d", "ctx": True}],
            active_profile="a", router={"alias": "auto"})
cfgc.get = lambda k, d=None, _c=dict(cfgc): _c.get(k, d)
class _Sup:
    def __init__(self, cur): self.cur = cur
    def serving(self): return self.cur
pc = b.profile_contexts(cfgc, _Sup(None))
check("profile_contexts: valid ints only (None and bool skipped)", pc.get("a") == 32768 and pc.get("b") == 16384 and "c" not in pc and "d" not in pc)
check("auto = active profile's ctx when nothing is loaded", pc.get("auto") == 32768)
check("auto = loaded profile's ctx when one is loaded", b.profile_contexts(cfgc, _Sup("b")).get("auto") == 16384)
check("auto ignores an unknown loaded name, falls back to active", b.profile_contexts(cfgc, _Sup("zzz")).get("auto") == 32768)
class _Boom:
    def serving(self): raise RuntimeError("x")
check("profile_contexts never raises (supervisor error)", b.profile_contexts(cfgc, _Boom()).get("auto") == 32768)
check("profile_contexts never raises (garbage cfg)", b.profile_contexts(None, None) == {})


# require_api_key (default true): off means no key needed and an empty key list is allowed
check("require_api_key defaults to true and empty api_keys is rejected then", not v(api_keys=[]))
check("require_api_key false allows an empty api_keys list", v(api_keys=[], require_api_key=False))
check("require_api_key false with keys still validates", v(require_api_key=False))
check("require_api_key must be a bool", not any(v(require_api_key=x) for x in ("no", 0, None, "false")))
print("FAILED: %d" % len(fails) if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
