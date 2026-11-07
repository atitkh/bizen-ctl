#!/usr/bin/env python3
"""Unit checks for the activity recorder: summarize_activity (pure), ActivityLog (dedupe, caps, window, persistence,
offline marker, corrupt file) and the Supervisor.gen_end hook (metadata only, never prompt text)."""
import json, os, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

# ---- summarize_activity
ev = [{"t": 100, "state": "starting", "profile": "A"}, {"t": 130, "state": "running", "profile": "A"},
      {"t": 200, "state": "stopping", "profile": "A"}, {"t": 210, "state": "stopped", "profile": None},
      {"t": 300, "state": "starting", "profile": "B"}, {"t": 310, "state": "failed", "profile": None},
      {"t": 400, "state": "starting", "profile": "B"}, {"t": 460, "state": "running", "profile": "B"}]
rq = [{"t0": 140, "t1": 150, "profile": "A", "tokens": 100, "ttft": 1.0, "tps": 10.0},
      {"t0": 160, "t1": 170, "profile": "A", "tokens": 50, "ttft": 3.0, "tps": 30.0},
      {"t0": 170, "t1": 175, "profile": "A", "tokens": 20, "ttft": 2.0, "tps": 20.0},
      {"t0": 470, "t1": 480, "profile": "B", "tokens": 400, "ttft": 0.5, "tps": 40.0},
      {"t0": 10, "t1": 20, "profile": "A", "tokens": 999, "ttft": 9.0, "tps": 99.0}]
s = b.summarize_activity(ev, rq, since=50)
check("two profiles reported", sorted(s) == ["A", "B"])
check("A: 3 requests in the window (the old one is excluded)", s["A"]["requests"] == 3)
check("A: tokens summed (170)", s["A"]["tokens"] == 170)
check("A: average tok/s 20.0 and median 20.0", s["A"]["avg_tps"] == 20.0 and s["A"]["median_tps"] == 20.0)
check("A: median time to first token 2.0", s["A"]["median_ttft"] == 2.0)
check("A: one load of 30 s", s["A"]["loads"] == 1 and s["A"]["avg_load_s"] == 30.0)
check("B: the failed load is not counted, the second is (60 s)", s["B"]["loads"] == 1 and s["B"]["avg_load_s"] == 60.0)
check("B: single request numbers", s["B"]["requests"] == 1 and s["B"]["avg_tps"] == 40.0 and s["B"]["median_ttft"] == 0.5)
s0 = b.summarize_activity(ev, rq, since=0)
check("since=0 includes the old request", s0["A"]["requests"] == 4 and s0["A"]["tokens"] == 1169)
check("empty input -> {}", b.summarize_activity([], []) == {} and b.summarize_activity(None, None) == {})
junk = b.summarize_activity([None, 5, {"t": "x"}, {"t": 1, "state": None}], [None, 3, {"t1": "x"}, {"t1": 10, "profile": "J", "tokens": "many", "tps": True}], 0)
check("junk records never raise (bool/str values ignored)", junk == {"J": {"requests": 1, "tokens": 0, "avg_tps": None, "median_tps": None, "median_ttft": None, "loads": 0, "avg_load_s": None}})
check("even-length median is the mean of the middle two", b._median([1, 2, 3, 4]) == 2.5 and b._median([]) is None)

# ---- ActivityLog
a = b.ActivityLog(path=None, max_events=5, max_requests=3)
a.observe("stopped", None, now=1); a.observe("stopped", None, now=2); a.observe("starting", "A", now=3)
check("same (state, profile) twice records one event", len(a.events) == 2 and a.events[0]["t"] == 1)
for i in range(10):
    a.observe("running" if i % 2 == 0 else "stopped", "A" if i % 2 == 0 else None, now=10 + i)
check("events are capped (oldest dropped)", len(a.events) == 5 and a.events[-1]["t"] == 19)
for i in range(5):
    a.request({"t0": i, "t1": i + 1, "profile": "A", "tokens": i, "ttft": None, "tps": None})
check("requests are capped", len(a.requests) == 3 and a.requests[-1]["tokens"] == 4)
a2 = b.ActivityLog(path=None)
a2.observe("running", "A", now=100); a2.observe("stopped", None, now=200); a2.observe("starting", "B", now=300)
ev2, rq2 = a2.snapshot(since=250)
check("snapshot keeps the last event before the window", [e["t"] for e in ev2] == [200, 300])
check("snapshot of a window after everything keeps the last state", [e["t"] for e in a2.snapshot(since=999)[0]] == [300])
api = a2.api(hours=1)
check("api() carries now/since/events/requests/profiles", {"now", "since", "events", "requests", "profiles", "request_count"} <= set(api))

# ---- persistence
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "activity.json")
    p1 = b.ActivityLog(path=path)
    p1.maybe_save()
    check("nothing recorded -> nothing written", not os.path.exists(path))
    p1.observe("running", "A", now=time.time() - 100)
    p1.request({"t0": 1.0, "t1": 2.0, "profile": "A", "tokens": 5, "ttft": 0.4, "tps": 12.5})
    p1.maybe_save()
    check("dirty but within the save interval -> not written yet", not os.path.exists(path))
    p1.maybe_save(force=True)
    check("force writes the file, no temp file left", os.path.exists(path) and not os.path.exists(path + ".tmp"))
    saved = json.load(open(path))
    check("file has only metadata keys (no text fields)", set(saved["requests"][0]) == {"t0", "t1", "profile", "tokens", "ttft", "tps"})
    p2 = b.ActivityLog(path=path)
    check("reload restores requests", len(p2.requests) == 1 and p2.requests[0]["tps"] == 12.5)
    check("reload appends an offline marker after the last event", p2.events[-1]["state"] == "offline" and p2.events[-1]["t"] == saved["saved_at"])
    p2.observe("running", "A")
    check("the first observe after a reload always adds an event", p2.events[-1]["state"] == "running")
    open(path, "w").write("{broken json")
    p3 = b.ActivityLog(path=path)
    check("a corrupt file is ignored, not fatal", len(p3.events) == 0 and len(p3.requests) == 0)
    json.dump({"saved_at": "x", "events": [None, {"t": 1}, {"t": 2, "state": "running", "profile": "A"}], "requests": [None, {"t1": 5, "profile": "A"}, "z"]}, open(path, "w"))
    p4 = b.ActivityLog(path=path)
    check("junk entries in a valid file are filtered", len(p4.events) == 1 and len(p4.requests) == 1)

# ---- Supervisor.gen_end writes exactly the metadata record
class C:
    def get(self, k): return None
sup = b.Supervisor(C())
act = b.ActivityLog(path=None)
sup.activity = act
gid = sup.gen_begin()
for _ in range(5):
    sup.gen_chunk(gid, 1); time.sleep(0.02)
sup.gen_end(gid)
r = act.requests[-1]
check("gen_end recorded one request with the documented keys only", len(act.requests) == 1 and set(r) == {"t0", "t1", "profile", "tokens", "ttft", "tps"})
check("tokens counted, ttft and tps present, profile None when nothing is loaded", r["tokens"] == 5 and r["ttft"] is not None and r["tps"] and r["profile"] is None)
sup2 = b.Supervisor(C())
sup2.gen_end(sup2.gen_begin())
check("a supervisor without an ActivityLog still works", True)


# ---- UI: the Activity block of ui.html run in node with a recording fake DOM
import shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.environ.get("BIZEN_UI") or os.path.join(HERE, "..", "ui.html")
if shutil.which("node"):
    html = open(UI).read()
    js = html[html.index("/* ---- Activity tab"):html.index("const TABS = [")]
    harness = r"""
const mk = tag => ({ tag, attrs: {}, style: {}, children: [], textContent: '', className: '', dataset: {},
  classList: { toggle() {}, add() {} }, setAttribute(k, v) { this.attrs[k] = String(v); },
  appendChild(c) { this.children.push(c); return c; }, replaceChildren(...c) { this.children = c; }, addEventListener() {} });
const document = { createElement: mk, createElementNS: (ns, t) => mk(t), createTextNode: t => ({ tag: '#text', textContent: String(t), children: [] }),
  querySelectorAll: () => [], hidden: false };
const tabHooks = {}; const $ = id => mk('x'); const api = async () => ({}); let cfg = { profiles: [] };
""" + js + r"""
const walk = (e, f) => { f(e); (e.children || []).forEach(c => walk(c, f)); };
const all = (e, tag) => { const o = []; walk(e, x => { if (x.tag === tag) o.push(x); }); return o; };
const text = e => { let s = ''; walk(e, x => { if (x.textContent) s += x.textContent + ' '; }); return s; };
const out = {};
const A = [{ t: 50, state: 'stopped', profile: null }, { t: 120, state: 'starting', profile: 'A' }, { t: 150, state: 'running', profile: 'A' }];
const s1 = activitySegments(A, 100, 200);
out.seg = s1.map(s => [s.t0, s.t1, s.state, s.profile]);
out.segSum = s1.reduce((a, s) => a + s.t1 - s.t0, 0);
out.segEarly = activitySegments(A, 10, 60).map(s => [s.t0, s.t1, s.state]);
out.segEmpty = activitySegments([], 0, 10).map(s => [s.t0, s.t1, s.state]);
out.segLate = activitySegments([{ t: 500, state: 'running', profile: 'Z' }], 0, 100).map(s => s.state);
out.segUnsorted = activitySegments([A[2], A[0], A[1]], 100, 200).map(s => s.state);
const t1 = activityTicks(0, 3600), t7 = activityTicks(0, 7 * 86400);
out.ticks = [t1.step, t1.ticks.length, t7.step, t7.ticks.length, t1.ticks.every(t => t % t1.step === 0)];
out.colors = [activityColor('x', ['a', 'x']) === ACT_PALETTE[1], activityColor('zz') === activityColor('zz'), ACT_PALETTE.includes(activityColor('q'))];
const D = { since: 100, now: 200, state: 'running', profile: 'A', request_count: 4, events: A,
  requests: [{ t0: 160, t1: 170, profile: 'A', tokens: 100, ttft: 1.2, tps: 25 }, { t0: 171, t1: 175, profile: 'A', tokens: 50, ttft: 0.5, tps: 40 },
             { t0: 180, t1: 190, profile: '<img onerror=x>', tokens: 10, ttft: null, tps: 8 }, { t0: 191, t1: 195, profile: 'A', tokens: null, ttft: null, tps: null }],
  profiles: { A: { requests: 3, tokens: 150, avg_tps: 32.5, median_tps: 32.5, median_ttft: 0.85, loads: 1, avg_load_s: 30 }, B: { requests: 0, tokens: 0, avg_tps: null, median_tps: null, median_ttft: null, loads: 0, avg_load_s: null } } };
const r = renderActivity(D, ['A', 'B']);
out.rects = all(r, 'rect').length; out.circles = all(r, 'circle').length;
out.titles = all(r, 'title').map(t => t.textContent);
out.hasImg = all(r, 'img').length; out.txt = text(r);
out.empty = text(renderActivity(Object.assign({}, D, { requests: [] }), ['A']));
out.bad = [text(renderActivity(null)), text(renderActivity({})), text(renderActivity({ since: 'x', now: 5 }))];
const cards = renderActivityCards(D.profiles, ['A', 'B']);
out.cards = [all(cards, 'h3').length, text(all(cards, 'h3')[0]).trim(), text(cards).includes('32.5 tok/s'), text(cards).includes('–')];
out.cardsEmpty = text(renderActivityCards({}, []));
const tbl = renderActivityTable(D.requests, ['A']);
out.rows = all(tbl, 'tr').length; out.firstRow = text(all(tbl, 'tr')[1]); out.secondRow = text(all(tbl, 'tr')[2]);
const many = Array.from({ length: 40 }, (_, i) => ({ t0: i, t1: i + 1, profile: 'A', tokens: i, ttft: 1, tps: 2 }));
out.rows40 = all(renderActivityTable(many, ['A']), 'tr').length; out.newestFirst = text(all(renderActivityTable(many, ['A']), 'tr')[1]).includes(' 39 ');
out.tblEmpty = text(renderActivityTable([], []));
out.fmt = [actFmtDur(null), actFmtDur(30), actFmtDur(300), actFmtDur(7200)];
console.log(JSON.stringify(out));
"""
    hp = os.path.join(tempfile.mkdtemp(), "harness.js"); open(hp, "w").write(harness)
    r = subprocess.run(["node", hp], capture_output=True, text=True, timeout=60)
    try:
        o = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        o = None
        print(r.stderr[-800:])
    check("ui: the activity block loads and renders without throwing", o is not None)
    if o:
        check("segments: stopped, starting A, running A clipped to the window", o["seg"] == [[100, 120, "stopped", None], [120, 150, "starting", "A"], [150, 200, "running", "A"]])
        check("segments always cover the whole window (sum 100)", o["segSum"] == 100)
        check("before the first event the state is 'unknown'", o["segEarly"] == [[10, 50, "unknown"], [50, 60, "stopped"]])
        check("no events -> one unknown segment; a later event is ignored", o["segEmpty"] == [[0, 10, "unknown"]] and o["segLate"] == ["unknown"])
        check("unsorted events give the same result", o["segUnsorted"] == ["stopped", "starting", "running"])
        check("ticks: 1 h window -> at most 8 ticks on round times; 7 d -> daily", o["ticks"][0] <= 3600 and o["ticks"][1] <= 8 and o["ticks"][4] and o["ticks"][2] == 86400 and o["ticks"][3] == 8)
        check("colours: index in the profile list wins, stable hash otherwise", all(o["colors"]))
        check("chart: one rect per segment (3) and one dot per request that has a tok/s (3 of 4)", o["rects"] == 3 and o["circles"] == 3)
        check("dot tooltips carry profile, tok/s, tokens", any("25 tok/s" in t and "100 tokens" in t and t.startswith("A:") for t in o["titles"]))
        check("a profile name with markup stays text (no element created)", o["hasImg"] == 0 and "<img onerror=x>" in o["txt"])
        check("empty window says so", "no requests in this window" in o["empty"])
        check("missing/garbage data -> 'activity unavailable', never a throw", all("activity unavailable" in x for x in o["bad"]))
        check("cards: busiest first, median shown, missing numbers as a dash", o["cards"][0] == 2 and o["cards"][1].startswith("A") and o["cards"][2] and o["cards"][3])
        check("cards: empty profiles -> message", "nothing recorded" in o["cardsEmpty"])
        check("table: header + 4 rows, newest first (the t1=195 request without numbers, then the markup-named one)", o["rows"] == 5 and o["firstRow"].count("–") == 3 and "4s" in o["firstRow"] and "<img onerror=x>" in o["secondRow"] and "10s" in o["secondRow"])
        check("table: capped at 25 rows (+ header), newest first", o["rows40"] == 26 and o["newestFirst"])
        check("table: empty -> message", "no requests yet" in o["tblEmpty"])
        check("duration formatting", o["fmt"] == ["–", "30s", "5 min", "2.0 h"])
else:
    print("SKIP ui checks: node not installed")

print("FAILED: %d" % len(fails) if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
