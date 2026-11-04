#!/usr/bin/env python3
"""Placement from VRAM growth (compute_placement, Supervisor.placement_info/_freeze_placement), the phase order on the
real startup log and parse_startup_log on a build that prints no layer lines. Never needs a GPU."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

def g(i, used, name="NVIDIA RTX 6000"):
    return {"index": i, "name": name, "mem_used": used, "mem_total": 49140, "util": 0, "temp": 30}

base = [g(0, 500), g(1, 600), g(2, 700), g(3, 800)]
now = [g(0, 20500), g(1, 10600), g(2, 9000), g(3, 90000)]       # GPU 3 is somebody else's
p = b.compute_placement(base, now, [0, 1], 4096)
check("growth per profile GPU", [d["mib"] for d in p["devices"]] == [20000, 10000] and p["devices"][0]["name"] == "NVIDIA RTX 6000")
check("other GPUs ignored", [d["index"] for d in p["devices"]] == [0, 1] and p["total_mib"] == 30000 + 4096)
check("share = gpu / (gpu + host)", p["gpu_share_pct"] == round(100 * 30000 / 34096, 1) and p["host_mib"] == 4096 and not p["frozen"])
check("source tag", p["source"] == "vram-delta")
n = b.compute_placement(base, [g(0, 100), g(1, 700)], [0, 1], 10)
check("negative delta clamped to 0", [d["mib"] for d in n["devices"]] == [0, 100])
m = b.compute_placement(base, [g(0, 1500)], [0, 1], None)
check("GPU missing now -> None, host unknown -> share None", [d["mib"] for d in m["devices"]] == [1000, None]
      and m["host_mib"] is None and m["gpu_share_pct"] is None and m["total_mib"] == 1000)
e = b.compute_placement([], [], [0], None)
check("no data at all: unknown everywhere", e["devices"][0]["mib"] is None and e["total_mib"] is None and e["gpu_share_pct"] is None)
h = b.compute_placement(base, [], [], 2048)
check("no GPUs (cpu profile): host only", h["devices"] == [] and h["total_mib"] == 2048 and h["gpu_share_pct"] is None)
check("garbage input does not raise", b.compute_placement([None, 5, {"index": 0, "mem_used": "x"}], [{"index": 0}], [0], "x") is not None
      and b.compute_placement(None, None, None, None) is not None)

# Supervisor: baseline -> live while loading -> frozen at ready
class C:
    def get(self, k): return None
class P:
    pid = 1
    def poll(self): return None
s = b.Supervisor(C())
check("never started: None", s.placement_info() is None)
s.proc, s.state, s.running_gpus, s.place_base = P(), "starting", [0, 1], base
li = s.placement_info(gpus=[g(0, 5500), g(1, 600), g(2, 99999)], stats={"rss_mib": 1000})
check("live while loading", [d["mib"] for d in li["devices"]] == [5000, 0] and li["host_mib"] == 1000 and not li["frozen"])
orig_gi = b.gpu_info
b.gpu_info = lambda: [g(0, 20500), g(1, 10600)]
s._freeze_placement()
s.state = "running"
b.gpu_info = lambda: [g(0, 21000), g(1, 11000)]       # KV grows after the freeze
s.place_gpus_cache = (0.0, [])
fz = s.placement_info(stats={"rss_mib": 2000})
b.gpu_info = orig_gi
check("frozen copy kept, live keeps moving", fz["frozen"] and [d["mib"] for d in fz["devices"]] == [20000, 10000]
      and [d["mib"] for d in fz["live"]["devices"]] == [20500, 10400])
cp = s._placement_compact()
check("compact status field", cp["frozen"] and cp["gpu_mib"] == [20000, 10000] and cp["total_mib"] == fz["total_mib"])
s.state, s.proc, s.running_gpus = "stopped", None, []
sp = s.placement_info()
check("after stop: frozen summary stays, no live", sp["frozen"] and sp["live"] is None)
s._load_reset(None)
check("new start resets", s.placement_info() is None)

# phase order on the real log: starting -> reading tensors -> allocating/warming -> ready
lines = open(os.path.join(HERE, "fixtures", "startup-dsv4-real.log")).read().splitlines()
ph, seen = 0, []
for l in lines:
    ph = b.phase_of_line(l, ph)
    if not seen or seen[-1] != ph:
        seen.append(ph)
check("real log: phases 1, 2, 3 in order", seen == [0, 1, 2, 3])
check("threadpool init alone is phase 2", b.phase_of_line("5.09 I cmn init: llama threadpool init, n_threads = 8", 1) == 2)
check("phase never goes back", b.phase_of_line("load_model: loading model", 3) == 3)
r = b.parse_startup_log(lines)
check("real log: no layer data, no raise", not r["have_data"] and r["devices"] == [] and r["per_layer"] == []
      and r["profile"] == "dsv4flash-q3-4gpu")

print("FAILED: %s" % fails if fails else "all passed")
sys.exit(1 if fails else 0)
