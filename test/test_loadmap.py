#!/usr/bin/env python3
"""Unit checks for parse_startup_log(), model_total_bytes() and Supervisor.load_info(): tolerant, never raise."""
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

lines = open(os.path.join(HERE, "fixtures", "startup-q3.log")).read().splitlines()
r = b.parse_startup_log(lines)
check("fixture: profile", r["profile"] == "dsv4-flash-q3")
check("fixture: layers 24/44", (r["layers_offloaded"], r["layers_total"]) == (24, 44))
check("fixture: 4 devices in order", [d["name"] for d in r["devices"]] == ["CUDA0", "CUDA1", "CUDA2", "CUDA3"])
d0 = r["devices"][0]
check("fixture: CUDA0 buffers", (d0["weights_mib"], d0["kv_mib"], d0["compute_mib"]) == (8120.4, 384.0, 912.0))
check("fixture: cpu total includes CPU + CUDA_Host", abs(r["cpu_mib"] - (84210.77 + 1664.0 + 96.05)) < 0.5)
check("fixture: 43 per-layer entries, layer 0 on CUDA0, layer 42 on CPU",
      len(r["per_layer"]) == 43 and r["per_layer"][0] == {"layer": 0, "device": "CUDA0"} and r["per_layer"][42]["device"] == "CPU")
check("fixture: fit raw kept (<=12), no [ctl] or print_info lines",
      0 < len(r["fit"]["raw"]) <= 12 and not any(l.startswith("[ctl]") or "print_info" in l for l in r["fit"]["raw"]))
check("fixture: comment header does not leak into fit", not any("SYNTHETIC" in l for l in r["fit"]["raw"]))

# only the part after the last 'starting profile' counts
r2 = b.parse_startup_log(["[ctl] starting profile old", "CUDA0 model buffer size = 1.00 MiB"] + lines)
check("last start wins", r2["profile"] == "dsv4-flash-q3" and len(r2["devices"]) == 4 and r2["devices"][0]["weights_mib"] == 8120.4)

e = b.parse_startup_log([])
check("empty log: all unknown", e["devices"] == [] and e["cpu_mib"] is None and e["layers_total"] is None
      and e["per_layer"] == [] and e["fit"]["raw"] == [] and not e["have_data"])
g = b.parse_startup_log(["", "\x00\xff garbage", "CUDA0 model buffer size = lots MiB", "offloaded x/y layers", "layer 3 assigned to", None, 42])
check("garbage lines: no raise, no data", g["devices"] == [] and g["layers_offloaded"] is None and not g["have_data"])
p = b.parse_startup_log(["CUDA1 model buffer size = 2 GiB", "offloaded 3 / 9 layers to GPU"])
check("GiB converted, spaced N / M accepted", p["devices"][0]["weights_mib"] == 2048.0 and p["layers_total"] == 9)

# model size: shards and single file
d = tempfile.mkdtemp()
for i in (1, 2, 3):
    with open(f"{d}/m-0000{i}-of-00003.gguf", "wb") as f:
        f.truncate(1000 * i)
with open(f"{d}/one.gguf", "wb") as f:
    f.truncate(777)
check("sharded size sums all shards", b.model_total_bytes(f"{d}/m-00001-of-00003.gguf") == 6000)
check("sharded size from any shard", b.model_total_bytes(f"{d}/m-00003-of-00003.gguf") == 6000)
check("single file size", b.model_total_bytes(f"{d}/one.gguf") == 777)
check("missing model -> None", b.model_total_bytes(f"{d}/nope.gguf") is None and b.model_total_bytes("") is None)

# load_info: monotonic pct, capped at 99 until ready, phase falls back to state
class C:
    def get(self, k): return None
s = b.Supervisor(C())
check("stopped: phase is state, rest None", s.load_info()["phase"] == "stopped" and s.load_info()["pct"] is None)
s._load_reset(f"{d}/m-00001-of-00003.gguf")
class P:
    pid = 1
    def poll(self): return None
s.proc, s.state = P(), "starting"
seq = [{"rchar": 3000}, {"rchar": 1000}, {"rchar": 9000}, {"rchar": None, "read_bytes": None, "rss_mib": None}]
pcts = [s.load_info(x)["pct"] for x in seq]
check("pct monotonic and capped 99 before ready", pcts == [50, 50, 99, 99] and all(p <= 99 for p in pcts))
check("bytes_read clamped to total", s.load_info({"rchar": 9000})["bytes_read"] == 6000)
s._note_line(s.proc, "load_tensors: offloaded 24/44 layers to GPU")
check("phase advances from log", s.load_info({"rchar": 100})["phase"] == "allocating/warming")
s._note_line(s.proc, "load_model: loading model")
check("phase never goes back", s.load_info({"rchar": 100})["phase"] == "allocating/warming")
s.state = "running"
li = s.load_info()
check("running: ready, 100", li["phase"] == "ready" and li["pct"] == 100 and li["eta_s"] == 0)
s2 = b.Supervisor(C()); s2.state = "starting"
check("no stats at all: unknown, no raise", s2.load_info({})["pct"] is None)

print("FAILED: %s" % fails if fails else "all passed")
sys.exit(1 if fails else 0)
