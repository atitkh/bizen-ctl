#!/usr/bin/env python3
"""Unit checks for meminfo()/proc_stats()/Sampler: they must never raise, even with /proc files missing."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

m = b.meminfo(os.getpid())
need = ["total_mib", "available_mib", "used_mib", "free_mib", "buffers_mib", "cached_mib",
        "swap_total_mib", "swap_used_mib", "nodes", "cgroup", "child"]
check("meminfo has all fields", all(k in m for k in need))
check("used = total - available", m["total_mib"] is None or m["used_mib"] == m["total_mib"] - m["available_mib"])
check("child rss for own pid", isinstance(m["child"]["rss_mib"], int) and m["child"]["rss_mib"] > 0)
check("nodes shape", all({"node", "total_mib", "free_mib", "used_mib"} <= set(n) and n["used_mib"] == n["total_mib"] - n["free_mib"] for n in m["nodes"]))

# a pid that does not exist, and None: every field None, no exception
check("proc_stats(missing pid)", b.proc_stats(2**22 + 12345) == {"rss_mib": None, "read_bytes": None, "rchar": None})
check("proc_stats(None)", b.proc_stats(None) == {"rss_mib": None, "read_bytes": None, "rchar": None})

# every /proc and /sys read missing: meminfo still returns the full shape
real = b.read
b.read = lambda path, default="": default
try:
    m2 = b.meminfo(1)
    check("meminfo with all reads missing", all(k in m2 for k in need) and m2["total_mib"] is None and m2["nodes"] == [])
    smp = b.Sampler(type("S", (), {"proc": None})())
    s = smp.sample()
    check("sampler.sample with reads missing", "t" in s and s["cpu"] == [])
finally:
    b.read = real

# garbage in meminfo-style text is skipped, not fatal
check("_kb_map tolerates junk", b._kb_map("MemTotal: 2048 kB\nbogus\nX: abc kB\n\nMemFree: 1024 kB") == {"MemTotal": 2, "MemFree": 1})

# sampler: ring buffer + since filter
smp = b.Sampler(type("S", (), {"proc": None})(), keep=5)
for i in range(8):
    smp.samples.append({"t": float(i)})
check("ring buffer keeps last N", [s["t"] for s in smp.samples] == [3.0, 4.0, 5.0, 6.0, 7.0])
check("since filter is strictly newer", [s["t"] for s in smp.since(5.0)] == [6.0, 7.0])
real_s = smp.sample(); real_s2 = smp.sample()
check("cpu busy % within 0..100", all(0 <= x <= 100 for x in real_s2["cpu"]))
sys.exit(1 if fails else 0)
