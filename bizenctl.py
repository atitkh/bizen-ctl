#!/usr/bin/env python3
"""bizen-ctl: one-port controller for llama-server on a shared GPU box.

- /v1/* (and other llama-server routes): OpenAI-compatible proxy, API-key protected.
  Starts the active profile on first request, unloads it after idle_ttl_min.
- / and /api/*: admin web UI (password login) to edit profiles, start/stop,
  watch GPUs/logs, and send a test prompt.

Stdlib only. Run: python3 bizenctl.py --config config.json
"""
import argparse
import collections
import glob
import hashlib
import hmac
import http.client
import http.server
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host"}
TPS_RE = re.compile(r"eval time\s*=.*?([\d.]+)\s*tokens per second")
PROMPT_TPS_RE = re.compile(r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second")
GEN_PATHS = ("/v1/chat/completions", "/v1/completions", "/completion", "/chat/completions", "/completions")
# load phases, in order; each log line can only move the phase forward. Patterns are llama.cpp-style and optional.
LOAD_PHASES = ["starting", "reading tensors", "allocating/warming", "ready"]
PHASE_RES = [
    (1, re.compile(r"load_model|llama_model_load|load_tensors|loading model tensors|llama_model_loader", re.I)),
    (2, re.compile(r"offloaded\s+\d+\s*/\s*\d+\s+layers|llama_context|KV buffer size|compute buffer size|warming up|initializing slots"
                   r"|threadpool init|initializing, n_slots", re.I)),
    (3, re.compile(r"server is listening|ready after|all slots are idle", re.I)),
]


def phase_of_line(line, cur=0):
    """Load phase after one more log line (0..3); never goes back. Never raises."""
    try:
        hit = [i for i, rx in PHASE_RES if rx.search(line)]
        return max([cur] + hit)
    except Exception:  # noqa: BLE001
        return cur


def compute_placement(base, now, indices, rss_mib):
    """Placement from resource growth: per profile GPU, max(0, mem_used_now - baseline) MiB (None when either
    snapshot lacks the GPU); other GPUs are ignored. base/now are gpu_info() lists. Never raises."""
    try:
        bm = {g.get("index"): g for g in (base or []) if isinstance(g, dict)}
        nm = {g.get("index"): g for g in (now or []) if isinstance(g, dict)}
        devs = []
        for i in indices or []:
            b0, n0 = bm.get(i) or {}, nm.get(i) or {}
            bu, nu = b0.get("mem_used"), n0.get("mem_used")
            mib = max(0, nu - bu) if isinstance(bu, (int, float)) and isinstance(nu, (int, float)) else None
            devs.append({"index": i, "name": n0.get("name") or b0.get("name"), "mib": mib})
        known = [d["mib"] for d in devs if d["mib"] is not None]
        gpu_total = sum(known) if known else None
        host = rss_mib if isinstance(rss_mib, (int, float)) else None
        total = (gpu_total or 0) + (host or 0) if (known or host is not None) else None
        share = None
        if gpu_total is not None and host is not None and gpu_total + host > 0:
            share = round(100.0 * gpu_total / (gpu_total + host), 1)
        return {"source": "vram-delta", "devices": devs, "host_mib": host, "total_mib": total,
                "gpu_share_pct": share, "frozen": False}
    except Exception:  # noqa: BLE001
        return None


def log(msg):
    sys.stdout.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- config

DEFAULTS = {
    "listen_host": "0.0.0.0",
    "listen_port": 8080,
    "api_keys": [],
    "admin_password": "",
    "llama_server": "~/llama.cpp/build/bin/llama-server",
    "model_dirs": ["/srv/fs/models"],
    "gpu_bucket_mib": 512,          # auto:N ranks GPUs by used VRAM in buckets of this size (ties go to the highest index)
    "gpu_fallback_count": 8,        # GPU indices assumed for auto:N when nvidia-smi is unavailable
    "download_mbps": 11.7,          # only used for the "about N min" download estimate in the Kev hint
    "download_dir": "",             # "" = the first model_dirs entry
    "hf_token": "",                 # write-only: never returned by the API
    "hf_endpoint": "https://huggingface.co",
    "min_free_gib_for_download": 20,
    "internal_port": 18081,
    "idle_ttl_min": 10,
    "keep_loaded": False,
    "start_timeout_s": 600,
    "active_profile": "",
    "profiles": [],
    "kev": {},
    "router": {},
}

KEV_DEFAULTS = {
    "managed": True,                    # bizen-ctl starts/stops Kev; false = use `url` as-is
    "url": "http://127.0.0.1:8009",     # only used when managed is false
    "api_key": "",                      # only for an external Kev started with KEV_API_KEY
    "dir": "~/kev",                     # checkout with .venv (uv sync --extra serve)
    "model": "jaredpalmer/kev-0.8b",
    "port": 8009,
    "dtype": "fp32",                    # fp32 = the path Kev's authors evaluate; Turing has no bf16
    "cuda_graphs": "auto",              # auto = Kev's default (on); off = KEV_CUDA_GRAPHS=0, saves ~2 GB on Kev-4B/9B
    "gpus": "auto:1",                   # never overlaps the llama-server GPUs
    "hf_home": "/srv/fs/hf",            # HF cache for the weights ("" = default)
    "idle_ttl_min": 10,                 # unload Kev after this long without routing (0 = never)
    "start_timeout_s": 900,             # a new model's first start downloads its base weights
    "command": "",                      # override; placeholders {dir} {model} {port}
}

# Settings that change what is launched; editing one while Kev runs flags "restart needed".
KEV_LAUNCH_KEYS = ("dir", "model", "port", "dtype", "cuda_graphs", "gpus", "hf_home", "command")

# Published Kev checkpoints (huggingface.co/jaredpalmer). Most are LoRA adapters on a base
# model that is downloaded on first start. weights_gib is the base in bf16/fp16 (fp32 = 2x).
# Kev loads on ONE GPU: no sharding in its serving code.
KEV_PRESETS = [
    {"id": "jaredpalmer/kev-0.8b", "base": "Qwen3.5-0.8B-Base", "weights_gib": 1.5, "approx": False,
     "note": "default; the size tested here in fp32"},
    {"id": "jaredpalmer/kev-4b", "base": "Qwen3.5-4B-Base", "weights_gib": 8.4, "approx": False,
     "note": "better decisions; weights are about 8.4 GiB in fp16 (fp32 about 17 GiB) and CUDA-graph buffers add about 2 GB"},
    {"id": "jaredpalmer/kev-0.6b", "base": "Qwen3-0.6B-Base", "weights_gib": 1.2, "approx": True,
     "note": "smallest current checkpoint"},
    {"id": "jaredpalmer/kev-8b", "base": "Qwen3-8B", "weights_gib": 16.4, "approx": True,
     "note": "about 16 GiB of weights in fp16"},
    {"id": "jaredpalmer/kev-9b", "base": "Qwen3.5-9B-Base", "weights_gib": 19.3, "approx": False,
     "note": "about 19 GiB of weights in fp16"},
    {"id": "jaredpalmer/kev-27b", "base": "Qwen3.8-27B", "weights_gib": 51.3, "approx": False,
     "note": "bf16 only, about 51 GiB of weights: needs an 80 GB-class GPU"},
    {"id": "jaredpalmer/kev-0.5b", "base": "Qwen2.5-0.5B", "weights_gib": 1.0, "approx": True,
     "note": "legacy prototype"},
]

ROUTER_DEFAULTS = {
    "enabled": False,
    "alias": "auto",                    # clients send model="auto" to be routed
    "instructions": "Which model is best suited to answer this prompt?",
    "default_profile": "",              # used when Kev is unsure or unavailable
    "threshold": 0.6,                   # minimum probability for Kev's pick to be trusted (flat = 1/routes)
    "max_chars": 6000,                  # prompt text sent to Kev (the tail is kept)
    "routes": [],                       # [{"profile": name, "description": text}]
}

PROFILE_DEFAULTS = {
    "name": "",
    "model": "",
    "gpus": "auto:1",          # "auto:N" = N least-used GPUs, or "0,1"
    "ctx": 32768,
    "threads": "auto",         # "auto" = one per physical core on the best socket
    "pin": "auto",             # "auto" | "off" | explicit cpu list "0,8,9"
    "fit": True,               # let llama.cpp --fit choose GPU layers / expert offload
    "fit_target_mib": 2300,    # free VRAM to leave per GPU when fit=true
    "n_cpu_moe": 32,           # used when fit=false
    "kv_quant": "q8_0",        # "" = f16
    "extra_args": "-fa on --load-mode none",
    "llama_server": "",        # "" = the global llama_server; set a path to use another build for this profile
}


class Config:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        self.load()

    def load(self):
        with open(self.path) as f:
            raw = json.load(f)
        self.data = self.normalize(raw)

    @staticmethod
    def normalize(raw):
        d = dict(DEFAULTS)
        d.update(raw)
        d["profiles"] = [dict(PROFILE_DEFAULTS, **p) for p in d.get("profiles", [])]
        d["kev"] = dict(KEV_DEFAULTS, **(raw.get("kev") or {}))
        d["router"] = dict(ROUTER_DEFAULTS, **(raw.get("router") or {}))
        d["router"]["routes"] = [dict(r) for r in d["router"]["routes"]]
        return d

    @staticmethod
    def validate(d):
        names = [p["name"] for p in d["profiles"]]
        if not names:
            raise ValueError("at least one profile is required")
        if any(not n or not re.fullmatch(r"[A-Za-z0-9._-]+", n) for n in names):
            raise ValueError("profile names must be non-empty [A-Za-z0-9._-]")
        if len(set(names)) != len(names):
            raise ValueError("profile names must be unique")
        if d["active_profile"] not in names:
            raise ValueError("active_profile must be one of the profile names")
        for p in d["profiles"]:
            if not p["model"]:
                raise ValueError(f"profile {p['name']}: model path is empty")
            g = str(p["gpus"]).strip()
            if not (re.fullmatch(r"auto:\d+", g) or re.fullmatch(r"\d+(,\d+)*", g) or g in ("", "none")):
                raise ValueError(f"profile {p['name']}: gpus must be auto:N, a list like 0,1, or none")
            shlex.split(p.get("extra_args") or "")
        r, k = d["router"], d["kev"]
        for key, lo in (("gpu_bucket_mib", 1), ("gpu_fallback_count", 1), ("download_mbps", 0.01)):
            try:
                ok = float(d[key]) >= lo
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"{key} must be a number >= {lo}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", str(r["alias"])) or r["alias"] in names:
            raise ValueError("router alias must be [A-Za-z0-9._-] and not a profile name")
        if not 0 <= float(r["threshold"]) <= 1:
            raise ValueError("router threshold must be between 0 and 1")
        if not re.fullmatch(r"auto:\d+|\d+(,\d+)*", str(k["gpus"]).strip()):
            raise ValueError("kev gpus must be auto:N or a list like 5,6")
        if str(k["dtype"]) not in ("fp32", "fp16", "bf16"):
            raise ValueError("kev dtype must be fp32, fp16 or bf16")
        if str(k["cuda_graphs"]) not in ("auto", "off", "on"):
            raise ValueError("kev cuda_graphs must be auto, off or on")
        if k["managed"] and not (str(k["model"]).strip() or str(k["command"]).strip()):
            raise ValueError("kev model is empty")
        if r["enabled"]:
            used = [x.get("profile") for x in r["routes"]]
            if len(used) < 2:
                raise ValueError("router needs at least two routes")
            if len(set(used)) != len(used) or any(u not in names for u in used):
                raise ValueError("router routes must name distinct, existing profiles")
            if any(not str(x.get("description", "")).strip() for x in r["routes"]):
                raise ValueError("every route needs a description")
            if r["default_profile"] not in names:
                raise ValueError("router default_profile must be an existing profile")
        if not str(d.get("hf_endpoint") or "").startswith(("http://", "https://")):
            raise ValueError("hf_endpoint must be an http(s) URL")
        if not isinstance(d["model_dirs"], list):
            raise ValueError("model_dirs must be a list")
        try:
            if float(d["min_free_gib_for_download"]) < 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("min_free_gib_for_download must be a number >= 0") from None
        if not d["api_keys"]:
            raise ValueError("api_keys must contain at least one key")
        if not d["admin_password"]:
            raise ValueError("admin_password must be set")

    def save(self, new, clear_token=False):
        d = self.normalize(new)
        d.pop("has_hf_token", None)         # echoed by GET /api/config, never stored
        if clear_token:
            d["hf_token"] = ""
        elif not d.get("hf_token"):
            d["hf_token"] = self.data.get("hf_token", "")   # empty or omitted keeps the saved token
        self.validate(d)
        with self.lock:
            if os.path.exists(self.path):
                shutil.copy2(self.path, self.path + ".bak")
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
            self.data = d

    def get(self, k):
        return self.data[k]

    def profile(self, name=None):
        name = name or self.data["active_profile"]
        for p in self.data["profiles"]:
            if p["name"] == name:
                return p
        raise KeyError(name)


# ---------------------------------------------------------------- host info

def read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def cpu_topology():
    """Allowed CPUs grouped by socket -> {core_id: [cpu,...]}, plus numa node per cpu."""
    allowed = sorted(os.sched_getaffinity(0))
    sockets, node_of = {}, {}
    for c in allowed:
        base = f"/sys/devices/system/cpu/cpu{c}"
        pkg = int(read(f"{base}/topology/physical_package_id", "0") or 0)
        core = int(read(f"{base}/topology/core_id", str(c)) or c)
        sockets.setdefault(pkg, {}).setdefault(core, []).append(c)
        nodes = glob.glob(f"{base}/node*")
        node_of[c] = int(os.path.basename(nodes[0])[4:]) if nodes else pkg
    return allowed, sockets, node_of


def auto_pin():
    """One CPU per physical core on the socket where we own the most cores."""
    allowed, sockets, node_of = cpu_topology()
    if not sockets:
        return allowed, None
    pkg = max(sockets, key=lambda s: (len(sockets[s]), -s))
    cpus = sorted(min(v) for v in sockets[pkg].values())
    return cpus, node_of.get(cpus[0])


def gpu_info(timeout=10):
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue
        num = lambda s: int(float(s)) if re.fullmatch(r"[\d.]+", s) else None
        gpus.append({"index": int(p[0]), "name": p[1], "mem_used": num(p[2]), "mem_total": num(p[3]),
                     "util": num(p[4]), "temp": num(p[5])})
    return gpus


def pick_gpus(spec, own=(), exclude=(), bucket=512, fallback=8):
    """Resolve a gpus spec. For auto:N, GPUs we already hold count as free and `exclude` is skipped."""
    spec = str(spec).strip()
    if spec in ("", "none"):
        return []
    m = re.fullmatch(r"auto:(\d+)", spec)
    if not m:
        return [int(x) for x in spec.split(",")]
    n = int(m.group(1))
    gpus = [g for g in gpu_info() if g["index"] not in exclude]
    if not gpus:
        return [i for i in range(fallback) if i not in exclude][-n:]
    # Least-used first in `bucket` MiB steps (512 by default); ties go to the highest index, since
    # other users' jobs default to GPU 0 and up.
    bucket = max(1, int(bucket or 512))
    ranked = sorted(gpus, key=lambda g: (0 if g["index"] in own else (g["mem_used"] or 0) // bucket, -g["index"]))
    return sorted(g["index"] for g in ranked[:n])


def _kb_map(text):
    """Parse '/proc/meminfo'-style lines ('Key:  123 kB') into {key: MiB}; tolerant of junk."""
    m = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        if parts:
            try:
                m[k.strip()] = int(parts[0]) // 1024
            except ValueError:
                pass
    return m


def numa_meminfo():
    """Per-NUMA-node memory from sysfs (host-wide even inside LXD, unlike lxcfs's /proc/meminfo)."""
    nodes = []
    for path in sorted(glob.glob("/sys/devices/system/node/node[0-9]*/meminfo"),
                       key=lambda p: int(re.search(r"node(\d+)", p).group(1))):
        try:
            idx = int(re.search(r"node(\d+)", path).group(1))
            # lines look like 'Node 0 MemTotal:  131072 kB'
            m = _kb_map(re.sub(r"^Node\s+\d+\s+", "", read(path), flags=re.M))
            tot, free = m.get("MemTotal"), m.get("MemFree")
            if tot is None or free is None:
                continue
            cached = m.get("FilePages", 0)
            nodes.append({"node": idx, "total_mib": tot, "free_mib": free, "used_mib": tot - free,
                          "cached_mib": cached, "avail_mib": free + min(cached, tot - free)})
        except Exception:  # noqa: BLE001 - metrics must never raise
            continue
    return nodes


def proc_stats(pid):
    """RSS and cumulative read bytes of a process; each field None when unreadable."""
    out = {"rss_mib": None, "read_bytes": None, "rchar": None}
    if not pid:
        return out
    try:
        for line in read(f"/proc/{pid}/status").splitlines():
            if line.startswith("VmRSS:"):
                out["rss_mib"] = int(line.split()[1]) // 1024
    except (ValueError, IndexError):
        pass
    try:
        for line in read(f"/proc/{pid}/io").splitlines():
            if line.startswith("read_bytes:"):
                out["read_bytes"] = int(line.split()[1])
            elif line.startswith("rchar:"):
                out["rchar"] = int(line.split()[1])   # bytes via read(): what --load-mode none does
    except (ValueError, IndexError):
        pass
    return out


SHARD_RE = re.compile(r"^(.*)-(\d+)-of-(\d+)\.gguf$")


def model_total_bytes(path):
    """Total size of a model: all `-0000N-of-0000M.gguf` shards next to each other, or the single file.
    None when nothing is readable. Never raises."""
    try:
        m = SHARD_RE.match(path)
        if m:
            w, n = len(m.group(2)), int(m.group(3))
            files = [f"{m.group(1)}-{i:0{w}d}-of-{m.group(3)}.gguf" for i in range(1, n + 1)]
        else:
            files = [path]
        tot, found = 0, 0
        for f in files:
            try:
                tot += os.path.getsize(f)
                found += 1
            except OSError:
                pass
        return tot if found else None
    except Exception:  # noqa: BLE001
        return None

# -- startup log -> layer/buffer map (best effort; every pattern optional, builds differ)
OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", re.I)
NLAYER_RE = re.compile(r"\bn_layer\s*=\s*(\d+)")
BUF_RE = re.compile(r"(?:^|[\s:])([A-Za-z][\w.\-]*?)\s+(model|KV|compute|RS)\s+buffer size\s*=\s*([\d.]+)\s*(KiB|MiB|GiB|B)\b", re.I)
LAYER_RE = re.compile(r"\blayer\s+(\d+)\s+assigned to device\s+([\w.\-]+?)(?=[,\s]|$)", re.I)
FIT_RE = re.compile(r"params_fit|projected|n_layer|overflow|\bfit(?:ting|ted)?\b", re.I)
HOSTLIKE_RE = re.compile(r"^(cpu|host)|_(host|mapped|repack)$", re.I)
_UNIT = {"b": 1 / 1048576, "kib": 1 / 1024, "mib": 1.0, "gib": 1024.0}


def parse_startup_log(lines):
    """Parse llama-server startup output into {profile, layers_total, layers_offloaded, devices, cpu_mib, cpu,
    per_layer, fit}. Looks only at the part after the last '[ctl] starting profile' line. Never raises; anything
    not found stays None / empty."""
    out = {"profile": None, "layers_total": None, "layers_offloaded": None, "devices": [], "cpu_mib": None,
           "cpu": {"weights_mib": None, "kv_mib": None, "compute_mib": None},
           "per_layer": [], "fit": {"raw": []}, "have_data": False}
    try:
        lines = [l for l in lines if isinstance(l, str)]
        for i in range(len(lines) - 1, -1, -1):
            m = re.search(r"\[ctl\] starting profile (\S+)", lines[i])
            if m:
                out["profile"], lines = m.group(1), lines[i + 1:]
                break
        bufs, layers, nlayer = {}, {}, None
        for line in lines:
            if line.startswith("[ctl]"):
                continue
            m = OFFLOAD_RE.search(line)
            if m:
                out["layers_offloaded"], out["layers_total"] = int(m.group(1)), int(m.group(2))
            m = NLAYER_RE.search(line)
            if m:
                nlayer = int(m.group(1))
            m = BUF_RE.search(line)
            if m:
                kind = {"model": "weights", "kv": "kv", "compute": "compute", "rs": "kv"}[m.group(2).lower()]
                bufs[(m.group(1), kind)] = float(m.group(3)) * _UNIT[m.group(4).lower()]   # a re-print replaces
            m = LAYER_RE.search(line)
            if m:
                layers[int(m.group(1))] = m.group(2)
            if FIT_RE.search(line) and "print_info" not in line and len(out["fit"]["raw"]) < 12:
                out["fit"]["raw"].append(line.strip()[:300])
        if out["layers_total"] is None:
            out["layers_total"] = nlayer
        devs, cpu = {}, {"weights": 0.0, "kv": 0.0, "compute": 0.0}
        seen_cpu = False
        for (name, kind), mib in bufs.items():
            if HOSTLIKE_RE.search(name):
                cpu[kind] += mib
                seen_cpu = True
            else:
                devs.setdefault(name, {"name": name, "weights_mib": None, "kv_mib": None, "compute_mib": None})[kind + "_mib"] = round(mib, 1)
        def natural(d):
            m = re.search(r"(\d+)$", d["name"])
            return (d["name"].rstrip("0123456789"), int(m.group(1)) if m else -1)
        out["devices"] = sorted(devs.values(), key=natural)
        for d in out["devices"]:
            m = re.search(r"(\d+)$", d["name"])
            d["index"] = int(m.group(1)) if m else None
        if seen_cpu:
            out["cpu"] = {"weights_mib": round(cpu["weights"], 1), "kv_mib": round(cpu["kv"], 1), "compute_mib": round(cpu["compute"], 1)}
            out["cpu_mib"] = round(sum(cpu.values()), 1)
        out["per_layer"] = [{"layer": k, "device": layers[k]} for k in sorted(layers)]
        out["have_data"] = bool(out["devices"] or seen_cpu or out["layers_offloaded"] is not None or layers or out["fit"]["raw"])
    except Exception:  # noqa: BLE001
        pass
    return out



def cgroup_mem():
    """Container cgroup usage/limit (cgroup v2, then v1); None when unavailable."""
    for cur, lim in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                     ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        c = read(cur)
        if c.isdigit():
            l = read(lim)
            limit = int(l) // 2**20 if l.isdigit() and int(l) < 2**60 else None
            return {"used_mib": int(c) // 2**20, "limit_mib": limit}
    return None


def meminfo(pid=None):
    """Memory snapshot. Never raises. total/available keep their old names; the rest is extra.
    Note 'available' counts reclaimable page cache (an mmap'd model is cache), so it barely
    moves when a model loads; 'used' (= total - available) and the NUMA node numbers do."""
    try:
        m = _kb_map(read("/proc/meminfo"))
    except Exception:  # noqa: BLE001
        m = {}
    tot, avail = m.get("MemTotal"), m.get("MemAvailable")
    out = {"total_mib": tot, "available_mib": avail,
           "used_mib": tot - avail if tot is not None and avail is not None else None,
           "free_mib": m.get("MemFree"), "buffers_mib": m.get("Buffers"), "cached_mib": m.get("Cached"),
           "swap_total_mib": m.get("SwapTotal"),
           "swap_used_mib": (m["SwapTotal"] - m["SwapFree"]) if "SwapTotal" in m and "SwapFree" in m else None,
           "nodes": [], "cgroup": None, "child": proc_stats(pid), "ts": time.time()}
    try:
        out["nodes"] = numa_meminfo()
        out["cgroup"] = cgroup_mem()
    except Exception:  # noqa: BLE001
        pass
    # host-wide reclaimable memory: sum over NUMA nodes of free + file cache (None without node data)
    out["host_free_mib"] = sum(n["avail_mib"] for n in out["nodes"]) if out["nodes"] else None
    return out


# ---------------------------------------------------------------- system info (GET /api/system) and settings

PROBE_TIMEOUT = 3          # seconds; every subprocess used for hardware/tool detection is capped at this


def _run_probe(cmd, timeout=PROBE_TIMEOUT):
    """stdout of a short probe command, or None on any failure (missing, timeout, non-zero exit)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    return r.stdout if r.returncode == 0 else None


def parse_cpulist(text):
    """'0,8-9,13' -> [0, 8, 9, 13]; junk pieces are skipped."""
    out = []
    for part in str(text or "").strip().split(","):
        part = part.strip()
        try:
            if "-" in part:
                a, _, b = part.partition("-")
                out.extend(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
        except ValueError:
            continue
    return sorted(set(out))


def parse_cpuinfo(text):
    """/proc/cpuinfo text -> {model, threads (processor entries), sockets (distinct physical ids, or None)}."""
    model, procs, pkgs = None, 0, set()
    for line in str(text or "").splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "model name" and model is None and v:
            model = re.sub(r"\s+", " ", v)
        elif k == "processor":
            procs += 1
        elif k == "physical id" and v.isdigit():
            pkgs.add(int(v))
    return {"model": model, "threads": procs or None, "sockets": len(pkgs) or None}


def os_pretty_name(text):
    for line in str(text or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"') or None
    return None


def numa_nodes(root="/sys/devices/system/node"):
    """[{id, cpus, mem_total_mib}] from sysfs; [] when there is no NUMA information."""
    out = []
    for path in glob.glob(os.path.join(root, "node[0-9]*")):
        try:
            idx = int(re.search(r"node(\d+)$", path).group(1))
            m = _kb_map(re.sub(r"^Node\s+\d+\s+", "", read(os.path.join(path, "meminfo")), flags=re.M))
            out.append({"id": idx, "cpus": parse_cpulist(read(os.path.join(path, "cpulist"))),
                        "mem_total_mib": m.get("MemTotal")})
        except Exception:  # noqa: BLE001
            continue
    return sorted(out, key=lambda n: n["id"])


def parse_smi_gpus(text, with_cap=True):
    """nvidia-smi csv (index,name,memory.total,driver_version[,compute_cap]) -> gpu dicts. Unknown fields are None."""
    gpus = []
    for line in str(text or "").strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4 or not p[0].isdigit():
            continue
        vram = int(float(p[2])) if re.fullmatch(r"[\d.]+", p[2]) else None
        cap = p[4] if with_cap and len(p) > 4 and re.fullmatch(r"\d+\.\d+", p[4]) else None
        gpus.append({"index": int(p[0]), "name": p[1] or None, "vram_mib": vram,
                     "driver": p[3] or None, "compute_cap": cap})
    return gpus


def smi_cuda_version(text):
    m = re.search(r"CUDA Version:\s*([\d.]+)", str(text or ""))
    return m.group(1) if m else None


def detect_gpus():
    """GPU list with driver / compute capability / CUDA version where nvidia-smi can say; [] without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    q = "--query-gpu=index,name,memory.total,driver_version"
    out = _run_probe(["nvidia-smi", q + ",compute_cap", "--format=csv,noheader,nounits"])
    gpus = parse_smi_gpus(out, True) if out is not None else None
    if gpus is None:        # older drivers reject the compute_cap field
        gpus = parse_smi_gpus(_run_probe(["nvidia-smi", q, "--format=csv,noheader,nounits"]), False)
    cuda = smi_cuda_version(_run_probe(["nvidia-smi"]))
    for g in gpus:
        if cuda:
            g["cuda"] = cuda
    return gpus


def disk_usage(path):
    """{free_gib, total_gib, writable, exists} for the nearest existing ancestor of path; unknown -> None."""
    out = {"free_gib": None, "total_gib": None, "writable": None, "exists": False}
    try:
        p = os.path.expanduser(str(path))
        out["exists"] = os.path.isdir(p)
        while p and not os.path.exists(p) and os.path.dirname(p) != p:
            p = os.path.dirname(p)
        st = os.statvfs(p)
        out["free_gib"] = round(st.f_bavail * st.f_frsize / 2**30, 1)
        out["total_gib"] = round(st.f_blocks * st.f_frsize / 2**30, 1)
        out["writable"] = os.access(p, os.W_OK | os.X_OK)
    except (OSError, ValueError):
        pass
    return out


def effective_download_dir(cfg):
    d = str(cfg.get("download_dir") or "").strip()
    if d:
        return d
    md = cfg.get("model_dirs") or []
    return md[0] if md else ""


def effective_hf_home(cfg):
    h = str((cfg.get("kev") or {}).get("hf_home") or "").strip()
    return h or "~/.cache/huggingface"


_VERSION_CACHE = {}


def tool_version(path, ttl=60, timeout=PROBE_TIMEOUT, now=None):
    """First useful line of `<path> --version` (stdout+stderr, llama.cpp prints to stderr); cached `ttl` s per path."""
    now = time.time() if now is None else now
    hit = _VERSION_CACHE.get(path)
    if hit and now - hit[0] < ttl:
        return hit[1]
    ver = None
    try:
        r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=timeout)
        txt = (r.stdout or "") + "\n" + (r.stderr or "")
        if r.returncode == 0:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            pick = [l for l in lines if re.search(r"version|build", l, re.I)] or lines
            ver = pick[0][:200] if pick else None
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    _VERSION_CACHE[path] = (now, ver)
    return ver


def tools_info(cfg):
    ls = os.path.expanduser(str(cfg.get("llama_server") or ""))
    exists = bool(ls) and os.path.isfile(ls)
    exe = exists and os.access(ls, os.X_OK)
    ver = tool_version(ls) if exe else None
    return {"llama_server": {"path": ls or None, "exists": exists, "executable": bool(exe), "version": ver},
            "numactl": shutil.which("numactl"), "taskset": shutil.which("taskset"),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "hf_cli": bool(shutil.which("hf") or shutil.which("huggingface-cli"))}


def system_info(cfg):
    """Everything GET /api/system returns, minus secrets. Never raises: each block degrades to None/[]."""
    def safe(fn, default):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            return default
    import platform
    ci = safe(lambda: parse_cpuinfo(read("/proc/cpuinfo")), {"model": None, "threads": None, "sockets": None})
    allowed, sockets, _ = safe(cpu_topology, ([], {}, {}))
    cores = sum(len(v) for v in sockets.values()) or None
    pin_cpus, pin_node = safe(auto_pin, ([], None))
    mi = safe(meminfo, {})
    cg = mi.get("cgroup") or {}
    nodes = safe(numa_nodes, [])
    gpus = safe(detect_gpus, [])
    disks = []
    seen = set()
    for role, path in [("download", effective_download_dir(cfg))] + [("model_dir", d) for d in (cfg.get("model_dirs") or [])] \
            + [("hf_home", effective_hf_home(cfg))]:
        if not path or (role, path) in seen:
            continue
        seen.add((role, path))
        du = safe(lambda p=path: disk_usage(p), {})
        disks.append({"path": path, "role": role, "free_gib": du.get("free_gib"), "total_gib": du.get("total_gib"),
                      "writable": du.get("writable"), "exists": du.get("exists")})
    return {
        "os": safe(lambda: os_pretty_name(read("/etc/os-release")), None) or safe(platform.platform, None),
        "kernel": safe(lambda: platform.release(), None), "python": safe(platform.python_version, None),
        "hostname": safe(platform.node, None),
        "cpu": {"model": ci["model"], "sockets": ci["sockets"] or (len(sockets) or None),
                "physical_cores": cores, "threads": ci["threads"] or (len(allowed) or None),
                "allowed_cpus": allowed, "numa_nodes": nodes,
                "pinned_default": {"cpus": pin_cpus, "node": pin_node}},
        "ram": {"total_mib": mi.get("total_mib"), "cgroup_limit_mib": cg.get("limit_mib"), "nodes": mi.get("nodes") or []},
        "gpus": gpus, "disks": disks, "tools": safe(lambda: tools_info(cfg), {}),
        "settings": public_settings(cfg),
        "ts": time.time(),
    }


SETTING_KEYS = ("download_dir", "model_dirs", "hf_home", "llama_server", "hf_endpoint", "hf_token",
                "clear_hf_token", "min_free_gib_for_download")


def public_settings(cfg):
    """The editable settings as the UI shows them. The token is never included, only whether one is set."""
    return {"download_dir": cfg.get("download_dir") or "", "effective_download_dir": effective_download_dir(cfg),
            "model_dirs": list(cfg.get("model_dirs") or []), "hf_home": (cfg.get("kev") or {}).get("hf_home") or "",
            "llama_server": cfg.get("llama_server") or "", "hf_endpoint": cfg.get("hf_endpoint") or DEFAULTS["hf_endpoint"],
            "has_hf_token": bool(cfg.get("hf_token")),
            "min_free_gib_for_download": cfg.get("min_free_gib_for_download")}


def public_config(data):
    """Config for GET /api/config: the secret token is replaced by has_hf_token."""
    d = dict(data)
    d["has_hf_token"] = bool(d.pop("hf_token", ""))
    return d


def _abs_path(label, v, allow_empty=False):
    v = str(v if v is not None else "").strip()
    if not v:
        if allow_empty:
            return ""
        raise ValueError(f"{label} is required")
    if "\x00" in v or "\n" in v:
        raise ValueError(f"{label} contains invalid characters")
    if not os.path.isabs(os.path.expanduser(v)):
        raise ValueError(f"{label} must be an absolute path")
    return v


def check_writable_dir(path, create):
    """Prove a directory is writable by creating and removing a temp file. With create=False an absent
    directory is judged by its nearest existing ancestor and nothing is created. Raises ValueError."""
    p = os.path.expanduser(path)
    target = p
    if not os.path.isdir(p):
        if os.path.exists(p):
            raise ValueError(f"{path} exists and is not a directory")
        if create:
            try:
                os.makedirs(p, exist_ok=True)
            except OSError as e:
                raise ValueError(f"cannot create {path}: {e.strerror or e}") from None
        else:
            while not os.path.isdir(target) and os.path.dirname(target) != target:
                target = os.path.dirname(target)
    try:
        fd, tmp = tempfile.mkstemp(prefix=".bizen-write-test-", dir=target)
        os.close(fd)
        os.remove(tmp)
    except OSError as e:
        raise ValueError(f"{path} is not writable: {e.strerror or e}") from None


def validate_settings(upd, current, create=True):
    """Validate a settings update against the current config. Returns (changes, warnings, checks).
    `changes` is a dict of normalized values for the keys present in `upd`; omitted keys are left alone.
    Raises ValueError (message never contains the token)."""
    changes, warnings, checks = {}, [], {}
    if "model_dirs" in upd:
        md = upd["model_dirs"]
        if isinstance(md, str):
            md = [x for x in md.splitlines()]
        if not isinstance(md, list) or not md:
            raise ValueError("model_dirs needs at least one directory")
        dirs = []
        for x in md:
            x = _abs_path("model_dirs entry", x)
            if x not in dirs:
                dirs.append(x)
            if not os.path.isdir(os.path.expanduser(x)):
                warnings.append(f"model dir {x} does not exist right now")
        changes["model_dirs"] = dirs
    if "download_dir" in upd:
        dd = _abs_path("download_dir", upd["download_dir"], allow_empty=True)
        if dd:
            check_writable_dir(dd, create)
        changes["download_dir"] = dd
        checks["download_dir"] = "writable" if dd else "default (first model dir)"
    if "hf_home" in upd:
        hh = _abs_path("hf_home", upd["hf_home"], allow_empty=True)
        if hh and not os.path.isdir(os.path.expanduser(hh)):
            warnings.append(f"HF cache {hh} does not exist yet")
        changes["hf_home"] = hh
    if "llama_server" in upd:
        ls = _abs_path("llama_server", upd["llama_server"])
        p = os.path.expanduser(ls)
        if not os.path.isfile(p):
            raise ValueError(f"llama_server {ls} is not a file")
        if not os.access(p, os.X_OK):
            raise ValueError(f"llama_server {ls} is not executable")
        ver = tool_version(p, ttl=0)
        checks["llama_server_version"] = ver
        if ver is None:
            warnings.append("llama-server --version failed or timed out; the path is executable, so it was accepted")
        changes["llama_server"] = ls
    if "hf_endpoint" in upd:
        ep = str(upd["hf_endpoint"] or "").strip().rstrip("/") or DEFAULTS["hf_endpoint"]
        u = urllib.parse.urlsplit(ep)
        if u.scheme not in ("http", "https") or not u.netloc or any(c.isspace() for c in ep):
            raise ValueError("hf_endpoint must be an http(s) URL")
        changes["hf_endpoint"] = ep
    if "min_free_gib_for_download" in upd:
        try:
            v = float(upd["min_free_gib_for_download"])
        except (TypeError, ValueError):
            raise ValueError("min_free_gib_for_download must be a number") from None
        if v < 0 or v != v or v > 1e7:
            raise ValueError("min_free_gib_for_download must be >= 0")
        changes["min_free_gib_for_download"] = int(v) if v == int(v) else v
    tok = upd.get("hf_token")
    if tok not in (None, ""):
        if not isinstance(tok, str) or len(tok) > 512 or re.search(r"\s|[\x00-\x1f]", tok):
            raise ValueError("hf_token must be a single token without whitespace")
        changes["hf_token"] = tok
    if upd.get("clear_hf_token") and "hf_token" in changes:
        raise ValueError("send either hf_token or clear_hf_token, not both")
    return changes, warnings, checks


def apply_settings(cfg, upd, dry_run=False):
    """Validate and (unless dry_run) persist a settings update through Config.save, the same path as PUT /api/config.
    Nothing here restarts llama-server or Kev. Returns {ok, saved, warnings, checks, settings}."""
    changes, warnings, checks = validate_settings(upd, cfg.data, create=not dry_run)
    if not dry_run:
        new = json.loads(json.dumps(cfg.data))
        kev_hf = changes.pop("hf_home", None)
        new.update(changes)
        if kev_hf is not None:
            new["kev"]["hf_home"] = kev_hf
        cfg.save(new, clear_token=bool(upd.get("clear_hf_token")))
    return {"ok": True, "saved": not dry_run, "warnings": warnings, "checks": checks, "settings": public_settings(cfg.data)}


# ---------------------------------------------------------------- library (GET /api/library, delete, draft profile)

QUANT_RE = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))(?![A-Za-z0-9])", re.I)
PARTIAL_SUFFIXES = (".part", ".incomplete")
FIT_OVERHEAD = 0.10      # share of VRAM / RAM assumed lost to KV cache, compute buffers and the OS
FIT_NOTE = "Rough size-based estimate, not a speed prediction."


def parse_quant(name):
    """Quantisation token from a model base name ('UD-Q3_K_XL', 'Q4_K_M', 'IQ2_XXS', 'BF16', ...), else None."""
    try:
        found = QUANT_RE.findall(str(name or ""))
        return found[-1].upper() if found else None
    except Exception:  # noqa: BLE001
        return None


def _norm(p):
    try:
        return os.path.realpath(os.path.expanduser(str(p)))
    except (OSError, ValueError):
        return str(p)


def _inside(path, root):
    """True when the (already resolved) path is root itself or below it."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def library_roots(cfg):
    """[(walk_path, realpath)] for model_dirs + download_dir: existing directories only, deduped by realpath."""
    out, seen = [], set()
    dirs = list(cfg.get("model_dirs") or []) + [str(cfg.get("download_dir") or "").strip()]
    for d in dirs:
        if not d:
            continue
        try:
            p = os.path.abspath(os.path.expanduser(str(d)))
            rp = os.path.realpath(p)
            if rp in seen or not os.path.isdir(rp):
                continue
            seen.add(rp)
            out.append((p, rp))
        except (OSError, ValueError):
            pass
    return out


def fit_verdict(size_bytes, vram_mib, ram_mib):
    """Plain arithmetic. vram_mib: total over all GPUs (None/0 when unknown); ram_mib: system RAM (or None).
    fits-vram: weights <= 90% of VRAM; needs-ram: weights <= 90% of VRAM + 90% of RAM; else too-big. None without GPU data."""
    if not size_bytes or not vram_mib:
        return {"vram_total_mib": vram_mib or None, "ram_total_mib": ram_mib, "verdict": None,
                "text": "No GPU data or model size, so no estimate. " + FIT_NOTE}
    size = size_bytes / 1048576
    use_v = vram_mib * (1 - FIT_OVERHEAD)
    use_r = (ram_mib or 0) * (1 - FIT_OVERHEAD)
    sg = f"{size / 1024:.1f} GiB"
    if size <= use_v:
        v, t = "fits-vram", f"{sg} of weights vs {use_v / 1024:.1f} GiB usable VRAM ({vram_mib / 1024:.1f} GiB minus 10%): should fit in VRAM."
    elif size <= use_v + use_r:
        v, t = "needs-ram", f"{sg} of weights vs {use_v / 1024:.1f} GiB usable VRAM: the rest ({(size - use_v) / 1024:.1f} GiB) has to sit in system RAM ({use_r / 1024:.1f} GiB usable), so expect CPU offload."
    else:
        v, t = "too-big", f"{sg} of weights vs {(use_v + use_r) / 1024:.1f} GiB usable VRAM + RAM combined: too big for this machine."
    return {"vram_total_mib": vram_mib, "ram_total_mib": ram_mib, "verdict": v, "text": t + " " + FIT_NOTE}


def scan_library(cfg, gpu_mems=None, ram_mib=None, loaded_model=None):
    """Models on disk under model_dirs + download_dir. Never raises; unreadable bits are skipped.
    cfg: plain config dict; gpu_mems: per-GPU total MiB; loaded_model: model path of the running/starting profile."""
    gpu_mems = [m for m in (gpu_mems or []) if m]
    vram = sum(gpu_mems) or None
    try:
        roots = library_roots(cfg)
    except Exception:  # noqa: BLE001
        roots = []
    real_roots = [rp for _, rp in roots]
    seen = set()
    groups = {}            # (dir, prefix, total) or (path,) -> record
    mmprojs = {}           # dir -> [path]
    partials = {}          # (dir, prefix, total) -> {index: path}
    loose_partials = []    # (dir, path, base)

    def okfile(path):
        """stat of a regular file whose resolved path stays inside a model root; None otherwise (also for repeats)."""
        try:
            rp = os.path.realpath(path)
            if rp in seen or not any(_inside(rp, r) for r in real_roots):
                return None
            st = os.stat(path)
            if not stat.S_ISREG(st.st_mode):
                return None
            seen.add(rp)
            return st
        except (OSError, ValueError):
            return None

    def handle(dp, fn):
        low = fn.lower()
        path = os.path.join(dp, fn)
        if low.endswith(PARTIAL_SUFFIXES):
            base = fn[:fn.rfind(".")]
            if ".gguf" not in low or okfile(path) is None:
                return
            sm = SHARD_RE.match(base)
            if sm:
                partials.setdefault((dp, sm.group(1), sm.group(3)), {})[int(sm.group(2))] = path
            else:
                loose_partials.append((dp, path, base))
            return
        if not low.endswith(".gguf"):
            return
        st = okfile(path)
        if st is None:
            return
        if "mmproj" in low:
            mmprojs.setdefault(dp, []).append(path)
            return
        sm = SHARD_RE.match(fn)
        if sm:
            g = groups.setdefault((dp, sm.group(1), sm.group(3)), {
                "dir": dp, "prefix": sm.group(1), "tstr": sm.group(3), "total": int(sm.group(3)),
                "width": len(sm.group(2)), "shards": {}})
            g["shards"][int(sm.group(2))] = (path, st.st_size, st.st_mtime)
        else:
            groups[(path,)] = {"dir": dp, "prefix": fn[:-5], "tstr": "1", "total": 1, "width": 0,
                               "single": path, "shards": {1: (path, st.st_size, st.st_mtime)}}

    later = []             # symlinked files go last so a real file is always listed under its own name
    for wp, _rp in roots:
        try:
            for dp, dns, fns in os.walk(wp, followlinks=False):
                dns.sort()
                for fn in sorted(fns):
                    try:
                        if os.path.islink(os.path.join(dp, fn)):
                            later.append((dp, fn))
                        else:
                            handle(dp, fn)
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            continue
    for dp, fn in later:
        try:
            handle(dp, fn)
        except Exception:  # noqa: BLE001
            continue

    profs = []
    for p in cfg.get("profiles") or []:
        try:
            profs.append((p.get("name"), _norm(p.get("model") or "") if p.get("model") else None))
        except Exception:  # noqa: BLE001
            pass
    loaded_n = _norm(loaded_model) if loaded_model else None

    def unfit(why):
        return {"vram_total_mib": vram, "ram_total_mib": ram_mib, "verdict": None, "text": why + " " + FIT_NOTE}

    models = []
    for g in groups.values():
        try:
            total, sh = g["total"], g["shards"]
            if g.get("single"):
                first = g["single"]
            elif 1 in sh:
                first = sh[1][0]
            else:
                first = os.path.join(g["dir"], f"{g['prefix']}-{1:0{g['width']}d}-of-{g['tstr']}.gguf")
            found = len(sh)
            shards = [{"path": sh[i][0], "bytes": sh[i][1]} for i in sorted(sh)]
            size = sum(s["bytes"] for s in shards)
            missing = [i for i in range(1, total + 1) if i not in sh]
            part = partials.get((g["dir"], g["prefix"], g["tstr"]), {})
            hint = None
            if missing:
                still = [i for i in missing if i in part]
                hint = f"{len(missing)} of {total} shards missing"
                if still:
                    hint += " (shard " + ", ".join(map(str, still)) + " is still a partial download)"
                hint += ": finish the download or delete this set."
            mm = sorted(mmprojs.get(g["dir"], []), key=lambda p: ("f16" not in os.path.basename(p).lower(), p))
            names = {_norm(s["path"]) for s in shards}
            used = [n for n, mp in profs if mp and mp in names]
            models.append({
                "id": hashlib.sha1(first.encode("utf-8", "replace")).hexdigest()[:12],
                "name": g["prefix"], "quant": parse_quant(g["prefix"]), "dir": g["dir"], "first_shard": first,
                "shards": shards, "shards_found": found, "shards_expected": total, "complete": found == total,
                "bytes": size, "gib": round(size / 2**30, 2), "mtime": max(s[2] for s in sh.values()),
                "mmproj": mm[0] if mm else None, "used_by": used, "loaded": bool(loaded_n and loaded_n in names),
                "partial": False, "hint": hint,
                "fit": fit_verdict(size, vram, ram_mib) if found == total else unfit("Incomplete set, so no estimate.")})
        except Exception:  # noqa: BLE001
            continue
    for dp, path, base in loose_partials:
        try:
            st = os.stat(path)
            models.append({
                "id": hashlib.sha1(path.encode("utf-8", "replace")).hexdigest()[:12],
                "name": base[:-5] if base.lower().endswith(".gguf") else base, "quant": parse_quant(base),
                "dir": dp, "first_shard": path, "shards": [{"path": path, "bytes": st.st_size}],
                "shards_found": 0, "shards_expected": 1, "complete": False, "bytes": st.st_size,
                "gib": round(st.st_size / 2**30, 2), "mtime": st.st_mtime, "mmproj": None, "used_by": [],
                "loaded": False, "partial": True,
                "hint": "Partial download (" + os.path.splitext(path)[1] + "): resume it or remove the file by hand.",
                "fit": unfit("Incomplete download, so no estimate.")})
        except OSError:
            pass
    models.sort(key=lambda m: (-m["mtime"], m["name"]))
    return {"models": models, "roots": [p for p, _ in roots], "ts": time.time()}


def live_gpu_mems():
    """Total MiB of each detected GPU ([] without nvidia-smi). Never raises."""
    try:
        return [g.get("mem_total") for g in gpu_info(timeout=3) if g.get("mem_total")]
    except Exception:  # noqa: BLE001
        return []


def library_scan_now(cfg, sup):
    """scan_library with live GPU / RAM / loaded-model data from the running controller."""
    gm, ram = fit_inputs()
    lm = None
    try:
        if sup.serving():
            lm = (sup.running_profile or {}).get("model")
    except Exception:  # noqa: BLE001
        pass
    return scan_library(cfg.data, gm, ram, lm)


def library_delete(cfg, sup, entry_id, confirm):
    """Delete one library entry's own .gguf files. Returns (http_status, body). The target is resolved from a fresh
    scan by id, never from a client path. Never rmtree: os.remove per file, then os.rmdir of an emptied subfolder."""
    if confirm != "DELETE":
        return 400, {"error": 'confirmation missing: send "confirm": "DELETE"'}
    if not isinstance(entry_id, str) or not entry_id:
        return 400, {"error": "id is required"}
    scan = library_scan_now(cfg, sup)
    models = scan["models"]
    ent = next((m for m in models if m["id"] == entry_id), None)
    if ent is None:
        return 404, {"error": "model not found (rescan the library)"}
    if ent["partial"]:
        return 400, {"error": "partial downloads are not deleted here; remove the file by hand"}
    if ent["loaded"]:
        return 409, {"error": "model is loaded by the running server; stop it first", "used_by": ent["used_by"]}
    if ent["used_by"]:
        return 409, {"error": "model is used by profile(s): " + ", ".join(ent["used_by"]), "used_by": ent["used_by"]}
    files = [s["path"] for s in ent["shards"]]
    mm = ent.get("mmproj")
    if mm:
        others = [m for m in models if m["id"] != ent["id"] and m["dir"] == ent["dir"]]
        in_args = any(mm in str(p.get("extra_args") or "") for p in cfg.get("profiles") or [])
        if not others and not in_args:
            files.append(mm)
    roots = [rp for _, rp in library_roots(cfg)]
    freed, removed, errors = 0, [], []
    for f in files:
        try:
            rp = os.path.realpath(f)
            if not any(_inside(rp, r) for r in roots):
                errors.append(f"{f}: outside the model directories")
                continue
            n = os.lstat(f).st_size
            os.remove(f)
            freed += n
            removed.append(f)
        except OSError as e:
            errors.append(f"{f}: {e.strerror or e}")
    dir_removed = False
    try:
        d = ent["dir"]
        rd = os.path.realpath(d)
        if removed and not os.listdir(d) and rd not in roots and any(_inside(rd, r) for r in roots):
            os.rmdir(d)
            dir_removed = True
    except OSError:
        pass
    log(f"library: deleted {ent['name']} ({len(removed)} file(s), {freed} bytes) from {ent['dir']}"
        + (", folder removed" if dir_removed else "") + (f"; {len(errors)} error(s): {errors[0]}" if errors else ""))
    body = {"ok": not errors, "id": ent["id"], "name": ent["name"], "removed": removed, "freed_bytes": freed,
            "freed_gib": round(freed / 2**30, 2), "dir_removed": dir_removed, "errors": errors}
    if errors:
        body["error"] = f"{len(errors)} file(s) could not be removed: {errors[0]}"
        return 500, body
    return 200, body


def choose_gpus(size_bytes, gpu_mems):
    """'auto:N' for a model of this size: every GPU when it needs more than half the total VRAM, else 1 (it fits
    one card with ~10% overhead) or 2. Without GPU data or a size, the profile default."""
    gm = [m for m in (gpu_mems or []) if m]
    if not gm or not size_bytes:
        return PROFILE_DEFAULTS["gpus"]
    size = size_bytes / 1048576
    if size > sum(gm) / 2:
        return f"auto:{len(gm)}"
    if size <= max(gm) * (1 - FIT_OVERHEAD):
        return "auto:1"
    return f"auto:{min(2, len(gm))}"


def unique_profile_name(base, taken):
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", str(base or "")).strip("-.")[:48] or "model"
    name, i = base, 2
    while name in taken:
        name, i = f"{base}-{i}", i + 1
    return name


def library_draft(cfg, sup, d):
    """Draft profile (not saved) for a library entry. Returns (http_status, body)."""
    ent = next((m for m in library_scan_now(cfg, sup)["models"] if m["id"] == d.get("id")), None)
    if ent is None:
        return 404, {"error": "model not found (rescan the library)"}
    if not ent["complete"] or ent["partial"]:
        return 409, {"error": "model is incomplete (" + (ent["hint"] or "partial download") + ")"}
    gpus = d.get("gpus")
    gm = live_gpu_mems()
    if gpus in (None, ""):
        gpus = choose_gpus(ent["bytes"], gm)
    else:
        gpus = str(gpus).strip()
        if not (re.fullmatch(r"auto:\d+", gpus) or re.fullmatch(r"\d+(,\d+)*", gpus) or gpus == "none"):
            return 400, {"error": "gpus must be auto:N, a list like 0,1, or none"}
    ctx = d.get("ctx")
    if ctx in (None, ""):
        ctx = PROFILE_DEFAULTS["ctx"]
    else:
        try:
            ctx = int(ctx)
            if not 256 <= ctx <= 16777216:
                raise ValueError
        except (TypeError, ValueError):
            return 400, {"error": "ctx must be an integer between 256 and 16777216"}
    want = d.get("name")
    if want not in (None, "") and not re.fullmatch(r"[A-Za-z0-9._-]+", str(want)):
        return 400, {"error": "name must be [A-Za-z0-9._-]"}
    taken = {p["name"] for p in cfg.get("profiles") or []}
    name = unique_profile_name(want or ent["name"], taken)
    prof = dict(PROFILE_DEFAULTS, name=name, model=ent["first_shard"], gpus=gpus, ctx=ctx, fit=True)
    notes = []
    if want and name != want:
        notes.append(f"name {want!r} is taken, using {name!r}")
    if ent["mmproj"]:
        notes.append(f"vision projector found: add --mmproj {ent['mmproj']} to extra args if you want image input")
    return 200, {"draft": True, "profile": prof, "fit": ent["fit"], "notes": notes}


# ---------------------------------------------------------------- Hugging Face: browse (GET /api/hf/search, /api/hf/repo)

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
HF_FILE_RE = re.compile(r"^[\w.\-]+(/[\w.\-]+)*$")
HF_TIMEOUT = 15                  # seconds: connect and every socket read
HF_CALL_BUDGET = 30              # seconds for one JSON/text fetch including a slow trickle
HF_SORTS = ("downloads", "likes", "lastModified")
HF_CACHE_TTL = 60
HF_CARD_CHARS = 20000
_HF_CACHE = {}
_HF_CACHE_LOCK = threading.Lock()


class HFError(Exception):
    """A readable failure from the Hugging Face side. code = the HTTP status our own API answers with."""
    def __init__(self, code, msg, upstream=None):
        super().__init__(msg)
        self.code, self.msg, self.upstream = code, msg, upstream


def hf_valid_repo(repo):
    r = str(repo or "")
    if len(r) > 200 or not HF_REPO_RE.fullmatch(r):
        return False
    return not any(set(part) <= {"."} for part in r.split("/"))


def hf_valid_file(path):
    p = str(path or "")
    if len(p) > 400 or not HF_FILE_RE.fullmatch(p) or ".." in p or "." in p.split("/"):
        return False
    return p.lower().endswith(".gguf")


def _origin(url):
    u = urllib.parse.urlsplit(url)
    return (u.scheme, (u.hostname or "").lower(), u.port or (443 if u.scheme == "https" else 80))


def hf_request(url, token="", origin_url=None, headers=None, timeout=HF_TIMEOUT, max_redirects=5, on_conn=None):
    """GET url and follow redirects by hand. The bearer token goes only to the origin of origin_url (scheme, host,
    port): any hop to another host (the CDN) is requested without it. Returns (conn, resp, final_url); the caller
    closes conn. Raises OSError / http.client.HTTPException on network trouble, HFError on a bad redirect."""
    for _ in range(max_redirects + 1):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise HFError(502, "Hugging Face sent a redirect to a non-http URL")
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = cls(u.hostname, u.port, timeout=timeout)
        if on_conn:
            on_conn(conn)
        hdrs = {"User-Agent": "bizen-ctl", "Accept-Encoding": "identity"}
        hdrs.update(headers or {})
        if token and origin_url and _origin(url) == _origin(origin_url):
            hdrs["Authorization"] = "Bearer " + token
        try:
            conn.request("GET", (u.path or "/") + (("?" + u.query) if u.query else ""), headers=hdrs)
            resp = conn.getresponse()
        except BaseException:
            conn.close()
            raise
        if resp.status in (301, 302, 303, 307, 308) and resp.getheader("Location"):
            nxt = urllib.parse.urljoin(url, resp.getheader("Location"))
            resp.close()
            conn.close()
            url = nxt
            continue
        return conn, resp, url
    raise HFError(502, "too many redirects from Hugging Face")


def hf_status_error(status, token_sent):
    if status in (401, 403):
        return HFError(403, "repo not found or gated; token rejected (check the token in System)" if token_sent
                       else "repo not found or gated (set a token in System)", status)
    if status == 404:
        return HFError(404, "repo not found, or gated and the saved token has no access" if token_sent
                       else "repo not found or gated (set a token in System)", status)
    if status == 429:
        return HFError(429, "Hugging Face is rate limiting requests; try again in a minute", status)
    return HFError(502, f"Hugging Face answered HTTP {status}", status)


def hf_net_error(e, host):
    if isinstance(e, HFError):
        return e
    if isinstance(e, (TimeoutError, socket.timeout)):
        return HFError(504, f"timed out talking to {host} ({HF_TIMEOUT} s)")
    if isinstance(e, (OSError, http.client.HTTPException)):
        return HFError(502, f"could not reach {host}: {getattr(e, 'reason', None) or e.__class__.__name__ + ': ' + str(e)}"[:300])
    return HFError(502, "unexpected error talking to Hugging Face")


def hf_get(url, token, endpoint, max_bytes, ok_missing=False):
    """Fetch url -> (headers dict with lower-case keys, body bytes, truncated). Non-2xx raises HFError (404 -> None
    when ok_missing). Never raises anything else; bounded in bytes and in time."""
    host = urllib.parse.urlsplit(endpoint).netloc
    t_end = time.monotonic() + HF_CALL_BUDGET
    conn = None
    try:
        conn, resp, _ = hf_request(url, token, endpoint)
        if resp.status == 404 and ok_missing:
            return {}, None, False
        if not 200 <= resp.status < 300:
            raise hf_status_error(resp.status, bool(token))
        buf, trunc = bytearray(), False
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            buf += chunk
            if len(buf) > max_bytes:
                del buf[max_bytes:]
                trunc = True
                break
            if time.monotonic() > t_end:
                raise HFError(504, f"{host} answered too slowly")
        return {k.lower(): v for k, v in resp.getheaders()}, bytes(buf), trunc
    except HFError:
        raise
    except Exception as e:  # noqa: BLE001
        raise hf_net_error(e, host) from None
    finally:
        if conn:
            conn.close()


def hf_json(url, token, endpoint, max_bytes=8 << 20):
    headers, body, trunc = hf_get(url, token, endpoint, max_bytes)
    if trunc:
        raise HFError(502, "Hugging Face response is too large")
    try:
        return headers, json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        raise HFError(502, "Hugging Face sent a response that is not JSON") from None


def run_with_deadline(fn, deadline):
    """Run fn on a helper thread so the request thread never sits in a network call forever."""
    box = {}

    def target():
        try:
            box["v"] = fn()
        except HFError as e:
            box["e"] = e
        except Exception as e:  # noqa: BLE001
            box["e"] = HFError(502, f"unexpected error ({e.__class__.__name__})")

    t = threading.Thread(target=target, daemon=True, name="hf-fetch")
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise HFError(504, "Hugging Face took too long to answer")
    if "e" in box:
        raise box["e"]
    return box["v"]


def _tok_id(token):
    """In-memory cache key part: a different token never reuses another token's cached answer (the token itself is not kept)."""
    return hashlib.sha256(token.encode()).hexdigest()[:10] if token else ""


def _hf_cache_get(key):
    with _HF_CACHE_LOCK:
        hit = _HF_CACHE.get(key)
        if hit and time.time() - hit[0] < HF_CACHE_TTL:
            return hit[1]
        _HF_CACHE.pop(key, None)
    return None


def _hf_cache_put(key, val):
    with _HF_CACHE_LOCK:
        if len(_HF_CACHE) > 64:
            for k in sorted(_HF_CACHE, key=lambda k: _HF_CACHE[k][0])[:16]:
                _HF_CACHE.pop(k, None)
        _HF_CACHE[key] = (time.time(), val)


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def hf_map_model(m):
    """One search hit -> the shape the UI uses; None for rows without a valid repo id."""
    if not isinstance(m, dict):
        return None
    mid = m.get("id") or m.get("modelId") or ""
    if not hf_valid_repo(mid):
        return None
    tags = [t for t in (m.get("tags") or []) if isinstance(t, str)][:40]
    gated = m.get("gated")
    return {"id": mid, "author": str(m.get("author") or mid.split("/")[0]), "downloads": _int(m.get("downloads")),
            "likes": _int(m.get("likes")), "last_modified": m.get("lastModified") or m.get("last_modified"),
            "tags": tags, "private": bool(m.get("private")), "gated": gated if isinstance(gated, str) and gated else bool(gated)}


def hf_search(cfg, q, sort, limit):
    """cfg: plain config dict. Returns {"results": [...], "query", "sort", "cached"}. Raises HFError."""
    endpoint = str(cfg.get("hf_endpoint") or DEFAULTS["hf_endpoint"]).rstrip("/")
    token = str(cfg.get("hf_token") or "")
    key = ("search", endpoint, _tok_id(token), q, sort, limit)
    hit = _hf_cache_get(key)
    if hit is not None:
        return dict(hit, cached=True)

    def fetch():
        base = [("search", q), ("filter", "gguf"), ("sort", sort), ("direction", "-1"), ("limit", str(limit))] if q else \
               [("filter", "gguf"), ("sort", sort), ("direction", "-1"), ("limit", str(limit))]
        expand = [("expand[]", f) for f in ("author", "downloads", "likes", "lastModified", "tags", "private", "gated")]
        try:
            _, data = hf_json(endpoint + "/api/models?" + urllib.parse.urlencode(base + expand), token, endpoint)
        except HFError as e:
            if e.upstream != 400:
                raise
            _, data = hf_json(endpoint + "/api/models?" + urllib.parse.urlencode(base), token, endpoint)
        if not isinstance(data, list):
            raise HFError(502, "unexpected search response from Hugging Face")
        return [r for r in (hf_map_model(m) for m in data[:limit]) if r]

    res = {"results": run_with_deadline(fetch, 2 * HF_CALL_BUDGET), "query": q, "sort": sort, "cached": False}
    _hf_cache_put(key, res)
    return res


def hf_group_sets(files):
    """files: [{path, bytes}] (.gguf only) -> quant sets, multi-shard sets collapsed, mmproj attached per folder.
    Each set: {name, quant, dir, files:[{path,bytes}], bytes, gib, shards, shards_expected, complete, mmproj}."""
    mm, groups = {}, {}
    for f in files:
        d, _, base = f["path"].rpartition("/")
        if "mmproj" in base.lower():
            mm.setdefault(d, []).append(f)
            continue
        sm = SHARD_RE.match(base)
        if sm:
            g = groups.setdefault((d, sm.group(1), sm.group(3)), {"dir": d, "name": sm.group(1), "total": int(sm.group(3)), "sh": {}})
            g["sh"][int(sm.group(2))] = f
        else:
            groups[(d, base, "1")] = {"dir": d, "name": base[:-5], "total": 1, "sh": {1: f}}

    def pick_mm(d):
        c = mm.get(d) or mm.get("") or []
        c = sorted(c, key=lambda x: ("f16" not in x["path"].lower(), x["path"]))
        return {"path": c[0]["path"], "bytes": c[0]["bytes"]} if c else None

    sets = []
    for g in groups.values():
        fl = [{"path": g["sh"][i]["path"], "bytes": g["sh"][i]["bytes"]} for i in sorted(g["sh"])]
        size = sum(x["bytes"] for x in fl)
        sets.append({"name": g["name"], "quant": parse_quant(g["name"]) or parse_quant(g["dir"].rpartition("/")[2]),
                     "dir": g["dir"], "files": fl, "bytes": size, "gib": round(size / 2**30, 2), "shards": len(fl),
                     "shards_expected": g["total"], "complete": len(fl) == g["total"], "mmproj": pick_mm(g["dir"])})
    sets.sort(key=lambda s: (s["bytes"], s["name"]))
    return sets


_FM_RE = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


def split_front_matter(text):
    m = _FM_RE.match(text)
    return (m.group(1), text[m.end():]) if m else ("", text)


def _fm_value(front, key):
    m = re.search(r"^" + key + r":[ \t]*(.*)$", front, re.M)
    if not m:
        return None
    v = m.group(1).strip().strip("'\"")
    if v:
        return v
    items = re.findall(r"^[ \t]*-[ \t]*(\S.*)$", front[m.end():], re.M)
    return items[0].strip().strip("'\"") if items else None


def hf_repo_hints(meta, front):
    """license / base_model / architecture from the repo metadata, falling back to the card front matter."""
    card = meta.get("cardData") if isinstance(meta.get("cardData"), dict) else {}
    tags = [t for t in (meta.get("tags") or []) if isinstance(t, str)]

    def first(v):
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v)[:120] if v not in (None, "") else None

    lic = first(card.get("license")) or next((t[8:] for t in tags if t.startswith("license:")), None) or _fm_value(front, "license")
    bm = card.get("base_model")
    if isinstance(bm, str):
        bm = [bm]
    if not isinstance(bm, list):
        bm = [t.rsplit(":", 1)[1] for t in tags if t.startswith("base_model:")]
    bm = [str(x)[:120] for x in bm if isinstance(x, str)][:4] or ([_fm_value(front, "base_model")] if _fm_value(front, "base_model") else [])
    gg = meta.get("gguf") if isinstance(meta.get("gguf"), dict) else {}
    return {"license": lic, "base_model": bm, "architecture": first(gg.get("architecture"))}


def hf_next_link(headers, endpoint):
    """URL from a `Link: <...>; rel="next"` header, only when it stays on the endpoint's own origin."""
    m = re.search(r'<([^>]+)>\s*;\s*rel="?next"?', headers.get("link", ""))
    if not m:
        return None
    url = urllib.parse.urljoin(endpoint + "/", m.group(1))
    return url if _origin(url) == _origin(endpoint) else None


def hf_repo_raw(cfg, repo):
    """Network part of GET /api/hf/repo: {meta, files:[{path,bytes}], skipped, tree_truncated, card, card_truncated}.
    Cached 60 s. Raises HFError."""
    endpoint = str(cfg.get("hf_endpoint") or DEFAULTS["hf_endpoint"]).rstrip("/")
    token = str(cfg.get("hf_token") or "")
    key = ("repo", endpoint, _tok_id(token), repo)
    hit = _hf_cache_get(key)
    if hit is not None:
        return hit
    rq = urllib.parse.quote(repo, safe="/")

    def fetch():
        _, meta = hf_json(f"{endpoint}/api/models/{rq}", token, endpoint, 2 << 20)
        if not isinstance(meta, dict):
            raise HFError(502, "unexpected repo response from Hugging Face")
        files, skipped, truncated = [], 0, False
        url, pages = f"{endpoint}/api/models/{rq}/tree/main?recursive=1", 0
        while url and pages < 30:
            hdrs, data = hf_json(url, token, endpoint, 16 << 20)
            pages += 1
            if not isinstance(data, list):
                raise HFError(502, "unexpected file list from Hugging Face")
            for e in data:
                if not isinstance(e, dict) or e.get("type", "file") != "file":
                    continue
                p = str(e.get("path") or "")
                if not p.lower().endswith(".gguf"):
                    continue
                lfs = e.get("lfs") if isinstance(e.get("lfs"), dict) else {}
                size = _int(lfs.get("size")) or _int(e.get("size"))
                if not hf_valid_file(p) or size <= 0:
                    skipped += 1
                    continue
                files.append({"path": p, "bytes": size})
            url = hf_next_link(hdrs, endpoint)
            if len(files) > 5000:
                truncated = True
                break
        if url and pages >= 30:
            truncated = True
        card, card_trunc, front = "", False, ""
        _, body, _t = hf_get(f"{endpoint}/{rq}/raw/main/README.md", token, endpoint, 200000, ok_missing=True)
        if body is not None:
            front, text = split_front_matter(body.decode("utf-8", "replace"))
            card_trunc = len(text) > HF_CARD_CHARS
            card = text[:HF_CARD_CHARS]
        return {"meta": meta, "files": files, "skipped": skipped, "tree_truncated": truncated,
                "card": card, "card_truncated": card_trunc, "front": front}

    raw = run_with_deadline(fetch, 4 * HF_CALL_BUDGET)
    _hf_cache_put(key, raw)
    return raw


def fit_inputs():
    """(per-GPU total MiB, system RAM MiB or None) as the Library uses them for its size-based verdicts."""
    gm = live_gpu_mems()
    try:
        ram = meminfo().get("total_mib")
    except Exception:  # noqa: BLE001
        ram = None
    return gm, ram


def hf_repo_view(cfg, repo):
    """Everything GET /api/hf/repo returns. cfg: plain config dict. Raises HFError."""
    raw = hf_repo_raw(cfg, repo)
    meta = raw["meta"]
    gm, ram = fit_inputs()
    vram = sum(m for m in gm if m) or None
    try:
        have = {m["name"] for m in scan_library(cfg, gm, ram)["models"]
                if m["complete"] and not m["partial"]}
    except Exception:  # noqa: BLE001
        have = set()
    sets = hf_group_sets(raw["files"])
    for s in sets:
        s["fit"] = fit_verdict(s["bytes"], vram, ram) if s["complete"] else {
            "vram_total_mib": vram, "ram_total_mib": ram, "verdict": None, "text": "Incomplete set in the repo, so no estimate. " + FIT_NOTE}
        s["already_downloaded"] = s["name"] in have
    hint = hf_repo_hints(meta, raw["front"])
    m = hf_map_model(dict(meta, id=meta.get("id") or meta.get("modelId") or repo)) or {}
    out = {"id": repo, "author": m.get("author") or repo.split("/")[0], "downloads": m.get("downloads", 0),
           "likes": m.get("likes", 0), "last_modified": m.get("last_modified"), "tags": m.get("tags", []),
           "private": m.get("private", False), "gated": m.get("gated", False)}
    out.update(hint)
    out.update(sets=sets, card=raw["card"], card_truncated=raw["card_truncated"], skipped=raw["skipped"],
               tree_truncated=raw["tree_truncated"], has_token=bool(cfg.get("hf_token")))
    return out


# ---------------------------------------------------------------- Hugging Face: downloader (/api/downloads)

DL_CHUNK = 1 << 20               # one read1 of up to 1 MiB per loop turn; blocking, never a busy loop
DL_MAX_QUEUED = 3
DL_WINDOW_S = 10                 # speed / ETA are averaged over this window
DL_KEEP = 50                     # finished entries remembered
DL_ID_RE = re.compile(r"^[0-9a-f]{8}$")


class DlError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


class DlFail(Exception):
    """A download that stops with a readable error; the .part files stay."""


def _free_bytes(path):
    try:
        p = os.path.expanduser(str(path))
        while p and not os.path.exists(p) and os.path.dirname(p) != p:
            p = os.path.dirname(p)
        st = os.statvfs(p)
        return st.f_bavail * st.f_frsize
    except (OSError, ValueError):
        return None


def _part_size(final):
    try:
        return os.path.getsize(final + ".part") if os.path.isfile(final + ".part") and not os.path.islink(final + ".part") else 0
    except OSError:
        return 0


def dl_plan(data, repo, paths, sizes):
    """Validate a download request against the repo's file sizes (sizes: {path: bytes}) and work out where it lands:
    <download_dir>/<repo name>/<quant folder or 'files'>/<basename>. Returns the plan dict; raises DlError."""
    if not hf_valid_repo(repo):
        raise DlError(400, "repo must look like owner/name")
    if not isinstance(paths, list) or not 1 <= len(paths) <= 64 or not all(isinstance(p, str) for p in paths):
        raise DlError(400, "files must be a list of 1 to 64 repo paths")
    paths = list(dict.fromkeys(paths))
    for p in paths:
        if not hf_valid_file(p):
            raise DlError(400, f"bad file name {p[:80]!r}: only plain relative .gguf paths are accepted")
    mm = [p for p in paths if "mmproj" in p.rpartition("/")[2].lower()]
    models = [p for p in paths if p not in mm]
    if len(mm) > 1 or not models:
        raise DlError(400, "ask for one quant set (all its shards) plus at most one mmproj file")
    dirs = {p.rpartition("/")[0] for p in models}
    sets = hf_group_sets([{"path": p, "bytes": 1} for p in models])
    if len(dirs) != 1 or len(sets) != 1 or not sets[0]["complete"]:
        raise DlError(400, "files must be exactly one quant set: every shard, all in one folder")
    if len({p.rpartition("/")[2] for p in paths}) != len(paths):
        raise DlError(400, "two files have the same name")
    for p in paths:
        if not sizes.get(p):
            raise DlError(404, f"{p} is not a .gguf file in {repo}")
    root = effective_download_dir(data)
    if not root:
        raise DlError(400, "no download directory is configured (set one in System)")
    base = os.path.realpath(os.path.expanduser(root))
    folder = next(iter(dirs)).rpartition("/")[2] or "files"
    dest = os.path.join(base, repo.split("/", 1)[1], folder)
    if not _inside(os.path.realpath(dest), base):
        raise DlError(400, "destination would leave the download directory")
    return {"repo": repo, "name": sets[0]["name"], "base": base, "dest_dir": dest,
            "files": [{"path": p, "bytes": sizes[p], "done": 0} for p in models + mm]}


class Download:
    def __init__(self, d):
        self.id, self.repo, self.name = d["id"], d["repo"], d["name"]
        self.state = d.get("state", "queued")
        self.files = [{"path": f["path"], "bytes": int(f["bytes"]), "done": int(f.get("done") or 0)} for f in d["files"]]
        self.dest_dir = d["dest_dir"]
        self.started, self.finished = d.get("started"), d.get("finished")
        self.error, self.note = d.get("error"), d.get("note")
        self.validators = dict(d.get("validators") or {})
        self.stop = None            # None | "pause" | "cancel"
        self.conn = None
        self.samples = collections.deque(maxlen=64)
        self._t_sample = 0.0

    @property
    def bytes_total(self):
        return sum(f["bytes"] for f in self.files)

    @property
    def bytes_done(self):
        return sum(min(f["done"], f["bytes"]) for f in self.files)

    def rate(self, now):
        """Bytes per second over the last DL_WINDOW_S seconds (0 when not running or stalled)."""
        if self.state != "running":
            return 0.0
        pts = [s for s in list(self.samples) if now - s[0] <= DL_WINDOW_S]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.4:
            return 0.0
        return max(0.0, (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]))

    def public(self, now=None):
        now = now if now is not None else time.monotonic()
        done, total, r = self.bytes_done, self.bytes_total, self.rate(now)
        return {"id": self.id, "repo": self.repo, "name": self.name, "state": self.state,
                "files": [dict(f) for f in self.files], "bytes_total": total, "bytes_done": done,
                "pct": round(100.0 * done / total, 1) if total else 0.0,
                "mbps": round(r / 1e6, 2), "eta_s": int((total - done) / r) if r > 0 and self.state == "running" else None,
                "started": self.started, "finished": self.finished, "error": self.error, "note": self.note,
                "dest_dir": self.dest_dir}

    def to_json(self):
        d = self.public()
        for k in ("mbps", "eta_s", "pct", "bytes_total", "bytes_done"):
            d.pop(k, None)
        d["validators"] = self.validators
        return d


class Downloader:
    """One worker thread, one active download, FIFO queue of at most DL_MAX_QUEUED. State survives a restart in
    <config dir>/downloads.json (never the token); interrupted entries come back paused, nothing auto-starts."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.items, self.queue, self.cur = [], [], None
        self._quit = False
        self._t_save = 0.0
        self.path = os.path.join(os.path.dirname(os.path.abspath(cfg.path)), "downloads.json")
        self._load()
        self.thread = threading.Thread(target=self._worker, daemon=True, name="downloader")
        self.thread.start()

    # -- persistence
    def _load(self):
        try:
            with open(self.path) as f:
                raw = json.load(f)
            rows = raw.get("items") if isinstance(raw, dict) else None
        except (OSError, ValueError):
            rows = None
        for d in rows or []:
            try:
                if not (isinstance(d, dict) and DL_ID_RE.match(d["id"]) and hf_valid_repo(d["repo"])
                        and isinstance(d["dest_dir"], str) and isinstance(d["name"], str)
                        and all(hf_valid_file(f["path"]) for f in d["files"]) and d["files"]):
                    continue
                it = Download(d)
                if it.state in ("running", "queued"):
                    it.state, it.note = "paused", "interrupted by a controller restart; resume to continue"
                elif it.state not in ("paused", "done", "error", "canceled"):
                    continue
                if it.state != "done":
                    for fl in it.files:
                        fl["done"] = min(fl["bytes"], _part_size(os.path.join(it.dest_dir, os.path.basename(fl["path"]))))
                self.items.append(it)
            except (KeyError, TypeError, ValueError):
                continue

    def _save(self, force=True):
        """Caller holds the lock."""
        now = time.monotonic()
        if not force and now - self._t_save < 2.0:
            return
        self._t_save = now
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"version": 1, "items": [it.to_json() for it in self.items]}, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError as e:
            log(f"downloads: could not save {self.path}: {e}")

    # -- public api (each returns the entry's public dict or raises DlError)
    def list(self):
        with self.lock:
            now = time.monotonic()
            return [it.public(now) for it in self.items]

    def active(self):
        with self.lock:
            if self.cur is None or self.cur.state != "running":
                return None
            p = self.cur.public()
            return {"id": p["id"], "name": p["name"], "pct": p["pct"], "mbps": p["mbps"], "eta_s": p["eta_s"]}

    def _get(self, did):
        it = next((x for x in self.items if x.id == did), None)
        if it is None:
            raise DlError(404, "no such download")
        return it

    def add(self, plan):
        data = self.cfg.data
        base = plan["base"]
        du = disk_usage(base)
        if du["writable"] is False:
            raise DlError(400, "download directory is not writable")
        remaining = 0
        all_done = True
        for f in plan["files"]:
            final = os.path.join(plan["dest_dir"], os.path.basename(f["path"]))
            if os.path.isfile(final) and os.path.getsize(final) == f["bytes"]:
                f["done"] = f["bytes"]
                continue
            all_done = False
            f["done"] = min(f["bytes"], _part_size(final))
            remaining += f["bytes"] - f["done"]
        if all_done:
            raise DlError(409, "already downloaded")
        with self.lock:
            if any(x.repo == plan["repo"] and x.state in ("queued", "running", "paused")
                   and {f["path"] for f in x.files} == {f["path"] for f in plan["files"]} for x in self.items):
                raise DlError(409, "this download is already in the list")
            if len(self.queue) >= DL_MAX_QUEUED:
                raise DlError(409, f"the queue is full ({DL_MAX_QUEUED} waiting); wait for one to finish or cancel one")
            others = sum(x.bytes_total - x.bytes_done for x in self.items if x.state in ("queued", "running"))
            need = remaining + others + float(data.get("min_free_gib_for_download") or 0) * 2**30
            free = _free_bytes(base)
            if free is not None and free < need:
                raise DlError(409, f"not enough free space: need {need / 2**30:.1f} GiB, have {free / 2**30:.1f} GiB")
            it = Download({"id": secrets.token_hex(4), "repo": plan["repo"], "name": plan["name"], "state": "queued",
                           "files": plan["files"], "dest_dir": plan["dest_dir"]})
            self.items.append(it)
            self.queue.append(it)
            self._trim()
            self._save()
            self.cv.notify_all()
            return it.public()

    def _trim(self):
        fin = [x for x in self.items if x.state in ("done", "error", "canceled")]
        for x in fin[:max(0, len(fin) - DL_KEEP)]:
            self.items.remove(x)

    def _wait_stopped(self, it, secs=5.0):
        self.cv.wait_for(lambda: it.state != "running", secs)

    @staticmethod
    def _abort(it):
        try:
            c = it.conn
            if c is not None and c.sock is not None:
                c.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def pause(self, did):
        with self.lock:
            it = self._get(did)
            if it.state == "queued":
                self.queue.remove(it)
                it.state = "paused"
            elif it.state == "running":
                it.stop = "pause"
                self._abort(it)
                self._wait_stopped(it)
            else:
                raise DlError(409, f"cannot pause a {it.state} download")
            self._save()
            return it.public()

    def cancel(self, did):
        with self.lock:
            it = self._get(did)
            if it.state == "running":
                it.stop = "cancel"
                self._abort(it)
                self._wait_stopped(it)
            elif it.state in ("queued", "paused", "error"):
                if it in self.queue:
                    self.queue.remove(it)
                it.state, it.error, it.note = "canceled", None, None
            else:
                raise DlError(409, f"cannot cancel a {it.state} download")
            self._save()
            return it.public()

    def resume(self, did):
        with self.lock:
            it = self._get(did)
            if it.state not in ("paused", "error", "canceled"):
                raise DlError(409, f"cannot resume a {it.state} download")
            if len(self.queue) >= DL_MAX_QUEUED:
                raise DlError(409, f"the queue is full ({DL_MAX_QUEUED} waiting)")
            it.state, it.stop, it.error, it.note = "queued", None, None, None
            it.finished = None
            for fl in it.files:
                fl["done"] = min(fl["bytes"], _part_size(os.path.join(it.dest_dir, os.path.basename(fl["path"]))))
            self.queue.append(it)
            self._save()
            self.cv.notify_all()
            return it.public()

    def remove(self, did, delete_partial=False):
        with self.lock:
            it = self._get(did)
            if it.state not in ("done", "error", "canceled"):
                raise DlError(409, f"cancel the {it.state} download first")
            removed = 0
            if delete_partial:
                removed = self._delete_parts(it)
            self.items.remove(it)
            self._save()
            return {"ok": True, "removed_partial": removed}

    def _delete_parts(self, it):
        """Remove this download's .part files. Confined to the download directory, regular files only."""
        base = os.path.realpath(os.path.expanduser(effective_download_dir(self.cfg.data) or "/nonexistent"))
        d = os.path.realpath(it.dest_dir)
        if d == base or not _inside(d, base):
            return 0
        n = 0
        for f in it.files:
            part = os.path.join(d, os.path.basename(f["path"]) + ".part")
            try:
                if os.path.isfile(part) and not os.path.islink(part) and _inside(os.path.realpath(part), base):
                    os.remove(part)
                    n += 1
            except OSError:
                pass
        for dirp in (d, os.path.dirname(d)):
            try:
                if dirp != base and _inside(dirp, base):
                    os.rmdir(dirp)          # only succeeds when empty
            except OSError:
                break
        return n

    def shutdown(self):
        with self.lock:
            self._quit = True
            it = self.cur
            if it is not None and it.state == "running":
                it.stop = "pause"
                self._abort(it)
                self._wait_stopped(it, 3.0)
            for x in self.items:
                if x.state in ("running", "queued"):
                    x.state, x.note = "paused", "interrupted by a controller restart; resume to continue"
            self._save()
            self.cv.notify_all()

    # -- worker
    def _worker(self):
        while True:
            with self.lock:
                while not self._quit and not self.queue:
                    self.cv.wait(30)
                if self._quit:
                    return
                it = self.queue.pop(0)
                it.state, it.stop, it.error, it.note = "running", None, None, None
                if it.started is None:
                    it.started = time.time()
                it.samples.clear()
                it.samples.append((time.monotonic(), it.bytes_done))
                self.cur = it
                self._save()
            try:
                self._run(it)
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    it.state, it.error = "error", f"unexpected error ({e.__class__.__name__}: {e})"[:300]
            with self.lock:
                it.conn = None
                self.cur = None
                if it.state in ("done", "error", "canceled"):
                    it.finished = time.time()
                self._save()
                self.cv.notify_all()

    def _finish(self, it, state, error=None):
        with self.lock:
            it.state, it.error = state, error
            self.cv.notify_all()

    def _run(self, it):
        data = self.cfg.data
        endpoint = str(data.get("hf_endpoint") or DEFAULTS["hf_endpoint"]).rstrip("/")
        try:
            base = os.path.realpath(os.path.expanduser(effective_download_dir(data) or "/nonexistent"))
            if not _inside(os.path.realpath(it.dest_dir), base):
                raise DlFail("destination is outside the download directory (was download_dir changed?)")
            os.makedirs(it.dest_dir, exist_ok=True)
            if not _inside(os.path.realpath(it.dest_dir), base):
                raise DlFail("destination is outside the download directory")
            for f in it.files:
                if it.stop:
                    break
                self._fetch_file(it, f, base, endpoint)
        except DlFail as e:
            if it.stop is None:
                return self._finish(it, "error", str(e)[:400])
        except OSError as e:
            if it.stop is None:
                return self._finish(it, "error", f"disk error: {e.strerror or e}"[:300])
        self._finish(it, {"pause": "paused", "cancel": "canceled"}.get(it.stop, "done" if not it.stop else "paused"))

    def _tick(self, it):
        now = time.monotonic()
        if now - it._t_sample >= 0.5:
            it._t_sample = now
            it.samples.append((now, it.bytes_done))
            with self.lock:
                self._save(force=False)

    def _fetch_file(self, it, f, base, endpoint):
        name = os.path.basename(f["path"])
        final = os.path.join(it.dest_dir, name)
        part = final + ".part"
        if os.path.islink(final) or os.path.islink(part):
            raise DlFail(f"refusing to write through a symlink: {name}")
        if os.path.isfile(final) and os.path.getsize(final) == f["bytes"]:
            f["done"] = f["bytes"]
            return
        existing = _part_size(final)
        if existing > f["bytes"]:
            existing = 0
        url = f"{endpoint}/{urllib.parse.quote(it.repo, safe='/')}/resolve/main/{urllib.parse.quote(f['path'], safe='/')}"
        host = urllib.parse.urlsplit(endpoint).netloc

        def set_conn(c):
            it.conn = c
            if it.stop:
                raise DlFail("stopped")

        for _ in range(3):
            token = str(self.cfg.data.get("hf_token") or "")
            hdr = {}
            if existing > 0:
                hdr["Range"] = f"bytes={existing}-"
                if it.validators.get(f["path"]):
                    hdr["If-Range"] = it.validators[f["path"]]
            conn = None
            try:
                conn, resp, _u = hf_request(url, token, endpoint, hdr, on_conn=set_conn)
                st = resp.status
                complete = False
                if st == 416 and existing:
                    if existing == f["bytes"]:
                        complete = True
                    else:
                        existing = 0
                        continue
                elif st not in (200, 206):
                    err = hf_status_error(st, bool(token))
                    if st in (401, 403):
                        raise DlFail("access denied: accept the model's terms on Hugging Face and set a token in System"
                                     + ("; token rejected" if token else ""))
                    raise DlFail("file not found on Hugging Face" if st == 404 else err.msg)
                if not complete:
                    if st == 206:
                        m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", resp.getheader("Content-Range") or "")
                        if not m or int(m.group(1)) != existing:
                            existing = 0
                            continue
                        if m.group(3) != "*" and int(m.group(3)) != f["bytes"]:
                            raise DlFail(f"{name}: size changed on the server (expected {f['bytes']}, now {m.group(3)}); remove the entry and its partial file")
                        mode = "ab"
                    else:
                        existing, mode = 0, "wb"      # 200: the server ignored Range or the file changed: start over
                    et = resp.getheader("ETag")
                    if et and not et.startswith("W/"):
                        it.validators[f["path"]] = et
                    f["done"] = existing
                    try:
                        fh = open(part, mode)
                    except OSError as e:
                        raise DlFail(f"disk error opening {name}.part: {e.strerror or e}") from None
                    with fh:
                        while True:
                            chunk = resp.read1(DL_CHUNK)
                            if not chunk:
                                break
                            if f["done"] + len(chunk) > f["bytes"]:
                                raise DlFail(f"{name}: server sent more than the expected {f['bytes']} bytes")
                            try:
                                fh.write(chunk)
                            except OSError as e:
                                raise DlFail(f"disk error writing {name}.part: {e.strerror or e}") from None
                            f["done"] += len(chunk)
                            self._tick(it)
                            if it.stop:
                                return
                break
            except DlFail:
                raise
            except (OSError, http.client.HTTPException, HFError) as e:
                if it.stop:
                    return
                if isinstance(e, http.client.IncompleteRead):
                    raise DlFail(f"{name}: connection closed before the end of the file; resume to continue") from None
                raise DlFail(hf_net_error(e, host).msg) from None
            finally:
                if conn is not None:
                    conn.close()
        else:
            raise DlFail(f"{name}: the server kept changing its answer; try again")
        size = os.path.getsize(part) if os.path.isfile(part) else 0
        if size != f["bytes"]:
            raise DlFail(f"{name}: size mismatch, got {size} bytes but the repo lists {f['bytes']}; partial file kept, resume to retry")
        os.replace(part, final)
        f["done"] = f["bytes"]


class SSECount:
    """Counts token-bearing `data:` events in a stream that arrives in arbitrary pieces. Observes only;
    what is relayed to the client is never touched. Never raises."""

    def __init__(self):
        self.buf = b""

    @staticmethod
    def _token_event(raw):
        if not raw or raw == b"[DONE]":
            return 0
        try:
            d = json.loads(raw)
        except ValueError:
            return 1
        if not isinstance(d, dict) or "error" in d or "bizen" in d:
            return 0
        ch = d.get("choices")
        if isinstance(ch, list):
            c = ch[0] if ch and isinstance(ch[0], dict) else {}
            delta = c.get("delta") if isinstance(c.get("delta"), dict) else {}
            return 1 if (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls") or c.get("text")) else 0
        return 1 if d.get("content") else 0

    def feed(self, data):
        n = 0
        try:
            self.buf = (self.buf + data).replace(b"\r\n", b"\n")
            while True:
                i = self.buf.find(b"\n\n")
                if i < 0:
                    break
                ev, self.buf = self.buf[:i], self.buf[i + 2:]
                for line in ev.split(b"\n"):
                    if line.startswith(b"data:"):
                        n += self._token_event(line[5:].strip())
            if len(self.buf) > (1 << 20):
                self.buf = b""
        except Exception:  # noqa: BLE001
            self.buf = b""
        return n


# ---------------------------------------------------------------- supervisor

class Supervisor:
    """Owns the llama-server child: start/stop, health, idle unload, logs."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.proc = None
        self.state = "stopped"        # stopped | starting | running | stopping | failed
        self.error = ""
        self.started_at = None
        self.ready_at = None
        self.running_profile = None   # snapshot of the profile used to launch
        self.running_cmd = []
        self.running_gpus = []
        self.running_cpus = []
        self.inflight = 0
        self.last_activity = time.time()
        self.logs = collections.deque(maxlen=3000)
        self.last_tps = None
        self.last_prompt_tps = None
        self._load_reset(None)
        self.gen_reqs = {}                    # id -> {started, first, tokens, last}; guarded by self.lock
        self.gen_next = 0
        self.gen_recent = collections.deque(maxlen=512)   # arrival times of recent token chunks
        self.gen_last_tps = None
        self.internal_key = secrets.token_urlsafe(24)
        self.ready_event = threading.Event()
        self.switch_lock = threading.Lock()   # serialises routed profile switches
        self.routed = False                   # the running profile was chosen by the router
        self.kev = None                       # set in main(): its GPUs are off limits to us
        threading.Thread(target=self._idle_loop, daemon=True).start()

    # -- load progress
    def _load_reset(self, model):
        self.load_total = model_total_bytes(model) if model else None
        self.load_phase = 0
        self.load_bytes = 0
        self.load_pct = 0
        self.load_hist = collections.deque(maxlen=60)   # (t, bytes) for the recent MB/s
        self.place_base = None        # gpu_info() snapshot taken just before the child launched
        self.place_frozen = None      # placement copy taken when the state became ready
        self.place_gpus_cache = (0.0, [])

    def _note_line(self, proc, line):
        try:
            ph = phase_of_line(line)
            if ph:
                with self.lock:
                    if self.proc is proc and self.state == "starting":
                        self.load_phase = max(self.load_phase, min(ph, 2))   # ready comes from the health check
        except Exception:  # noqa: BLE001
            pass

    def load_info(self, stats=None):
        """Load progress for the child being started: phase, bytes_read/total, pct (monotonic, 99 max until
        ready), mbps (recent) and eta_s. Unknown values are None. Never raises."""
        out = {"phase": None, "bytes_read": None, "bytes_total": None, "pct": None,
               "eta_s": None, "mbps": None, "source": None}
        try:
            with self.lock:
                st, total = self.state, self.load_total
                out["phase"] = st
                out["bytes_total"] = total
                if st == "running":
                    out.update(phase="ready", pct=100, eta_s=0, bytes_read=total or self.load_bytes or None)
                    return out
                if st != "starting":
                    return out
                pid = self.proc.pid if self.proc else None
                out["phase"] = LOAD_PHASES[self.load_phase]
                stats = stats or proc_stats(pid)
                rss = stats.get("rss_mib")
                cand = {"rchar": stats.get("rchar"), "read_bytes": stats.get("read_bytes"),
                        "rss": rss * 1048576 if rss is not None else None}
                cand = {k: v for k, v in cand.items() if v is not None}
                now = time.time()
                if cand:
                    src = max(cand, key=cand.get)
                    b = cand[src]
                    if total:
                        b = min(b, total)
                    if b >= self.load_bytes:
                        self.load_bytes, out["source"] = b, src
                    else:
                        out["source"] = "held"
                    if not self.load_hist or now - self.load_hist[-1][0] >= 0.5:
                        self.load_hist.append((now, self.load_bytes))
                known = bool(cand) or self.load_bytes > 0   # a transient /proc failure keeps the last numbers
                out["bytes_read"] = self.load_bytes if known else None
                if total and known:
                    p = int(100 * self.load_bytes / total)
                    if self.load_phase >= 2:
                        p = max(p, 95)       # "offloaded N/M layers" is printed once the tensors are in
                    self.load_pct = max(self.load_pct, min(p, 99))
                    out["pct"] = self.load_pct
                if len(self.load_hist) >= 2:
                    old = next((h for h in self.load_hist if now - h[0] <= 10), self.load_hist[0])
                    dt = self.load_hist[-1][0] - old[0]
                    if dt >= 1:
                        out["mbps"] = round(max(0, self.load_hist[-1][1] - old[1]) / dt / 1e6, 1)
                if out["mbps"] and total and self.load_phase < 2 and self.load_bytes < total:
                    out["eta_s"] = int((total - self.load_bytes) / (out["mbps"] * 1e6))
        except Exception:  # noqa: BLE001
            pass
        return out

    def placement_info(self, gpus=None, stats=None):
        """Placement from VRAM growth since the profile started (see compute_placement). While starting or running
        it is live; once ready the frozen copy is the headline and the live numbers sit under `live`. None when
        nothing was ever launched. Never raises."""
        try:
            with self.lock:
                st, base, frozen = self.state, self.place_base, self.place_frozen
                idx, pid = list(self.running_gpus), (self.proc.pid if self.proc else None)
            if base is None:
                return None
            live = None
            if st in ("starting", "running"):
                if gpus is None:
                    t, cached = self.place_gpus_cache
                    if time.time() - t < 2:
                        gpus = cached
                    else:
                        gpus = gpu_info()
                        self.place_gpus_cache = (time.time(), gpus)
                rss = (stats or proc_stats(pid)).get("rss_mib")
                live = compute_placement(base, gpus, idx, rss)
            if frozen:
                out = dict(frozen)
                out["live"] = live
                return out
            return live
        except Exception:  # noqa: BLE001
            return None

    def _freeze_placement(self):
        try:
            with self.lock:
                if self.place_base is None or self.place_frozen:
                    return
            p = self.placement_info(gpus=gpu_info())
            if p:
                p["frozen"] = True
                with self.lock:
                    self.place_frozen = p
        except Exception:  # noqa: BLE001
            pass

    def _placement_compact(self):
        p = self.placement_info()
        if not p:
            return None
        return {"source": p["source"], "total_mib": p["total_mib"], "host_mib": p["host_mib"],
                "gpu_share_pct": p["gpu_share_pct"], "frozen": p["frozen"],
                "gpu_mib": [d["mib"] for d in p["devices"]]}

    # -- generation stats (counted from the SSE chunks that pass through the proxy; one chunk ~ one token)
    def gen_begin(self):
        with self.lock:
            self.gen_next += 1
            self.gen_reqs[self.gen_next] = {"started": time.time(), "first": None, "tokens": 0, "last": None}
            return self.gen_next

    def gen_chunk(self, gid, n):
        if n <= 0:
            return
        with self.lock:
            r = self.gen_reqs.get(gid)
            if r:
                now = time.time()
                if r["first"] is None:
                    r["first"] = now
                r["tokens"] += n
                r["last"] = now
                self.gen_recent.extend([now] * min(n, 64))

    def gen_end(self, gid, tokens=None, tps=None):
        """Finish a request. tokens/tps (from a non-streamed reply's usage/timings) fill in when no chunks were seen."""
        with self.lock:
            r = self.gen_reqs.pop(gid, None)
            if not r:
                return
            if tokens and not r["tokens"]:
                r["tokens"] = int(tokens)
            if tps is None and r["tokens"] >= 2 and r["first"] and r["last"] and r["last"] > r["first"]:
                tps = (r["tokens"] - 1) / (r["last"] - r["first"])
            if tps:
                self.gen_last_tps = round(float(tps), 2)

    def gen_info(self):
        """{active, requests, tokens, ttft_s, tps, last_tps}. tokens/ttft_s belong to the newest running request
        (None when idle); tps is the live rate over the last ~3 s (None until the first token); last_tps is the
        average of the last finished request. Never raises."""
        out = {"active": False, "requests": 0, "tokens": None, "ttft_s": None, "tps": None, "last_tps": self.gen_last_tps}
        try:
            with self.lock:
                out["requests"] = len(self.gen_reqs)
                out["active"] = bool(self.gen_reqs)
                if self.gen_reqs:
                    r = self.gen_reqs[max(self.gen_reqs)]
                    out["tokens"] = r["tokens"]
                    if r["first"]:
                        out["ttft_s"] = round(r["first"] - r["started"], 2)
                    now = time.time()
                    ev = [t for t in self.gen_recent if now - t <= 3.0]
                    if r["first"]:
                        out["tps"] = round((len(ev) - 1) / (ev[-1] - ev[0]), 1) if len(ev) >= 2 and ev[-1] > ev[0] and now - ev[-1] <= 2.0 else 0.0
        except Exception:  # noqa: BLE001
            pass
        return out

    # -- command building
    def build(self, prof):
        exe = os.path.expanduser(prof.get("llama_server") or self.cfg.get("llama_server"))
        kev_gpus = self.kev.gpus if self.kev else []
        gpus = pick_gpus(prof["gpus"], own=self.running_gpus if self.proc else (), exclude=kev_gpus,
                         bucket=self.cfg.get("gpu_bucket_mib") or 512, fallback=self.cfg.get("gpu_fallback_count") or 8)
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # make CUDA indices match nvidia-smi
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus)) if gpus else ""

        prefix, cpus, node = [], [], None
        pin = str(prof.get("pin", "auto")).strip()
        if pin == "auto":
            cpus, node = auto_pin()
        elif pin not in ("off", ""):
            cpus = [int(x) for x in pin.split(",")]
            _, _, node_of = cpu_topology()
            nodes = {node_of.get(c) for c in cpus}
            node = nodes.pop() if len(nodes) == 1 else None
        if cpus:
            cl = ",".join(map(str, cpus))
            if shutil.which("numactl"):
                prefix = ["numactl", f"--physcpubind={cl}"] + ([f"--preferred={node}"] if node is not None else [])
            elif shutil.which("taskset"):
                prefix = ["taskset", "-c", cl]

        thr = prof.get("threads", "auto")
        threads = len(cpus) if str(thr) == "auto" and cpus else (int(thr) if str(thr).isdigit() else None)

        args = [exe, "-m", prof["model"], "--host", "127.0.0.1", "--port", str(self.cfg.get("internal_port")),
                "--api-key", self.internal_key, "--alias", prof["name"], "-c", str(prof["ctx"])]
        if threads:
            args += ["-t", str(threads), "-tb", str(threads)]
        if gpus:
            if prof.get("fit", True):
                args += ["--fit", "on", "--fit-target", str(prof.get("fit_target_mib", 2300))]
            else:
                args += ["-ngl", "99"]
                if prof.get("n_cpu_moe") not in (None, "", 0):
                    args += ["--n-cpu-moe", str(prof["n_cpu_moe"])]
        else:
            args += ["-ngl", "0"]
        if prof.get("kv_quant"):
            args += ["-ctk", prof["kv_quant"], "-ctv", prof["kv_quant"]]
        args += shlex.split(prof.get("extra_args") or "")
        return prefix + args, env, gpus, cpus

    # -- lifecycle
    def start(self, profile_name=None, via_router=False):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            self.routed = via_router
            prof = json.loads(json.dumps(self.cfg.profile(profile_name)))
            cmd, env, gpus, cpus = self.build(prof)
            exe = cmd[cmd.index("-m") - 1]
            if not os.path.exists(exe):
                self._fail(f"llama-server not found at {exe}")
                return
            if not os.path.exists(prof["model"]):
                self._fail(f"model not found: {prof['model']}")
                return
            self.logs.append(f"[ctl] starting profile {prof['name']} on GPUs {gpus or 'none'} CPUs {cpus or 'unpinned'}")
            self.logs.append("[ctl] " + " ".join(shlex.quote(c) for c in cmd).replace(self.internal_key, "***"))
            try:
                place_base = gpu_info() if gpus else []     # baseline VRAM before the child allocates anything
            except Exception:  # noqa: BLE001
                place_base = []
            try:
                self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True, bufsize=1, text=True,
                                             errors="replace")
            except OSError as e:
                self._fail(f"failed to launch: {e}")
                return
            self.state, self.error = "starting", ""
            self.started_at, self.ready_at = time.time(), None
            self.running_profile, self.running_cmd, self.running_gpus = prof, cmd, gpus
            self.running_cpus = cpus or []
            self._load_reset(prof["model"])
            self.place_base = place_base
            self.last_activity = time.time()
            self.ready_event.clear()
            proc = self.proc
            threading.Thread(target=self._reader, args=(proc,), daemon=True).start()
            threading.Thread(target=self._health_wait, args=(proc,), daemon=True).start()
            log(f"started {prof['name']} pid={proc.pid}")

    def _fail(self, msg):
        self.state, self.error = "failed", msg
        self.logs.append("[ctl] ERROR " + msg)
        self.ready_event.set()
        log(msg)

    def stop(self, reason="manual"):
        with self.lock:
            proc = self.proc
            if not proc or proc.poll() is not None:
                self.proc = None
                if self.state != "failed":
                    self.state = "stopped"
                return
            self.state = "stopping"
            self.logs.append(f"[ctl] stopping ({reason})")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(10)
        with self.lock:
            if self.proc is proc:
                self.proc = None
                self.state = "stopped"
                self.running_gpus = []
                self.ready_event.set()
        log(f"stopped ({reason})")

    def restart(self, profile_name=None):
        self.stop("restart")
        self.start(profile_name)

    def _reader(self, proc):
        for line in proc.stdout:
            line = line.rstrip()
            self.logs.append(line)
            self._note_line(proc, line)
            m = PROMPT_TPS_RE.search(line)
            if m:
                self.last_prompt_tps = float(m.group(1))
            elif TPS_RE.search(line):
                self.last_tps = float(TPS_RE.search(line).group(1))
        code = proc.wait()
        with self.lock:
            if self.proc is proc and self.state != "stopping":
                self.proc = None
                self.running_gpus = []
                self._fail(f"llama-server exited with code {code} (see log)")

    def _health_wait(self, proc):
        deadline = time.time() + self.cfg.get("start_timeout_s")
        port = self.cfg.get("internal_port")
        while time.time() < deadline and proc.poll() is None:
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                c.request("GET", "/health")
                r = c.getresponse()
                r.read()
                c.close()
                if r.status == 200:
                    with self.lock:
                        if self.proc is proc:
                            self.state, self.ready_at = "running", time.time()
                            self.last_activity = time.time()
                            self.logs.append(f"[ctl] ready after {self.ready_at - self.started_at:.0f}s")
                    self._freeze_placement()
                    self.ready_event.set()
                    return
            except OSError:
                pass
            time.sleep(1)
        if proc.poll() is None:
            self.logs.append("[ctl] start timeout, killing")
            self.stop("start timeout")
            self._fail("start timed out")

    def ensure_ready(self, timeout, profile=None):
        """Make a profile ready. profile=None means the active one (plain requests);
        a named profile (the router's pick) may replace the running one, but only
        when this is the only request in flight, so nobody's answer gets cut off."""
        with self.switch_lock:
            with self.lock:
                alive = bool(self.proc and self.proc.poll() is None)
                cur = self.running_profile["name"] if alive and self.running_profile else None
                switch = bool(profile and cur and cur != profile and self.inflight <= 1)
                need_start = not alive or self.state in ("stopped", "failed")
            if switch:
                self.stop(f"route switch to {profile}")
                need_start = True
            if need_start:
                self.start(profile, via_router=bool(profile))
        self.ready_event.wait(timeout)
        return self.state == "running"

    def serving(self):
        """Name of the profile currently loaded or loading, else None."""
        with self.lock:
            alive = bool(self.proc and self.proc.poll() is None)
            return self.running_profile["name"] if alive and self.running_profile else None

    def _idle_loop(self):
        while True:
            time.sleep(15)
            ttl = self.cfg.get("idle_ttl_min")
            if self.cfg.get("keep_loaded") or not ttl or ttl <= 0:
                continue
            with self.lock:
                idle = self.state == "running" and self.inflight == 0 and \
                    time.time() - self.last_activity > ttl * 60
            if idle:
                self.stop(f"idle {ttl} min")

    def touch(self, delta):
        with self.lock:
            self.inflight += delta
            self.last_activity = time.time()

    def status(self):
        with self.lock:
            prof = self.running_profile
            ttl = self.cfg.get("idle_ttl_min")
            unload_in = None
            if self.state == "running" and not self.cfg.get("keep_loaded") and ttl and self.inflight == 0:
                unload_in = max(0, int(ttl * 60 - (time.time() - self.last_activity)))
            stale = False
            if prof and self.state in ("starting", "running"):
                try:
                    stale = self.cfg.profile(prof["name"]) != prof or \
                        (not self.routed and self.cfg.get("active_profile") != prof["name"])
                except KeyError:
                    stale = True
            return {
                "state": self.state, "error": self.error,
                "pid": self.proc.pid if self.proc else None,
                "profile": prof["name"] if prof else None,
                "routed": self.routed,
                "gpus": self.running_gpus,
                "uptime_s": int(time.time() - self.started_at) if self.started_at and self.proc else None,
                "load_s": int(self.ready_at - self.started_at) if self.ready_at and self.started_at else None,
                "inflight": self.inflight,
                "idle_s": int(time.time() - self.last_activity),
                "unload_in_s": unload_in,
                "last_tps": self.last_tps, "last_prompt_tps": self.last_prompt_tps,
                "restart_needed": stale,
                "load": self.load_info(),
                "gen": self.gen_info(),
                "placement": self._placement_compact(),
            }


# ---------------------------------------------------------------- kev + router

class RouterError(Exception):
    pass


class KevService:
    """Runs Kev (kev.serve) as a managed child, started on first use and unloaded when idle."""

    def __init__(self, cfg, llama):
        self.cfg = cfg
        self.llama = llama
        self.lock = threading.RLock()
        self.proc = None
        self.state = "stopped"          # stopped | starting | running | failed
        self.error = ""
        self.gpus = []
        self.started_at = None
        self.ready_at = None
        self.last_activity = time.time()
        self.last_ms = None
        self.requests = 0
        self.launch_cfg = None          # settings the running process was started with
        self.logs = collections.deque(maxlen=1500)
        self.ready_event = threading.Event()
        threading.Thread(target=self._idle_loop, daemon=True).start()

    @property
    def k(self):
        return self.cfg.get("kev")

    def endpoint(self):
        if self.k["managed"]:
            return "http", "127.0.0.1", int(self.k["port"])
        u = urllib.parse.urlsplit(self.k["url"])
        return u.scheme or "http", u.hostname or "127.0.0.1", u.port or (443 if u.scheme == "https" else 80)

    def build(self):
        k = self.k
        kdir = os.path.expanduser(k["dir"])
        py = os.path.join(kdir, ".venv", "bin", "python")
        if k["command"].strip():
            cmd = shlex.split(k["command"].format(dir=kdir, model=k["model"], port=k["port"]))
        elif os.path.exists(py):
            cmd = [py, "-m", "kev.serve", "--run", k["model"], "--port", str(k["port"])]
        else:
            uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
            cmd = [uv, "run", "--frozen", "--extra", "serve", "python", "-m", "kev.serve",
                   "--run", k["model"], "--port", str(k["port"])]
        gpus = pick_gpus(k["gpus"], own=self.gpus if self.proc else (), exclude=self.llama.running_gpus,
                         bucket=self.cfg.get("gpu_bucket_mib") or 512, fallback=self.cfg.get("gpu_fallback_count") or 8)
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        env["KEV_DTYPE"] = k["dtype"]
        env.pop("KEV_CUDA_GRAPHS", None)
        if k.get("cuda_graphs") == "off":
            env["KEV_CUDA_GRAPHS"] = "0"
        elif k.get("cuda_graphs") == "on":
            env["KEV_CUDA_GRAPHS"] = "1"
        if k["hf_home"]:
            env["HF_HOME"] = os.path.expanduser(k["hf_home"])
        return cmd, env, kdir, gpus

    def start(self):
        with self.lock:
            if not self.k["managed"] or (self.proc and self.proc.poll() is None):
                return
            cmd, env, kdir, gpus = self.build()
            if not os.path.isdir(kdir):
                return self._fail(f"Kev directory not found: {kdir}")
            if not (os.path.exists(cmd[0]) or shutil.which(cmd[0])):
                return self._fail(f"Kev python not found: {cmd[0]} (run `uv sync --extra serve` in {kdir})")
            self.logs.append(f"[ctl] starting Kev {self.k['model']} ({self.k['dtype']}) on GPU {gpus}")
            self.logs.append("[ctl] " + " ".join(shlex.quote(c) for c in cmd))
            try:
                self.proc = subprocess.Popen(cmd, cwd=kdir, env=env, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             start_new_session=True, bufsize=1, text=True, errors="replace")
            except OSError as e:
                return self._fail(f"failed to launch Kev: {e}")
            self.state, self.error, self.gpus = "starting", "", gpus
            self.launch_cfg = {x: self.k.get(x) for x in KEV_LAUNCH_KEYS}
            self.started_at, self.ready_at = time.time(), None
            self.last_activity = time.time()
            self.ready_event.clear()
            proc = self.proc
            threading.Thread(target=self._reader, args=(proc,), daemon=True).start()
            threading.Thread(target=self._health_wait, args=(proc,), daemon=True).start()
            log(f"kev started pid={proc.pid} gpu={gpus}")

    def restart(self):
        self.stop("restart")
        self.start()

    def _fail(self, msg):
        self.state, self.error = "failed", msg
        self.logs.append("[ctl] ERROR " + msg)
        self.ready_event.set()
        log("kev: " + msg)

    def stop(self, reason="manual"):
        with self.lock:
            proc = self.proc
            if not proc or proc.poll() is not None:
                self.proc = None
                if self.state != "failed":
                    self.state = "stopped"
                self.gpus = []
                return
            self.logs.append(f"[ctl] stopping Kev ({reason})")
            self.state = "stopped"
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(15)
                break
            except subprocess.TimeoutExpired:
                continue
        with self.lock:
            if self.proc is proc:
                self.proc, self.gpus = None, []
                self.ready_event.set()
        log(f"kev stopped ({reason})")

    def _reader(self, proc):
        for line in proc.stdout:
            self.logs.append(line.rstrip())
        code = proc.wait()
        with self.lock:
            if self.proc is proc and self.state != "stopped":
                self.proc, self.gpus = None, []
                self._fail(f"Kev exited with code {code} (see its log)")

    def _health_wait(self, proc):
        deadline = time.time() + int(self.k["start_timeout_s"])
        port = int(self.k["port"])
        while time.time() < deadline and proc.poll() is None:
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                c.request("GET", "/v1/models")
                r = c.getresponse()
                r.read()
                c.close()
                if r.status == 200:
                    with self.lock:
                        if self.proc is proc:
                            self.state, self.ready_at = "running", time.time()
                            self.last_activity = time.time()
                            self.logs.append(f"[ctl] Kev ready after {self.ready_at - self.started_at:.0f}s")
                    self.ready_event.set()
                    return
            except OSError:
                pass
            time.sleep(1)
        if proc.poll() is None:
            self.stop("start timeout")
            self._fail("Kev start timed out")

    def ensure_ready(self):
        if not self.k["managed"]:
            return True
        with self.lock:
            if self.state in ("stopped", "failed") or not self.proc:
                self.start()
        self.ready_event.wait(int(self.k["start_timeout_s"]))
        return self.state == "running"

    def _idle_loop(self):
        while True:
            time.sleep(15)
            ttl = self.k["idle_ttl_min"]
            if not self.k["managed"] or not ttl or ttl <= 0:
                continue
            if self.state == "running" and time.time() - self.last_activity > ttl * 60:
                self.stop(f"idle {ttl} min")

    def systemone(self, state, questions, timeout=120):
        if not self.ensure_ready():
            raise RouterError(self.error or f"Kev not ready ({self.state})")
        scheme, host, port = self.endpoint()
        conn = (http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection)(host, port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if self.k["api_key"]:
            headers["Authorization"] = f"Bearer {self.k['api_key']}"
        t0 = time.time()
        try:
            conn.request("POST", "/v1/systemone", json.dumps(
                {"model": "kev-latest", "state": state, "questions": questions}), headers)
            r = conn.getresponse()
            raw = r.read()
        except OSError as e:
            raise RouterError(f"Kev request failed: {e}") from e
        finally:
            conn.close()
        self.last_activity = time.time()
        self.last_ms = int((time.time() - t0) * 1000)
        self.requests += 1
        if r.status != 200:
            raise RouterError(f"Kev returned HTTP {r.status}: {raw[:200].decode(errors='replace')}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RouterError("Kev returned invalid JSON") from e

    def status(self):
        with self.lock:
            ttl = self.k["idle_ttl_min"]
            unload = None
            if self.state == "running" and self.k["managed"] and ttl:
                unload = max(0, int(ttl * 60 - (time.time() - self.last_activity)))
            stale = bool(self.k["managed"] and self.state in ("starting", "running") and self.launch_cfg
                         and any(self.launch_cfg.get(x) != self.k.get(x) for x in KEV_LAUNCH_KEYS))
            return {"state": self.state if self.k["managed"] else "external", "error": self.error,
                    "managed": self.k["managed"], "pid": self.proc.pid if self.proc else None,
                    "restart_needed": stale,
                    "running_model": self.launch_cfg.get("model") if self.launch_cfg and self.proc else None,
                    "gpus": self.gpus, "last_ms": self.last_ms, "requests": self.requests,
                    "load_s": int(self.ready_at - self.started_at) if self.ready_at and self.started_at else None,
                    "unload_in_s": unload}


class Router:
    """Asks Kev which profile should answer a prompt. Never switches anything itself."""

    def __init__(self, cfg, kev):
        self.cfg = cfg
        self.kev = kev

    @property
    def r(self):
        return self.cfg.get("router")

    def enabled(self):
        return bool(self.r["enabled"])

    @staticmethod
    def prompt_of(body):
        """Last user turn of a chat request, or the prompt of a completions request."""
        if isinstance(body.get("prompt"), str):
            return body["prompt"]
        for m in reversed(body.get("messages") or []):
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        return ""

    def decide(self, prompt):
        r = self.r
        routes = r["routes"]
        if not routes:
            raise RouterError("no routes configured")
        text = prompt.strip()[-int(r["max_chars"]):]
        if not text:
            return {"profile": r["default_profile"], "choice": None, "probability": None,
                    "probabilities": {}, "fallback": "empty prompt", "kev_ms": 0}
        questions = {"route": {"type": "choice", "instructions": r["instructions"],
                               "criteria": {x["profile"]: x["description"] for x in routes}}}
        res = self.kev.systemone(text, questions)
        ans = (res.get("answers") or {}).get("route")
        if not ans or ans.get("type") != "choice":
            raise RouterError("Kev gave no route answer")
        choice, probs = ans["choice"], ans.get("probabilities") or {}
        p = float(probs.get(choice, 0))
        out = {"choice": choice, "probability": p, "probabilities": probs,
               "kev_ms": self.kev.last_ms, "profile": choice, "fallback": None}
        if p < float(r["threshold"]):
            out["profile"] = r["default_profile"]
            out["fallback"] = f"p={p:.2f} below threshold {r['threshold']}"
        return out

    def safe_decide(self, prompt):
        """Like decide(), but a Kev failure falls back to the default profile."""
        try:
            return self.decide(prompt)
        except RouterError as e:
            return {"profile": self.r["default_profile"], "choice": None, "probability": None,
                    "probabilities": {}, "fallback": f"Kev unavailable: {e}", "kev_ms": None}


# ---------------------------------------------------------------- http

class Sampler:
    """Background 1 s sampler feeding /api/metrics: RAM, per-NUMA used, GPUs, per-core CPU, child RSS."""

    def __init__(self, sup, interval=1.0, keep=600):
        self.sup, self.interval = sup, interval
        self.samples = collections.deque(maxlen=keep)
        self.lock = threading.Lock()
        self._prev_cpu = None
        self.cpu_ids = []

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _cpu_busy(self):
        """Per-core busy % since the previous call, from /proc/stat deltas."""
        cur = {}
        for line in read("/proc/stat").splitlines():
            m = re.match(r"cpu(\d+)\s+(.*)", line)
            if not m:
                continue
            f = [int(x) for x in m.group(2).split()[:8]]
            if len(f) >= 4:
                cur[int(m.group(1))] = (sum(f), f[3] + (f[4] if len(f) > 4 else 0))  # total, idle+iowait
        prev, self._prev_cpu = self._prev_cpu, cur
        ids = sorted(cur)
        self.cpu_ids = ids
        if not prev:
            return ids, [0] * len(ids)
        out = []
        for i in ids:
            if i in prev and cur[i][0] > prev[i][0]:
                dt, di = cur[i][0] - prev[i][0], cur[i][1] - prev[i][1]
                out.append(max(0, min(100, round(100 * (dt - di) / dt))))
            else:
                out.append(0)
        return ids, out

    def _safe(self, name, *a):
        try:
            return getattr(self.sup, name)(*a)
        except Exception:  # noqa: BLE001
            return None

    def sample(self):
        pid = None
        try:
            pid = self.sup.proc.pid if self.sup.proc else None
        except Exception:  # noqa: BLE001
            pass
        mem = meminfo(pid)
        ids, busy = self._cpu_busy()
        gpus = gpu_info()
        self.sup.place_gpus_cache = (time.time(), gpus)
        return {"t": round(time.time(), 2),
                "ram_used": mem["used_mib"], "ram_cached": mem["cached_mib"], "ram_free": mem["free_mib"],
                "nodes": [n["used_mib"] for n in mem["nodes"]],
                "gpus": [[g["index"], g["mem_used"], g["util"]] for g in gpus],
                "cpu": busy, "rss": mem["child"]["rss_mib"], "read_bytes": mem["child"]["read_bytes"],
                "load": self._safe("load_info", mem["child"]), "gen": self._safe("gen_info")}

    def _loop(self):
        while True:
            t0 = time.time()
            try:
                s = self.sample()
                with self.lock:
                    self.samples.append(s)
            except Exception as e:  # noqa: BLE001 - sampler must survive anything
                log(f"sampler: {e}")
            time.sleep(max(0.05, self.interval - (time.time() - t0)))

    def since(self, ts):
        with self.lock:
            return [s for s in self.samples if s["t"] > ts]


# ---------------------------------------------------------------- in-app update (git fast-forward of this checkout)

CRED_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s]*@")
RESTART_FILES = ("bizenctl.py", "install.sh")   # a change to either needs a controller restart
UPDATE_FETCH_TIMEOUT = 60
UPDATE_PULL_TIMEOUT = 90


def redact(s):
    """Strip user:pass@ from any URL inside a string (anything shown or logged goes through this)."""
    return CRED_RE.sub(r"\1", s or "")


def _term_self():
    os.kill(os.getpid(), signal.SIGTERM)   # same graceful path as POST /api/controller/restart


class Updater:
    """Fast-forward-only self update from the checkout's upstream. Never raises; one operation at a time."""

    def __init__(self, repo=HERE, sup=None, restart_hook=None, delay=1.5):
        self.repo, self.sup = repo, sup
        self.restart_hook = restart_hook or _term_self
        self.delay = delay
        self.lock = threading.Lock()
        self.restarting = False

    # -- git plumbing
    def git(self, *args, timeout=15):
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="true", LC_ALL="C")
        try:
            r = subprocess.run(["git", "-C", self.repo, "--no-pager"] + list(args), capture_output=True,
                               text=True, timeout=timeout, env=env)
            return r.returncode, redact(r.stdout).strip(), redact(r.stderr).strip()
        except subprocess.TimeoutExpired:
            return 124, "", f"git {args[0]} timed out after {timeout} s"
        except (OSError, ValueError) as e:
            return 127, "", redact(str(e))

    def is_git(self):
        rc, out, _ = self.git("rev-parse", "--show-toplevel")
        return rc == 0 and os.path.realpath(out) == os.path.realpath(self.repo)

    @staticmethod
    def host_of(url):
        url = redact(url or "").strip()
        m = re.match(r"^[\w.-]+@([^:/\s]+):", url)            # scp-like git@host:path
        if m:
            return m.group(1)
        try:
            return urllib.parse.urlsplit(url).hostname
        except ValueError:
            return None

    def model_loaded(self):
        return bool(self.sup and getattr(self.sup, "state", "") in ("running", "starting"))

    def info(self):
        """Everything GET /api/controller/version returns."""
        out = {"is_git": False, "commit": None, "commit_full": None, "branch": None, "dirty": None,
               "remote_host": None, "upstream": None, "ahead": None, "behind": None, "last_fetch": None,
               "busy": self.lock.locked() or self.restarting}
        if not self.is_git():
            return out
        out["is_git"] = True
        rc, full, _ = self.git("rev-parse", "HEAD")
        if rc == 0:
            out["commit_full"], out["commit"] = full, full[:7]
        rc, br, _ = self.git("rev-parse", "--abbrev-ref", "HEAD")
        out["branch"] = br if rc == 0 else None
        rc, st, _ = self.git("status", "--porcelain")
        out["dirty"] = bool(st) if rc == 0 else None
        rc, up, _ = self.git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        out["upstream"] = up if rc == 0 and up else None
        remote = up.split("/", 1)[0] if out["upstream"] else "origin"
        rc, url, _ = self.git("remote", "get-url", remote)
        out["remote_host"] = self.host_of(url) if rc == 0 else None
        if out["upstream"]:
            rc, cnt, _ = self.git("rev-list", "--left-right", "--count", "HEAD...@{u}")
            if rc == 0 and len(cnt.split()) == 2:
                out["ahead"], out["behind"] = (int(x) for x in cnt.split())
        rc, gd, _ = self.git("rev-parse", "--git-dir")
        if rc == 0:
            try:
                out["last_fetch"] = os.path.getmtime(os.path.join(self.repo, gd, "FETCH_HEAD"))
            except OSError:
                pass
        return out

    # -- check / apply
    def check(self, fetch=True):
        """Fetch and describe what an update would bring. {ok: False, reason} when it must not be applied."""
        v = self.info()
        res = {"ok": False, "reason": None, "commit": v["commit"], "branch": v["branch"], "upstream": v["upstream"],
               "ahead": v["ahead"], "behind": v["behind"], "dirty": v["dirty"], "remote_host": v["remote_host"],
               "commits": [], "files": [], "restart_needed": False,
               "model_loaded": self.model_loaded(), "would_unload": False}
        if not v["is_git"]:
            res["reason"] = "this install is not a git checkout, so it cannot update itself (use your package or copy method)"
        elif not v["upstream"]:
            res["reason"] = "the current branch has no upstream to update from"
        elif v["dirty"]:
            res["reason"] = "the working tree has local changes; commit or stash them first"
        elif v["dirty"] is None:
            res["reason"] = "could not read the git status"
        if res["reason"]:
            return res
        if fetch:
            rc, _, err = self.git("fetch", "--quiet", timeout=UPDATE_FETCH_TIMEOUT)
            if rc != 0:
                res["reason"] = "git fetch failed: " + (err.splitlines()[-1] if err else f"exit {rc}")
                return res
            v = self.info()
            res.update(ahead=v["ahead"], behind=v["behind"])
        if v["ahead"] is None or v["behind"] is None:
            res["reason"] = "could not compare with the upstream"
        elif v["ahead"] and v["behind"]:
            res["reason"] = f"local and upstream have diverged ({v['ahead']} ahead, {v['behind']} behind); resolve by hand"
        elif v["ahead"]:
            res["reason"] = f"{v['ahead']} local commit(s) not on the upstream; nothing to pull"
        if res["reason"]:
            return res
        res["ok"] = True
        if v["behind"]:
            rc, lg, _ = self.git("log", "--oneline", "-n", "20", "HEAD..@{u}")
            res["commits"] = lg.splitlines() if rc == 0 and lg else []
            rc, fl, _ = self.git("diff", "--name-only", "HEAD..@{u}")
            res["files"] = fl.splitlines() if rc == 0 and fl else []
            res["restart_needed"] = any(f in RESTART_FILES for f in res["files"])
            res["would_unload"] = res["restart_needed"] and res["model_loaded"]
        return res

    def apply(self, confirm):
        """Returns (http_code, body). On a restart the caller must send the body, then call schedule_restart()."""
        if confirm is not True:
            return 400, {"ok": False, "reason": "confirm:true required"}
        if self.restarting or not self.lock.acquire(blocking=False):
            return 409, {"ok": False, "reason": "another update is already running"}
        try:
            chk = self.check()
            if not chk["ok"]:
                log("update apply refused: " + redact(chk["reason"]))
                return 409, {"ok": False, "reason": chk["reason"]}
            frm = chk["commit"]
            if not chk["behind"]:
                return 200, {"ok": True, "from": frm, "to": frm, "restart_needed": False, "restarting": False,
                             "note": "already up to date"}
            rc, _, err = self.git("pull", "--ff-only", "--quiet", timeout=UPDATE_PULL_TIMEOUT)
            if rc != 0:
                reason = "git pull --ff-only failed: " + (err.splitlines()[-1] if err else f"exit {rc}")
                log("update apply failed: " + reason)
                return 500, {"ok": False, "reason": reason}
            to = self.info()["commit"]
            restart = chk["restart_needed"]
            log(f"update applied: {frm} -> {to}" + (" (restarting controller)" if restart else " (no restart needed)"))
            if restart:
                self.restarting = True
            return 200, {"ok": True, "from": frm, "to": to, "restart_needed": restart, "restarting": restart,
                         "files": chk["files"]}
        finally:
            self.lock.release()

    def schedule_restart(self):
        def go():
            time.sleep(self.delay)
            try:
                self.restart_hook()
            except Exception as e:  # noqa: BLE001
                log("update restart hook failed: " + redact(str(e)))
        threading.Thread(target=go, daemon=True).start()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # close-delimited bodies keep streaming proxying simple
    server_version = "bizen-ctl"
    sessions = {}

    def log_message(self, fmt, *args):
        pass

    @property
    def cfg(self):
        return self.server.cfg

    @property
    def sup(self):
        return self.server.sup

    @property
    def dl(self):
        return self.server.dl

    # -- helpers
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def json_body(self):
        try:
            return json.loads(self.body() or b"{}")
        except json.JSONDecodeError:
            return None

    def cookie(self, name):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                return v
        return None

    def is_admin(self):
        tok = self.cookie("bizen_session")
        exp = self.sessions.get(tok) if tok else None
        return bool(exp and exp > time.time())

    def api_key_ok(self):
        auth = self.headers.get("Authorization") or ""
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else (self.headers.get("x-api-key") or "")
        return any(hmac.compare_digest(key, k) for k in self.cfg.get("api_keys")) if key else False

    # -- routing
    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def do_PUT(self):
        self.route("PUT")

    def do_DELETE(self):
        self.route("DELETE")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def route(self, method):
        path = urllib.parse.urlsplit(self.path).path
        try:
            if path in ("/", "/index.html") and method == "GET":
                return self.serve_ui()
            if path == "/api/login" and method == "POST":
                return self.api_login()
            if path.startswith("/api/"):
                if not self.is_admin():
                    return self.send_json({"error": "login required"}, 401)
                return self.admin_api(method, path)
            if path == "/health" and method == "GET":
                return self.send_json({"status": "ok", "model_state": self.sup.state})
            return self.proxy(method)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def serve_ui(self):
        with open(os.path.join(HERE, "ui.html"), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def api_login(self):
        d = self.json_body() or {}
        if not hmac.compare_digest(str(d.get("password", "")), self.cfg.get("admin_password")):
            time.sleep(1)
            return self.send_json({"error": "wrong password"}, 403)
        tok = secrets.token_urlsafe(32)
        self.sessions[tok] = time.time() + 7 * 86400
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Set-Cookie", f"bizen_session={tok}; HttpOnly; SameSite=Strict; Path=/; Max-Age={7*86400}")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- admin api
    def admin_api(self, method, path):
        sup, cfg = self.sup, self.cfg
        if path == "/api/status" and method == "GET":
            cpus, node = auto_pin()
            allowed, _, _ = cpu_topology()
            return self.send_json({
                "server": sup.status(), "gpus": gpu_info(), "mem": meminfo(sup.proc.pid if sup.proc else None),
                "cpus_allowed": allowed, "auto_pin": cpus, "auto_pin_node": node,
                "active_profile": cfg.get("active_profile"),
                "idle_ttl_min": cfg.get("idle_ttl_min"), "keep_loaded": cfg.get("keep_loaded"),
                "numactl": bool(shutil.which("numactl")),
                "unit": unit_status(),
                "kev": self.server.kev.status(),
                "router_enabled": self.server.router.enabled(),
                "downloads_active": self.dl.active(),
            })
        if path == "/api/loadmap" and method == "GET":
            lm = parse_startup_log(list(sup.logs))
            gp = list(sup.running_gpus)
            for d in lm["devices"]:
                d["gpu"] = gp[d["index"]] if d.get("index") is not None and d["index"] < len(gp) else None
            lm.update(state=sup.state, running_gpus=gp, placement=sup.placement_info())
            return self.send_json(lm)
        if path == "/api/metrics" and method == "GET":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                since = float((q.get("since") or ["0"])[0])
            except ValueError:
                since = 0.0
            smp = self.server.sampler
            pin = sup.running_cpus if sup.proc else auto_pin()[0]
            return self.send_json({"now": time.time(), "interval_s": smp.interval, "cpu_ids": smp.cpu_ids,
                                   "cpus_pinned": pin, "samples": smp.since(since)})
        if path == "/api/config" and method == "GET":
            return self.send_json(public_config(cfg.data))
        if path == "/api/system" and method == "GET":
            return self.send_json(system_info(cfg.data))
        if path == "/api/library" and method == "GET":
            return self.send_json(library_scan_now(cfg, sup))
        if path == "/api/hf/search" and method == "GET":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            term = (q.get("q") or [""])[0].strip()[:200]
            sort = (q.get("sort") or ["downloads"])[0]
            if sort not in HF_SORTS:
                return self.send_json({"error": "sort must be one of " + ", ".join(HF_SORTS)}, 400)
            try:
                limit = max(1, min(50, int((q.get("limit") or ["20"])[0])))
            except ValueError:
                return self.send_json({"error": "limit must be an integer"}, 400)
            try:
                return self.send_json(hf_search(cfg.data, term, sort, limit))
            except HFError as e:
                return self.send_json({"error": e.msg}, e.code)
        if path == "/api/hf/repo" and method == "GET":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            repo = (q.get("id") or [""])[0].strip()
            if not hf_valid_repo(repo):
                return self.send_json({"error": "id must look like owner/name"}, 400)
            try:
                return self.send_json(hf_repo_view(cfg.data, repo))
            except HFError as e:
                return self.send_json({"error": e.msg}, e.code)
        if path == "/api/downloads" and method == "GET":
            return self.send_json(self.dl.list())
        if path == "/api/downloads" and method == "POST":
            d = self.json_body()
            if not isinstance(d, dict):
                return self.send_json({"error": "invalid JSON"}, 400)
            repo = str(d.get("repo") or "")
            try:
                if not hf_valid_repo(repo):
                    raise DlError(400, "repo must look like owner/name")
                sizes = {f["path"]: f["bytes"] for f in hf_repo_raw(cfg.data, repo)["files"]}
                return self.send_json({"ok": True, "download": self.dl.add(dl_plan(cfg.data, repo, d.get("files"), sizes))})
            except HFError as e:
                return self.send_json({"error": e.msg}, e.code)
            except DlError as e:
                return self.send_json({"error": e.msg}, e.code)
        mm = re.fullmatch(r"/api/downloads/([0-9a-f]{8})/(cancel|pause|resume|remove)", path)
        if mm and method == "POST":
            d = self.json_body()
            if not isinstance(d, dict):
                return self.send_json({"error": "invalid JSON"}, 400)
            try:
                if mm.group(2) == "remove":
                    return self.send_json(self.dl.remove(mm.group(1), d.get("delete_partial") is True))
                return self.send_json({"ok": True, "download": getattr(self.dl, mm.group(2))(mm.group(1))})
            except DlError as e:
                return self.send_json({"error": e.msg}, e.code)
        if path == "/api/library/profile" and method == "POST":
            d = self.json_body()
            if not isinstance(d, dict):
                return self.send_json({"error": "invalid JSON"}, 400)
            code, body = library_draft(cfg, sup, d)
            return self.send_json(body, code)
        if path == "/api/library/delete" and method == "POST":
            d = self.json_body()
            if not isinstance(d, dict):
                return self.send_json({"error": "invalid JSON"}, 400)
            code, body = library_delete(cfg, sup, d.get("id"), d.get("confirm"))
            return self.send_json(body, code)
        if path == "/api/system/settings" and method == "POST":
            d = self.json_body()
            if not isinstance(d, dict):
                return self.send_json({"error": "invalid JSON"}, 400)
            try:
                return self.send_json(apply_settings(cfg, d, dry_run=bool(d.get("dry_run"))))
            except (ValueError, KeyError, TypeError, OSError) as e:
                return self.send_json({"error": str(e)}, 400)
        if path == "/api/config" and method == "PUT":
            d = self.json_body()
            if d is None:
                return self.send_json({"error": "invalid JSON"}, 400)
            try:
                cfg.save(d)
            except (ValueError, KeyError, TypeError) as e:
                return self.send_json({"error": str(e)}, 400)
            return self.send_json({"ok": True, "status": sup.status()})
        if path == "/api/preview" and method == "POST":
            d = self.json_body() or {}
            try:
                prof = dict(PROFILE_DEFAULTS, **d)
                cmd, env, gpus, cpus = sup.build(prof)
            except Exception as e:  # noqa: BLE001 - surface any build error to the UI
                return self.send_json({"error": str(e)}, 400)
            line = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} " + " ".join(shlex.quote(c) for c in cmd)
            return self.send_json({"cmd": line.replace(sup.internal_key, "***"), "gpus": gpus, "cpus": cpus})
        if path == "/api/start" and method == "POST":
            sup.start()
            return self.send_json({"ok": True, "status": sup.status()})
        if path == "/api/stop" and method == "POST":
            threading.Thread(target=sup.stop, args=("manual",), daemon=True).start()
            return self.send_json({"ok": True})
        if path == "/api/restart" and method == "POST":
            threading.Thread(target=sup.restart, daemon=True).start()
            return self.send_json({"ok": True})
        if path == "/api/logs" and method == "GET":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            n = int((q.get("n") or ["400"])[0])
            lines = list(sup.logs)[-n:]
            return self.send_json({"lines": [l.replace(sup.internal_key, "***") for l in lines]})
        if path == "/api/models" and method == "GET":
            models = []
            for d in cfg.get("model_dirs"):
                for p in sorted(glob.glob(os.path.join(os.path.expanduser(d), "**", "*.gguf"), recursive=True)):
                    if "mmproj" in os.path.basename(p).lower():
                        continue
                    try:
                        models.append({"path": p, "gib": round(os.path.getsize(p) / 2**30, 1)})
                    except OSError:
                        pass
            return self.send_json({"models": models})
        if path == "/api/test" and method == "POST":
            d = self.json_body() or {}
            return self.test_prompt(str(d.get("prompt") or "Say hi in one sentence."))
        if path == "/api/chat" and method == "POST":
            return self.chat_stream()
        if path == "/api/kev/start" and method == "POST":
            threading.Thread(target=self.server.kev.start, daemon=True).start()
            return self.send_json({"ok": True})
        if path == "/api/kev/stop" and method == "POST":
            threading.Thread(target=self.server.kev.stop, args=("manual",), daemon=True).start()
            return self.send_json({"ok": True})
        if path == "/api/kev/restart" and method == "POST":
            threading.Thread(target=self.server.kev.restart, daemon=True).start()
            return self.send_json({"ok": True})
        if path == "/api/kev/presets" and method == "GET":
            return self.send_json({"presets": KEV_PRESETS})
        if path == "/api/kev/logs" and method == "GET":
            return self.send_json({"lines": list(self.server.kev.logs)[-300:]})
        if path == "/api/route/test" and method == "POST":
            d = self.json_body() or {}
            prompt = str(d.get("prompt") or "")
            if not prompt.strip():
                return self.send_json({"error": "prompt is empty"}, 400)
            try:
                dec = self.server.router.decide(prompt)
            except RouterError as e:
                return self.send_json({"error": str(e)}, 502)
            dec["serving"] = sup.serving()
            if not d.get("run"):
                return self.send_json({"decision": dec})
            return self.test_prompt(prompt, profile=dec["profile"], decision=dec)
        if path == "/api/controller/version" and method == "GET":
            return self.send_json(self.server.updater.info())
        if path == "/api/controller/update" and method == "POST":
            d = self.json_body()
            up = self.server.updater
            if not isinstance(d, dict) or d.get("action") not in ("check", "apply"):
                return self.send_json({"error": "send {action: 'check'|'apply'}"}, 400)
            if d["action"] == "check":
                if not up.lock.acquire(blocking=False):
                    return self.send_json({"ok": False, "reason": "another update is already running"})
                try:
                    res = up.check()
                finally:
                    up.lock.release()
                log("update check: " + (f"{res['behind']} behind" if res["ok"] else "refused: " + redact(res["reason"])))
                return self.send_json(res)
            code, body = up.apply(d.get("confirm"))
            self.send_json(body, code)
            if body.get("restarting"):
                up.schedule_restart()
            return
        if path == "/api/controller/restart" and method == "POST":
            self.send_json({"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.5), os.kill(os.getpid(), signal.SIGTERM)), daemon=True).start()
            return
        return self.send_json({"error": "not found"}, 404)

    def chat_stream(self):
        """Admin chat: stream the child's /v1/chat/completions SSE back to the browser.
        Headers go out first so the UI can show a loading state: while the model loads we send
        `data: {"bizen":"loading",...}` keepalives, then `{"bizen":"ready"}`, then llama-server's chunks."""
        d = self.json_body()
        msgs = d.get("messages") if isinstance(d, dict) else None
        if not isinstance(msgs, list) or not msgs or not all(
                isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), (str, list))
                for m in msgs):
            return self.send_json({"error": "messages must be a non-empty list of {role, content}"}, 400)
        payload = {"messages": msgs, "stream": True, "stream_options": {"include_usage": True}}
        try:
            payload["max_tokens"] = max(1, min(int(d.get("max_tokens") or 1024), 131072))
            if d.get("temperature") is not None:
                payload["temperature"] = max(0.0, min(float(d["temperature"]), 5.0))
        except (TypeError, ValueError):
            return self.send_json({"error": "max_tokens/temperature must be numbers"}, 400)
        sup = self.sup
        sup.touch(+1)
        c = None
        gid = None
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            res = {}
            th = threading.Thread(target=lambda: res.update(ok=sup.ensure_ready(self.cfg.get("start_timeout_s"))), daemon=True)
            th.start()
            while th.is_alive():
                emit({"bizen": "loading", "state": sup.state})
                th.join(1.5)
            if not res.get("ok"):
                return emit({"error": sup.error or f"server not ready ({sup.state})"})
            emit({"bizen": "ready", "profile": sup.serving()})
            c = http.client.HTTPConnection("127.0.0.1", self.cfg.get("internal_port"), timeout=600)
            c.request("POST", "/v1/chat/completions", body=json.dumps(payload),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {sup.internal_key}"})
            gid = sup.gen_begin()
            counter = SSECount()
            r = c.getresponse()
            if r.status != 200:
                return emit({"error": r.read()[:500].decode(errors="replace") or f"upstream status {r.status}"})
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                sup.last_activity = time.time()
                sup.gen_chunk(gid, counter.feed(chunk))
        except (BrokenPipeError, ConnectionResetError):
            pass   # browser hit Stop: closing the upstream connection below stops generation
        except OSError as e:
            try:
                self.wfile.write(b"data: " + json.dumps({"error": f"upstream error: {e}"}).encode() + b"\n\n")
            except OSError:
                pass
        finally:
            if c:
                c.close()
            if gid:
                sup.gen_end(gid)
            sup.touch(-1)

    def test_prompt(self, prompt, profile=None, decision=None):
        sup = self.sup
        sup.touch(+1)
        try:
            if not sup.ensure_ready(self.cfg.get("start_timeout_s"), profile=profile):
                return self.send_json({"error": sup.error or f"server not ready ({sup.state})"}, 503)
            c = http.client.HTTPConnection("127.0.0.1", self.cfg.get("internal_port"), timeout=600)
            payload = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 256})
            t0 = time.time()
            c.request("POST", "/v1/chat/completions", body=payload,
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {sup.internal_key}"})
            r = c.getresponse()
            raw = r.read()
            c.close()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return self.send_json({"error": raw[:500].decode(errors="replace")}, 502)
            if r.status != 200:
                return self.send_json({"error": data}, 502)
            msg = data["choices"][0]["message"]
            t = data.get("timings") or {}
            return self.send_json({"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
                                   "tps": t.get("predicted_per_second"), "prompt_tps": t.get("prompt_per_second"),
                                   "tokens": t.get("predicted_n"), "wall_s": round(time.time() - t0, 1),
                                   "profile": sup.serving(), "decision": decision})
        finally:
            sup.touch(-1)

    # -- proxy
    def proxy(self, method):
        if not self.api_key_ok():
            return self.send_json({"error": {"message": "invalid or missing API key", "type": "auth"}}, 401)
        sup, router = self.sup, self.server.router
        path = urllib.parse.urlsplit(self.path).path
        if path == "/v1/models" and method == "GET":
            ids = [p["name"] for p in self.cfg.get("profiles")]
            if router.enabled():
                ids.append(self.cfg.get("router")["alias"])
            return self.send_json({"object": "list", "data": [
                {"id": i, "object": "model", "owned_by": "bizen-ctl"} for i in ids]})
        if path == "/v1/route" and method == "POST":
            return self.route_endpoint()
        body = self.body()
        sup.touch(+1)
        target, extra = None, {}
        gid, tail, tokens, tps = None, b"", None, None
        try:
            if method == "POST" and router.enabled() and path in ("/v1/chat/completions", "/v1/completions"):
                try:
                    req = json.loads(body or b"{}")
                except json.JSONDecodeError:
                    req = {}
                if isinstance(req, dict) and req.get("model") == self.cfg.get("router")["alias"]:
                    dec = router.safe_decide(Router.prompt_of(req))
                    target = dec["profile"]
                    req["model"] = target
                    body = json.dumps(req).encode()
                    extra["X-Bizen-Route"] = target
                    if dec.get("probability") is not None:
                        extra["X-Bizen-Route-P"] = f"{dec['probability']:.3f}"
                    if dec.get("fallback"):
                        extra["X-Bizen-Route-Fallback"] = re.sub(r"[^\x20-\x7e]", "?", dec["fallback"])[:200]
            if not sup.ensure_ready(self.cfg.get("start_timeout_s"), profile=target):
                return self.send_json({"error": {"message": sup.error or f"model not ready ({sup.state})",
                                                 "type": "unavailable"}}, 503)
            if target:
                served = sup.serving()
                if served and served != target:
                    extra["X-Bizen-Route-Served"] = served   # busy: kept the loaded profile
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS
                       and k.lower() not in ("authorization", "x-api-key", "cookie")}
            headers["Authorization"] = f"Bearer {sup.internal_key}"
            if body:
                headers["Content-Length"] = str(len(body))
            c = http.client.HTTPConnection("127.0.0.1", self.cfg.get("internal_port"), timeout=3600)
            if method == "POST" and path in GEN_PATHS:
                gid = sup.gen_begin()
            counter = SSECount()
            c.request(method, self.path, body=body or None, headers=headers)
            r = c.getresponse()
            streamed = "event-stream" in (r.getheader("Content-Type") or "")
            self.send_response(r.status, r.reason)
            for k, v in r.getheaders():
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                sup.last_activity = time.time()
                if gid:
                    if streamed:
                        sup.gen_chunk(gid, counter.feed(chunk))
                    elif len(tail) < (2 << 20):
                        tail += chunk
            c.close()
            if gid and not streamed and tail:
                try:   # a non-streamed reply: take the totals from its usage/timings
                    rep = json.loads(tail)
                    tokens = (rep.get("usage") or {}).get("completion_tokens") or (rep.get("timings") or {}).get("predicted_n")
                    tps = (rep.get("timings") or {}).get("predicted_per_second")
                except (ValueError, AttributeError):
                    pass
        except OSError as e:
            try:
                self.send_json({"error": {"message": f"upstream error: {e}", "type": "upstream"}}, 502)
            except OSError:
                pass
        finally:
            if gid:
                sup.gen_end(gid, tokens, tps)
            sup.touch(-1)


def _route_endpoint(self):
    """POST /v1/route {"prompt": ...} or {"messages": [...]}: Kev's decision only, nothing is loaded or switched."""
    router = self.server.router
    if not router.enabled():
        return self.send_json({"error": {"message": "router is disabled", "type": "unavailable"}}, 503)
    req = self.json_body()
    if not isinstance(req, dict):
        return self.send_json({"error": {"message": "invalid JSON", "type": "bad_request"}}, 400)
    prompt = Router.prompt_of(req)
    if not prompt.strip():
        return self.send_json({"error": {"message": "send 'prompt' or 'messages'", "type": "bad_request"}}, 400)
    try:
        dec = router.decide(prompt)
    except RouterError as e:
        return self.send_json({"error": {"message": str(e), "type": "upstream"}}, 502)
    dec["serving"] = self.sup.serving()
    dec["would_switch"] = bool(dec["serving"] and dec["serving"] != dec["profile"])
    return self.send_json(dec)


Handler.route_endpoint = _route_endpoint


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def unit_status():
    if not shutil.which("systemctl"):
        return None
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "bizen-ctl"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    a = ap.parse_args()
    cfg = Config(a.config)
    Config.validate(cfg.data)
    sup = Supervisor(cfg)
    kev = KevService(cfg, sup)
    sup.kev = kev
    router = Router(cfg, kev)
    srv = Server((cfg.get("listen_host"), cfg.get("listen_port")), Handler)
    srv.cfg, srv.sup, srv.kev, srv.router = cfg, sup, kev, router
    srv.dl = Downloader(cfg)
    srv.updater = Updater(HERE, sup)
    srv.sampler = Sampler(sup)
    srv.sampler.start()

    def shutdown(*_):
        log("shutting down")
        sup.stop("controller exit")
        kev.stop("controller exit")
        srv.dl.shutdown()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log(f"bizen-ctl listening on {cfg.get('listen_host')}:{cfg.get('listen_port')} "
        f"(profile {cfg.get('active_profile')}, idle unload {cfg.get('idle_ttl_min')} min)")
    if cfg.get("keep_loaded"):
        sup.start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
