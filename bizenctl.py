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
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host"}
TPS_RE = re.compile(r"eval time\s*=.*?([\d.]+)\s*tokens per second")
PROMPT_TPS_RE = re.compile(r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second")


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
     "note": "better decisions; fits an 11 GB card only in fp16 with CUDA graphs off (fp32 is about 17 GB, graph buffers add about 2 GB)"},
    {"id": "jaredpalmer/kev-0.6b", "base": "Qwen3-0.6B-Base", "weights_gib": 1.2, "approx": True,
     "note": "smallest current checkpoint"},
    {"id": "jaredpalmer/kev-8b", "base": "Qwen3-8B", "weights_gib": 16.4, "approx": True,
     "note": "needs a 24 GB card in fp16"},
    {"id": "jaredpalmer/kev-9b", "base": "Qwen3.5-9B-Base", "weights_gib": 19.3, "approx": False,
     "note": "needs a 24 GB card in fp16"},
    {"id": "jaredpalmer/kev-27b", "base": "Qwen3.8-27B", "weights_gib": 51.3, "approx": False,
     "note": "bf16 only, 80 GB-class GPU; not usable on 2080 Ti"},
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
        if not d["api_keys"]:
            raise ValueError("api_keys must contain at least one key")
        if not d["admin_password"]:
            raise ValueError("admin_password must be set")

    def save(self, new):
        d = self.normalize(new)
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


def gpu_info():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
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


def pick_gpus(spec, own=(), exclude=()):
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
        return [i for i in range(8) if i not in exclude][-n:]
    # Least-used first in 512 MiB buckets; ties go to the highest index, since
    # other users' jobs default to GPU 0 and up.
    ranked = sorted(gpus, key=lambda g: (0 if g["index"] in own else (g["mem_used"] or 0) // 512, -g["index"]))
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
            nodes.append({"node": idx, "total_mib": tot, "free_mib": free, "used_mib": tot - free,
                          "cached_mib": m.get("FilePages", 0)})
        except Exception:  # noqa: BLE001 - metrics must never raise
            continue
    return nodes


def proc_stats(pid):
    """RSS and cumulative read bytes of a process; each field None when unreadable."""
    out = {"rss_mib": None, "read_bytes": None}
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
    except (ValueError, IndexError):
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
    return out


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
        self.internal_key = secrets.token_urlsafe(24)
        self.ready_event = threading.Event()
        self.switch_lock = threading.Lock()   # serialises routed profile switches
        self.routed = False                   # the running profile was chosen by the router
        self.kev = None                       # set in main(): its GPUs are off limits to us
        threading.Thread(target=self._idle_loop, daemon=True).start()

    # -- command building
    def build(self, prof):
        exe = os.path.expanduser(prof.get("llama_server") or self.cfg.get("llama_server"))
        kev_gpus = self.kev.gpus if self.kev else []
        gpus = pick_gpus(prof["gpus"], own=self.running_gpus if self.proc else (), exclude=kev_gpus)
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
        gpus = pick_gpus(k["gpus"], own=self.gpus if self.proc else (), exclude=self.llama.running_gpus)
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

    def sample(self):
        pid = None
        try:
            pid = self.sup.proc.pid if self.sup.proc else None
        except Exception:  # noqa: BLE001
            pass
        mem = meminfo(pid)
        ids, busy = self._cpu_busy()
        return {"t": round(time.time(), 2),
                "ram_used": mem["used_mib"], "ram_cached": mem["cached_mib"], "ram_free": mem["free_mib"],
                "nodes": [n["used_mib"] for n in mem["nodes"]],
                "gpus": [[g["index"], g["mem_used"], g["util"]] for g in gpu_info()],
                "cpu": busy, "rss": mem["child"]["rss_mib"], "read_bytes": mem["child"]["read_bytes"]}

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
            })
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
            return self.send_json(cfg.data)
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
        if path == "/api/controller/restart" and method == "POST":
            self.send_json({"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.5), os.kill(os.getpid(), signal.SIGTERM)), daemon=True).start()
            return
        return self.send_json({"error": "not found"}, 404)

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
            c.request(method, self.path, body=body or None, headers=headers)
            r = c.getresponse()
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
            c.close()
        except OSError as e:
            try:
                self.send_json({"error": {"message": f"upstream error: {e}", "type": "upstream"}}, 502)
            except OSError:
                pass
        finally:
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
    srv.sampler = Sampler(sup)
    srv.sampler.start()

    def shutdown(*_):
        log("shutting down")
        sup.stop("controller exit")
        kev.stop("controller exit")
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
