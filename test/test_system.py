#!/usr/bin/env python3
"""System tab backend: cpuinfo / cpulist / NUMA sysfs / nvidia-smi parsing, disk probing, the settings validators and
token handling, plus a fake-DOM render of the System pane. Needs no GPU and no real /proc or /sys layout."""
import json, os, shutil, stat, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)
def raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except ValueError as e:
        return str(e)
    return None

T = tempfile.mkdtemp(prefix="bizensys.")

# ---- parsers
CPUINFO = "processor\t: 0\nmodel name\t: Intel(R) Xeon(R)  Silver 4116 CPU @ 2.10GHz\nphysical id\t: 0\n\nprocessor\t: 1\nmodel name\t: Intel(R) Xeon(R)  Silver 4116 CPU @ 2.10GHz\nphysical id\t: 1\n"
ci = b.parse_cpuinfo(CPUINFO)
check("cpuinfo model/threads/sockets", ci == {"model": "Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz", "threads": 2, "sockets": 2})
check("cpuinfo without model name or physical id -> None", b.parse_cpuinfo("processor : 0\n") == {"model": None, "threads": 1, "sockets": None})
check("cpuinfo empty/garbage", b.parse_cpuinfo("") == {"model": None, "threads": None, "sockets": None} and b.parse_cpuinfo(None)["model"] is None)
check("cpulist ranges", b.parse_cpulist("0,8-9,13,18") == [0, 8, 9, 13, 18] and b.parse_cpulist("") == [] and b.parse_cpulist("x,3-,4") == [4])
check("os-release", b.os_pretty_name('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 22.04.4 LTS"\n') == "Ubuntu 22.04.4 LTS" and b.os_pretty_name("") is None)

# fake sysfs NUMA tree
root = os.path.join(T, "node")
for i, (cl, mem) in enumerate([("0-3,8", 131072), ("4-7", 65536)]):
    d = os.path.join(root, f"node{i}"); os.makedirs(d)
    open(os.path.join(d, "cpulist"), "w").write(cl + "\n")
    open(os.path.join(d, "meminfo"), "w").write(f"Node {i} MemTotal:       {mem * 1024} kB\nNode {i} MemFree:        1 kB\n")
os.makedirs(os.path.join(root, "nodeX")); os.makedirs(os.path.join(root, "node2"))     # junk dirs must not raise
nn = b.numa_nodes(root)
check("numa nodes from fake sysfs", [(n["id"], n["cpus"], n["mem_total_mib"]) for n in nn[:2]] == [(0, [0, 1, 2, 3, 8], 131072), (1, [4, 5, 6, 7], 65536)])
check("numa: node without files tolerated; missing root -> []", len(nn) == 3 and nn[2]["cpus"] == [] and b.numa_nodes(os.path.join(T, "nope")) == [])

SMI = "0, NVIDIA GeForce RTX 2080 Ti, 11264, 535.183.01, 7.5\n1, NVIDIA GeForce RTX 2080 Ti, 11264, 535.183.01, 7.5\nbad line\n"
g = b.parse_smi_gpus(SMI)
check("smi gpus parsed", len(g) == 2 and g[1] == {"index": 1, "name": "NVIDIA GeForce RTX 2080 Ti", "vram_mib": 11264, "driver": "535.183.01", "compute_cap": "7.5"})
check("smi without compute_cap column / N/A", b.parse_smi_gpus("0, X, 8192, 550.1\n")[0]["compute_cap"] is None
      and b.parse_smi_gpus("0, X, [N/A], 550.1, [N/A]\n")[0]["vram_mib"] is None and b.parse_smi_gpus("", True) == [] and b.parse_smi_gpus(None) == [])
check("cuda version from nvidia-smi header", b.smi_cuda_version("| NVIDIA-SMI 535.183.01   Driver Version: 535.183.01   CUDA Version: 12.2     |") == "12.2" and b.smi_cuda_version("") is None)

# detect_gpus with a fake nvidia-smi on PATH (also: field rejected by an old driver, and a hanging smi)
bindir = os.path.join(T, "bin"); os.makedirs(bindir)
def fake(name, body):
    p = os.path.join(bindir, name); open(p, "w").write("#!/bin/sh\n" + body); os.chmod(p, 0o755); return p
fake("nvidia-smi", '''case "$*" in *compute_cap*) echo "0, RTX Fake, 24576, 550.54, 8.9";; *--query-gpu*) echo "0, RTX Fake, 24576, 550.54";; *) echo "CUDA Version: 12.4";; esac''')
oldpath = os.environ["PATH"]; os.environ["PATH"] = bindir + ":" + oldpath
d = b.detect_gpus()
check("detect_gpus via fake nvidia-smi", d == [{"index": 0, "name": "RTX Fake", "vram_mib": 24576, "driver": "550.54", "compute_cap": "8.9", "cuda": "12.4"}])
fake("nvidia-smi", '''case "$*" in *compute_cap*) echo "Field compute_cap is not a valid field" >&2; exit 2;; *--query-gpu*) echo "0, Old, 8192, 440.1";; *) echo "no header";; esac''')
d = b.detect_gpus()
check("detect_gpus falls back when compute_cap is rejected", d == [{"index": 0, "name": "Old", "vram_mib": 8192, "driver": "440.1", "compute_cap": None}])
fake("nvidia-smi", "sleep 30")
b.PROBE_TIMEOUT = 1
orig_run = b._run_probe
b._run_probe = lambda cmd, timeout=1: orig_run(cmd, timeout)
import time; t0 = time.time(); d = b.detect_gpus(); dt = time.time() - t0
check("hanging nvidia-smi times out, no raise", d == [] and dt < 8)
b._run_probe = orig_run
os.remove(os.path.join(bindir, "nvidia-smi"))
check("no nvidia-smi -> []", b.detect_gpus() == [] or shutil.which("nvidia-smi") is not None)

# ---- disks
du = b.disk_usage(T)
check("disk usage of an existing dir", du["exists"] and du["writable"] and du["free_gib"] is not None and du["total_gib"] >= du["free_gib"])
du = b.disk_usage(os.path.join(T, "not", "yet"))
check("disk usage of a missing path uses the ancestor", not du["exists"] and du["free_gib"] is not None and du["writable"])
check("disk usage garbage -> unknown, no raise", b.disk_usage("\x00")["free_gib"] is None and b.disk_usage(None)["free_gib"] is None)

# ---- tools: version cache + failure modes
ls = fake("llama-server", 'echo "version: 4321 (abc123)" >&2\necho "built with gcc" >&2')
b._VERSION_CACHE.clear()
v1 = b.tool_version(ls, now=1000.0)
open(ls, "w").write("#!/bin/sh\necho 'version: 9999'\n")
v2 = b.tool_version(ls, now=1030.0)
v3 = b.tool_version(ls, now=1061.0)
check("version parsed from stderr, cached 60 s, refreshed after", v1 == "version: 4321 (abc123)" and v2 == v1 and v3 == "version: 9999")
bad = fake("bad-server", "exit 3")
slow = fake("slow-server", "sleep 30")
t0 = time.time()
check("failing version -> None", b.tool_version(bad, now=5000.0) is None)
check("hanging --version times out", b.tool_version(slow, timeout=1, now=5000.0) is None and time.time() - t0 < 8)
check("missing binary -> None", b.tool_version(os.path.join(T, "nope"), now=5000.0) is None)

# ---- system_info shape on a config with none of the new keys
cfg = b.Config.normalize({"profiles": [], "model_dirs": [T], "llama_server": ls})
si = b.system_info(cfg)
check("system_info top-level shape", all(k in si for k in ("os", "kernel", "python", "hostname", "cpu", "ram", "gpus", "disks", "tools", "settings")))
check("cpu block shape", all(k in si["cpu"] for k in ("model", "sockets", "physical_cores", "threads", "allowed_cpus", "numa_nodes", "pinned_default")))
check("ram block shape", all(k in si["ram"] for k in ("total_mib", "cgroup_limit_mib", "nodes")))
roles = [x["role"] for x in si["disks"]]
check("disks: download defaults to first model dir, plus model_dir and hf_home", roles == ["download", "model_dir", "hf_home"] and si["disks"][0]["path"] == T)
check("tools block", set(si["tools"]) >= {"llama_server", "numactl", "taskset", "nvidia_smi", "hf_cli"} and si["tools"]["llama_server"]["executable"]
      and isinstance(si["tools"]["hf_cli"], bool))
check("settings: token only as has_hf_token", si["settings"]["has_hf_token"] is False and "\"hf_token\"" not in json.dumps(si))
check("defaults of the new keys", cfg["download_dir"] == "" and cfg["hf_token"] == "" and cfg["hf_endpoint"] == "https://huggingface.co"
      and cfg["min_free_gib_for_download"] == 20 and cfg["gpu_bucket_mib"] == 512 and cfg["gpu_fallback_count"] == 8)
cfg2 = b.Config.normalize({"profiles": [], "llama_server": "/nonexistent/llama-server", "model_dirs": []})
si2 = b.system_info(cfg2)
check("nothing found: unknown, not an exception", si2["tools"]["llama_server"]["exists"] is False and si2["tools"]["llama_server"]["version"] is None and si2["disks"][-1]["role"] == "hf_home")
orig_cpuinfo = b.read
b.read = lambda p, d="": (_ for _ in ()).throw(OSError("boom")) if p == "/proc/cpuinfo" else orig_cpuinfo(p, d)
try:
    si3 = b.system_info(cfg)
except Exception as e:  # noqa: BLE001
    si3 = None
check("a failing reader does not raise", si3 is not None)
b.read = orig_cpuinfo

# ---- settings validators
def cfgfile(extra=None):
    raw = {"api_keys": ["k"], "admin_password": "p", "active_profile": "a", "model_dirs": [T], "llama_server": ls,
           "profiles": [{"name": "a", "model": "/m.gguf"}]}
    raw.update(extra or {})
    path = os.path.join(T, "config.json"); json.dump(raw, open(path, "w")); os.chmod(path, 0o600)
    return b.Config(path)
c = cfgfile()
V = lambda upd, **kw: b.validate_settings(upd, c.data, **kw)
check("relative download_dir rejected", "absolute" in (raises(V, {"download_dir": "rel/dir"}) or ""))
check("relative model_dirs entry rejected", "absolute" in (raises(V, {"model_dirs": ["/ok", "rel"]}) or ""))
check("empty model_dirs rejected", raises(V, {"model_dirs": []}) is not None)
check("relative llama_server rejected", "absolute" in (raises(V, {"llama_server": "llama-server"}) or ""))
check("llama_server missing rejected", "not a file" in (raises(V, {"llama_server": os.path.join(T, "nope")}) or ""))
nx = os.path.join(T, "plain"); open(nx, "w").write("x")
check("llama_server not executable rejected", "not executable" in (raises(V, {"llama_server": nx}) or ""))
check("llama_server directory rejected", raises(V, {"llama_server": T}) is not None)
ch, w, ck = V({"llama_server": bad})
check("llama_server whose --version fails: warning, not failure", ch["llama_server"] == bad and any("--version" in x for x in w) and ck["llama_server_version"] is None)
ch, w, ck = V({"llama_server": ls})
check("llama_server ok reports version", w == [] and ck["llama_server_version"] == "version: 9999")
check("hf_endpoint must be http(s)", raises(V, {"hf_endpoint": "ftp://x"}) and raises(V, {"hf_endpoint": "huggingface.co"}) and raises(V, {"hf_endpoint": "http://"})
      and V({"hf_endpoint": "https://hf-mirror.example/"})[0]["hf_endpoint"] == "https://hf-mirror.example" and V({"hf_endpoint": ""})[0]["hf_endpoint"] == "https://huggingface.co")
check("min_free_gib validated", raises(V, {"min_free_gib_for_download": -1}) and raises(V, {"min_free_gib_for_download": "abc"}) and V({"min_free_gib_for_download": "30"})[0]["min_free_gib_for_download"] == 30)
check("token with whitespace rejected, error does not echo it", "secret" not in (raises(V, {"hf_token": "hf_secret token"}) or "x") and raises(V, {"hf_token": "a\nb"}))
check("token and clear together rejected", raises(V, {"hf_token": "hf_x", "clear_hf_token": True}) is not None)
new = os.path.join(T, "dl", "deep")
ch, w, ck = V({"download_dir": new}, create=False)
check("dry run: creatable download_dir passes without creating it", ch["download_dir"] == new and not os.path.exists(new))
ch, w, ck = V({"download_dir": new}, create=True)
check("real run: download_dir created and proven writable", os.path.isdir(new) and not [f for f in os.listdir(new) if f.startswith(".bizen-write-test")])
ro = os.path.join(T, "ro"); os.makedirs(ro); os.chmod(ro, 0o500)
if os.geteuid() != 0:
    check("read-only download_dir rejected", "not writable" in (raises(V, {"download_dir": ro}) or "") and raises(V, {"download_dir": os.path.join(ro, "sub")}, create=True) is not None
          and raises(V, {"download_dir": os.path.join(ro, "sub")}, create=False) is not None)
else:
    print("SKIP read-only checks (running as root)")
os.chmod(ro, 0o700)
check("download_dir that is a file rejected", "not a directory" in (raises(V, {"download_dir": nx}) or ""))
check("empty download_dir = default", V({"download_dir": ""})[0]["download_dir"] == "")
ch, w, ck = V({"model_dirs": [T, os.path.join(T, "gone"), T]})
check("model_dirs deduped, missing dir is a warning", ch["model_dirs"] == [T, os.path.join(T, "gone")] and len(w) == 1)

# ---- apply_settings + token handling through Config.save
SECRET = "hf_SECRET_abc123"
r = b.apply_settings(c, {"hf_token": SECRET, "hf_endpoint": "https://hub.example", "download_dir": new, "min_free_gib_for_download": 5, "hf_home": os.path.join(T, "hfh")})
disk = json.load(open(c.path))
check("saved to disk through Config.save", disk["hf_token"] == SECRET and disk["hf_endpoint"] == "https://hub.example" and disk["download_dir"] == new
      and disk["min_free_gib_for_download"] == 5 and disk["kev"]["hf_home"] == os.path.join(T, "hfh") and r["saved"])
check("response carries has_hf_token only", r["settings"]["has_hf_token"] is True and SECRET not in json.dumps(r))
check("config file stays mode 600", stat.S_IMODE(os.stat(c.path).st_mode) == 0o600)
check("public_config hides the token", "hf_token" not in b.public_config(c.data) and b.public_config(c.data)["has_hf_token"] is True and SECRET not in json.dumps(b.public_config(c.data)))
check("system_info hides the token", SECRET not in json.dumps(b.system_info(c.data)) and b.system_info(c.data)["settings"]["has_hf_token"] is True)
b.apply_settings(c, {"hf_endpoint": "https://hub2.example", "hf_token": ""})
check("empty token keeps the saved one", json.load(open(c.path))["hf_token"] == SECRET)
b.apply_settings(c, {"min_free_gib_for_download": 7})
check("omitted token keeps the saved one", json.load(open(c.path))["hf_token"] == SECRET and c.data["hf_endpoint"] == "https://hub2.example")
pc = b.public_config(c.data); pc["profiles"][0]["ctx"] = 4096          # what the UI does on "Save": GET /api/config -> PUT
c.save(pc)
check("PUT /api/config round trip keeps the token and does not persist has_hf_token", json.load(open(c.path))["hf_token"] == SECRET and "has_hf_token" not in json.load(open(c.path)))
b.apply_settings(c, {"hf_token": "hf_new"}, dry_run=True)
check("dry run saves nothing", json.load(open(c.path))["hf_token"] == SECRET)
b.apply_settings(c, {"clear_hf_token": True})
check("explicit clear removes it", json.load(open(c.path))["hf_token"] == "" and c.data["hf_token"] == "" and b.public_config(c.data)["has_hf_token"] is False)
err = raises(b.apply_settings, c, {"hf_token": "hf_leak_me", "download_dir": "relative"})
check("error message never contains the token", err is not None and "hf_leak_me" not in err)
check("old config.json without new keys behaves as before", cfgfile().get("download_dir") == "" and b.effective_download_dir(cfgfile()) == T and cfgfile().get("min_free_gib_for_download") == 20)

# changing llama_server / model_dirs must not touch a running model: Supervisor reads config per start, status shows no restart flag
class P:
    pid = 1
    def poll(self): return None
c = cfgfile()
s = b.Supervisor(c)
s.proc, s.state, s.running_profile = P(), "running", c.profile("a")
s.started_at = time.time()
before = s.status()
b.apply_settings(c, {"llama_server": bad, "model_dirs": [T, new]})
after = s.status()
check("llama_server/model_dirs change: running model untouched, no restart flagged", after["state"] == "running" and after["pid"] == 1 and after["restart_needed"] is False and before["restart_needed"] is False)
s.state, s.proc = "stopped", None

# config tuning keys
b_gpu = b.gpu_info; b.gpu_info = lambda *a, **k: [{"index": i, "mem_used": u, "mem_total": 1, "util": 0, "temp": 0} for i, u in enumerate([100, 700, 900])]
check("512 MiB buckets by default: gpu0 (100) and gpu1 (700) land in different buckets", b.pick_gpus("auto:1") == [0])
check("bigger bucket: ties go to the highest index", b.pick_gpus("auto:1", bucket=2048) == [2])
b.gpu_info = lambda *a, **k: []
check("no nvidia-smi: fallback count decides the indices", b.pick_gpus("auto:2") == [6, 7] and b.pick_gpus("auto:2", fallback=4) == [2, 3])
b.gpu_info = b_gpu
check("bad tuning values rejected", all(raises(b.Config.validate, dict(c.data, **{k: v})) for k, v in (("gpu_bucket_mib", 0), ("gpu_fallback_count", "x"), ("download_mbps", -1))))

# ---- UI: fake-DOM render of the System pane and the Kev hint card (node)
UI = os.path.join(HERE, "..", "ui.html")
if shutil.which("node"):
    html = open(UI).read()
    js = html[html.index("<script>") + 8:html.rindex("</script>")]
    a = js.index("function kevCard()"); e = js.index("async function saveRouter")
    kev = js[a:e]
    a2 = js.index("let sysDirs = []"); e2 = js.index("/* ---- tabs: selected tab")
    sysjs = js[a2:e2]
    harness = r'''
const els = {}; const $ = id => els[id] || (els[id] = { id, value: '', textContent: '', innerHTML: '', style: {}, disabled: false, addEventListener() {} });
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const gib = m => m == null ? '–' : (m / 1024).toFixed(m < 10240 ? 1 : 0);
const tabHooks = {}; let cfg = { download_mbps: 11.7, kev: { gpus: 'auto:1' } }, status = null, sysInfo = null; let kevPresets = [];
const document = { querySelector: s => { const k = s.match(/data-kev="(\w+)"/)[1]; return $('kev_' + k); } };
const toast = () => {}; const api = async () => ({}); const loadConfig = async () => {}; const loadModels = () => {}; const confirm = () => true;
''' + kev + sysjs + r'''
const out = {};
// 1) full detection, mixed GPUs: Kev hint must use the Kev GPU (index 1, 24 GiB, cc 8.9), not the smallest card overall
kevPresets = [{ id: 'k4', base: 'Base-4B', weights_gib: 8.4, approx: false, note: 'n' }];
$('kev_model').value = 'k4'; $('kev_dtype').value = 'fp32'; $('kev_cuda_graphs').value = 'auto';
status = { gpus: [{ index: 0, name: 'Small', mem_total: 8192, mem_used: 100 }, { index: 1, name: 'Big', mem_total: 24576, mem_used: 0 }], server: { gpus: [0] }, kev: { gpus: [] } };
sysInfo = { gpus: [{ index: 0, compute_cap: '7.5' }, { index: 1, compute_cap: '8.9' }] };
cfg.kev.gpus = 'auto:1'; kevHint(); out.big = $('kevHint').textContent;
cfg.kev.gpus = '0'; kevHint(); out.small = $('kevHint').textContent;
status.gpus = []; cfg.kev.gpus = 'auto:1'; kevHint(); out.none = $('kevHint').textContent;
// 2) System pane with nothing known
sysInfo = { os: null, cpu: {}, ram: {}, gpus: [], disks: [], tools: {}, settings: {} };
renderSystem(); out.empty = [$('sysHost').textContent, $('sysCpu').innerHTML, $('sysGpus').innerHTML, $('sysTools').innerHTML];
// 3) populated
sysInfo = { hostname: 'h', os: 'Ubuntu', kernel: '6', python: '3', ts: 1, cpu: { model: 'Xeon', sockets: 2, physical_cores: 8, threads: 12, allowed_cpus: [0, 1], numa_nodes: [{ id: 0, cpus: [0, 1], mem_total_mib: 1000 }], pinned_default: { cpus: [0], node: 0 } },
  ram: { total_mib: 2000, cgroup_limit_mib: null, nodes: [{ node: 0, total_mib: 1000, used_mib: 500, free_mib: 500 }] },
  gpus: [{ index: 1, name: '<b>G</b>', vram_mib: 11264, driver: '535', compute_cap: '7.5', cuda: '12.2' }],
  disks: [{ path: '/srv/fs/models', role: 'download', free_gib: 10, total_gib: 100, writable: true, exists: true }, { path: '/x', role: 'hf_home', free_gib: null, total_gib: null, writable: null, exists: false }],
  tools: { llama_server: { path: '/ls', exists: true, executable: true, version: 'v1' }, numactl: '/usr/bin/numactl', taskset: null, nvidia_smi: null, hf_cli: false },
  settings: { download_dir: '', effective_download_dir: '/srv/fs/models', model_dirs: ['/a', '/b'], hf_home: '', llama_server: '/ls', hf_endpoint: 'https://huggingface.co', has_hf_token: true, min_free_gib_for_download: 20 } };
status = { gpus: [{ index: 1, mem_used: 2048, mem_total: 11264 }], server: { gpus: [1] } };
renderSystem(); out.full = [$('sysCpu').innerHTML, $('sysGpus').innerHTML, $('sysDisks').innerHTML, $('sysTools').innerHTML, $('sDirs').innerHTML, $('sTokPill').textContent, $('sTokClear').disabled, $('sMin').value];
sysDirs = ['/a']; out.payload = JSON.stringify(sysPayload()); $('sTok').value = 'hf_typed'; out.payload2 = JSON.stringify(sysPayload());
console.log(JSON.stringify(out));
'''
    for k in ("kev_model", "kev_dtype", "kev_cuda_graphs"):
        pass
    hp = os.path.join(T, "harness.js"); open(hp, "w").write(harness)
    r = subprocess.run(["node", hp], capture_output=True, text=True, timeout=60)
    try:
        o = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        o = None
        print(r.stderr[-600:])
    check("ui: render does not throw", o is not None)
    if o:
        check("kev hint uses the Kev GPU's VRAM (24 GiB, not the 8 GiB card)", "24 GiB card" in o["big"] and "Fits" in o["big"] and "Big" in o["big"])
        check("kev hint on the small explicit GPU: does not fit, bf16 note from compute cap", "8 GiB card" in o["small"] and "NOT" in o["small"])
        check("kev hint without GPU data: no fit check, no invented card size", "unknown" in o["none"] and "GiB card" not in o["none"] and "11 GiB" not in o["none"])
        check("ui: unknown state is calm", "–" in o["empty"][0] and "No NVIDIA GPU detected" in o["empty"][2] and "not found" in o["empty"][3])
        full = o["full"]
        check("ui: cpu/ram/gpu/disk/tool cards", "Xeon" in full[0] and "NUMA node 0" in full[0] and "GPU 1" in full[1] and "compute 7.5" in full[1] and "CUDA 12.2" in full[1]
              and "10 GiB free" in full[2] and "missing" in full[2] and "v1" in full[3])
        check("ui: gpu name is escaped", "&lt;b>G" in full[1] and "<b>G</b>" not in full[1])
        check("ui: token indicator, no token value", full[5] == "token set" and full[6] is False and "hf_typed" not in json.dumps(full))
        check("ui: save payload omits the token unless typed; model dir list edited", "hf_token" not in o["payload"] and json.loads(o["payload"])["model_dirs"] == ["/a"] and json.loads(o["payload2"])["hf_token"] == "hf_typed")
else:
    print("SKIP ui render (node missing)")

shutil.rmtree(T, ignore_errors=True)
print("FAILED: " + ", ".join(fails) if fails else "ALL system checks passed")
sys.exit(1 if fails else 0)
