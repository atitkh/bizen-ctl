#!/usr/bin/env python3
"""Library tab backend: shard grouping, mmproj attach, quant parsing, symlink containment, used_by / loaded refusal,
delete confinement, draft profiles, fit arithmetic, plus a fake-DOM render of the Library pane. No GPU needed."""
import io, json, os, shutil, subprocess, sys, tempfile, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

T = tempfile.mkdtemp(prefix="bizenlib.")
GIB = 2**30

def w(path, n=100):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * n)

class Cfg:
    def __init__(self, d): self.data = d
    def get(self, k): return self.data.get(k)

class Sup:
    def __init__(self, model=None): self.running_profile = {"model": model} if model else None
    def serving(self): return "p" if self.running_profile else None

b.live_gpu_mems = lambda: [24576, 24576]
b._real_meminfo = b.meminfo
b.meminfo = lambda pid=None: {"total_mib": 65536}

def mk(model_dirs, profiles=(), download_dir=""):
    return Cfg({"model_dirs": list(model_dirs), "download_dir": download_dir,
                "profiles": [dict(b.PROFILE_DEFAULTS, **p) for p in profiles]})

def by_name(scan, n):
    return next((m for m in scan["models"] if m["name"] == n), None)

# ---- quant parsing
for name, q in [("Qwen3-235B-A22B-UD-Q3_K_XL", "UD-Q3_K_XL"), ("llama-Q4_K_M", "Q4_K_M"), ("m-IQ2_XXS", "IQ2_XXS"),
                ("Foo-7B-BF16", "BF16"), ("Foo.F16", "F16"), ("plain-model", None), ("Llama-3-8B-Instruct", None),
                ("gpt-oss-120b-MXFP4", "MXFP4"), ("x-q8_0", "Q8_0")]:
    check(f"quant {name} -> {q}", b.parse_quant(name) == q)

# ---- grouping, mmproj, mixed quants, partials, symlinks
M = os.path.join(T, "models"); DL = os.path.join(T, "dl"); OUT = os.path.join(T, "outside")
for i in (1, 2, 3):
    w(f"{M}/qwen/Qwen-UD-Q3_K_XL-0000{i}-of-00003.gguf", 1000)           # complete set
for i in (1, 3):
    w(f"{M}/qwen/Qwen-Q4_K_M-0000{i}-of-00003.gguf", 500)               # same dir, other quant, shard 2 missing
w(f"{M}/qwen/Qwen-Q4_K_M-00002-of-00003.gguf.part", 7)
w(f"{M}/qwen/mmproj-F16.gguf", 5); w(f"{M}/qwen/notes.txt", 9)
w(f"{M}/single-IQ2_XXS.gguf", 300)
w(f"{M}/half.gguf.part", 3)
w(f"{DL}/new/Dl-Q8_0.gguf", 200)
w(f"{OUT}/secret.gguf", 50)
os.symlink(f"{OUT}/secret.gguf", f"{M}/link.gguf")                       # file symlink out of the model dir
os.symlink(OUT, f"{M}/linkdir")                                           # directory symlink out of the model dir
os.symlink(f"{M}/single-IQ2_XXS.gguf", f"{M}/alias.gguf")                # symlink inside: same file, listed once
cfg = mk([M, M + "/", os.path.join(M, "qwen", "..")], download_dir=DL)
s = b.scan_library(cfg.data, [24576], 65536)
names = sorted(m["name"] for m in s["models"])
check("entries: 2 sets in one dir + single + download dir + partial, no mmproj / symlink escapes",
      names == ["Dl-Q8_0", "Qwen-Q4_K_M", "Qwen-UD-Q3_K_XL", "half", "single-IQ2_XXS"], )
check("roots deduped (model_dirs spelled 3 ways + download_dir)", len(s["roots"]) == 2)
q3 = by_name(s, "Qwen-UD-Q3_K_XL"); q4 = by_name(s, "Qwen-Q4_K_M")
check("complete shard set is ONE entry with 3 shards, first_shard is shard 1",
      q3["shards_found"] == 3 and q3["shards_expected"] == 3 and q3["complete"] and q3["bytes"] == 3000
      and q3["first_shard"].endswith("-00001-of-00003.gguf") and [x["bytes"] for x in q3["shards"]] == [1000] * 3)
check("quant parsed per set; two quants in one dir stay separate", q3["quant"] == "UD-Q3_K_XL" and q4["quant"] == "Q4_K_M" and q3["id"] != q4["id"])
check("missing shard: complete false, 2/3, hint names the partial shard", not q4["complete"] and q4["shards_found"] == 2 and q4["shards_expected"] == 3
      and "shard 2" in q4["hint"] and "partial" in q4["hint"] and q4["fit"]["verdict"] is None)
check("mmproj attached to models in its folder, not listed, absent elsewhere",
      q3["mmproj"].endswith("mmproj-F16.gguf") and q4["mmproj"] and by_name(s, "single-IQ2_XXS")["mmproj"] is None)
check("size accounting: gib rounded, mtime present", q3["gib"] == round(3000 / GIB, 2) and q3["mtime"] > 0)
hp = by_name(s, "half")
check("loose .part file: listed incomplete with a hint, not usable", hp["partial"] and not hp["complete"] and "Partial download" in hp["hint"])
check("symlink escape (file and dir) ignored, in-root symlink not double listed",
      not any("secret" in x["path"] or "link" in x["path"] or "alias" in x["path"] for m in s["models"] for x in m["shards"]))
check("sorted by mtime desc", [m["mtime"] for m in s["models"]] == sorted((m["mtime"] for m in s["models"]), reverse=True))
check("id is stable across scans", by_name(b.scan_library(cfg.data), "Qwen-UD-Q3_K_XL")["id"] == q3["id"])
check("never raises: missing / empty config", b.scan_library({})["models"] == [] and b.scan_library({"model_dirs": [T + "/nope"], "download_dir": None})["models"] == [])
if os.geteuid() != 0:
    os.makedirs(f"{T}/locked"); w(f"{T}/locked/x.gguf"); os.chmod(f"{T}/locked", 0)
    try:
        check("never raises: unreadable directory", isinstance(b.scan_library({"model_dirs": [T + "/locked"]})["models"], list))
    finally:
        os.chmod(f"{T}/locked", 0o755)

# ---- used_by / loaded
prof = [{"name": "main", "model": q3["first_shard"]}, {"name": "second-shard", "model": q3["shards"][1]["path"]}, {"name": "other", "model": M + "/nothing.gguf"}]
s2 = b.scan_library(mk([M], prof).data, [24576], 65536, loaded_model=q3["first_shard"])
e3 = by_name(s2, "Qwen-UD-Q3_K_XL")
check("used_by: profile on the first shard or any shard; loaded flag", e3["used_by"] == ["main", "second-shard"] and e3["loaded"] and not by_name(s2, "Qwen-Q4_K_M")["loaded"])
check("unrelated entries have empty used_by", by_name(s2, "single-IQ2_XXS")["used_by"] == [])

# ---- fit arithmetic
fv = b.fit_verdict
check("fit: exactly 90% of VRAM fits, one MiB more needs RAM",
      fv(int(0.9 * 24576) * 2**20, 24576, 65536)["verdict"] == "fits-vram" and fv((int(0.9 * 24576) + 1) * 2**20, 24576, 65536)["verdict"] == "needs-ram")
check("fit: bigger than 90% of VRAM + 90% of RAM is too big", fv(100 * GIB, 24576, 65536)["verdict"] == "too-big" and fv(75 * GIB, 24576, 65536)["verdict"] == "needs-ram")
check("fit: no GPU data -> null verdict, calm text", fv(5 * GIB, None, 65536)["verdict"] is None and "estimate" in fv(5 * GIB, 0, None)["text"])
check("fit: text says it is a rough estimate, not a speed prediction", "Rough size-based estimate, not a speed prediction" in fv(5 * GIB, 24576, 1000)["text"])
check("fit: vram is the SUM over GPUs", b.scan_library({"model_dirs": [M]}, [24576, 24576], 1024)["models"][0]["fit"]["vram_total_mib"] == 49152)
check("choose_gpus: > half of total VRAM -> all GPUs", b.choose_gpus(30 * GIB, [24576, 24576]) == "auto:2" and b.choose_gpus(40 * GIB, [24576] * 4) == "auto:2" and b.choose_gpus(50 * GIB, [24576] * 4) == "auto:4")
check("choose_gpus: small -> 1, fits no single card but under half -> 2",
      b.choose_gpus(10 * GIB, [24576] * 2) == "auto:1" and b.choose_gpus(22 * GIB, [24576] * 3) == "auto:2" and b.choose_gpus(5 * GIB, [24576]) == "auto:1")
check("choose_gpus: no GPU data -> profile default", b.choose_gpus(5 * GIB, []) == b.PROFILE_DEFAULTS["gpus"])

# ---- delete: refusals
D = os.path.join(T, "del"); other = os.path.join(T, "keep")
for i in (1, 2):
    w(f"{D}/Sub/Set-Q4_K_M-0000{i}-of-00002.gguf", 400)
w(f"{D}/Sub/mmproj-F16.gguf", 10); w(f"{D}/Sub/README.txt", 4); w(f"{D}/Sub/Set-Q4_K_M.json", 4)
w(f"{D}/Sub2/Other-Q8_0.gguf", 20); w(f"{D}/Sub2/mmproj-F16.gguf", 10)
w(f"{D}/Root-Q2_K.gguf", 60)
dcfg = mk([D], [{"name": "use", "model": D + "/Sub2/Other-Q8_0.gguf"}])
sc = b.scan_library(dcfg.data)
eid = lambda n: by_name(sc, n)["id"]
check("delete: no confirm -> 400", b.library_delete(dcfg, Sup(), eid("Set-Q4_K_M"), None)[0] == 400 and b.library_delete(dcfg, Sup(), eid("Set-Q4_K_M"), "delete")[0] == 400
      and b.library_delete(dcfg, Sup(), eid("Set-Q4_K_M"), True)[0] == 400)
check("delete: nothing removed without confirm", os.path.exists(f"{D}/Sub/Set-Q4_K_M-00001-of-00002.gguf"))
check("delete: unknown id 404; path-like id never touches disk", b.library_delete(dcfg, Sup(), "zzz", "DELETE")[0] == 404
      and b.library_delete(dcfg, Sup(), f"{D}/Root-Q2_K.gguf", "DELETE")[0] == 404 and b.library_delete(dcfg, Sup(), None, "DELETE")[0] == 400
      and os.path.exists(f"{D}/Root-Q2_K.gguf"))
code, body = b.library_delete(dcfg, Sup(), eid("Other-Q8_0"), "DELETE")
check("delete: used by a profile -> 409 naming it, file kept", code == 409 and "use" in body["error"] and body["used_by"] == ["use"] and os.path.exists(f"{D}/Sub2/Other-Q8_0.gguf"))
code, body = b.library_delete(mk([D]), Sup(D + "/Root-Q2_K.gguf"), eid("Root-Q2_K"), "DELETE")
check("delete: loaded -> 409", code == 409 and "loaded" in body["error"] and os.path.exists(f"{D}/Root-Q2_K.gguf"))

# ---- delete: success, confinement
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code, body = b.library_delete(dcfg, Sup(), eid("Set-Q4_K_M"), "DELETE")
left = sorted(os.listdir(f"{D}/Sub"))
check("delete: ok, frees the shard bytes + the sole model's mmproj", code == 200 and body["freed_bytes"] == 810 and body["ok"] and len(body["removed"]) == 3)
check("delete: only gguf files of the set removed, README and sidecar json left, folder kept", left == ["README.txt", "Set-Q4_K_M.json"] and not body["dir_removed"])
check("delete: one log line with name and bytes", buf.getvalue().count("library: deleted Set-Q4_K_M") == 1 and "810 bytes" in buf.getvalue() and buf.getvalue().count("\n") == 1)
check("delete: other folders / models untouched", os.path.exists(f"{D}/Sub2/Other-Q8_0.gguf") and os.path.exists(f"{D}/Sub2/mmproj-F16.gguf") and os.path.exists(f"{D}/Root-Q2_K.gguf"))

# mmproj shared with another model in the folder stays
w(f"{D}/Two/A-Q4_K_M.gguf", 30); w(f"{D}/Two/B-Q8_0.gguf", 40); w(f"{D}/Two/mmproj-F16.gguf", 5)
sc = b.scan_library(dcfg.data)
with contextlib.redirect_stdout(io.StringIO()):
    code, body = b.library_delete(dcfg, Sup(), eid("A-Q4_K_M"), "DELETE")
check("delete: mmproj kept while another model lives in the folder", code == 200 and body["freed_bytes"] == 30 and sorted(os.listdir(f"{D}/Two")) == ["B-Q8_0.gguf", "mmproj-F16.gguf"])
# mmproj referenced by a profile's extra_args stays
w(f"{D}/Three/Solo-Q4_K_M.gguf", 30); w(f"{D}/Three/mmproj-F16.gguf", 5)
pc = mk([D], [{"name": "v", "model": D + "/x.gguf", "extra_args": f"-fa on --mmproj {D}/Three/mmproj-F16.gguf"}])
sc = b.scan_library(pc.data)
with contextlib.redirect_stdout(io.StringIO()):
    code, body = b.library_delete(pc, Sup(), eid("Solo-Q4_K_M"), "DELETE")
check("delete: mmproj named in a profile's extra_args is kept", code == 200 and os.listdir(f"{D}/Three") == ["mmproj-F16.gguf"])

# empty folder cleanup: strict subdirectory only, never a model dir
w(f"{D}/Gone/Only-Q4_K_M.gguf", 11)
sc = b.scan_library(dcfg.data)
with contextlib.redirect_stdout(io.StringIO()):
    code, body = b.library_delete(dcfg, Sup(), eid("Only-Q4_K_M"), "DELETE")
check("delete: emptied subfolder is removed", code == 200 and body["dir_removed"] and not os.path.exists(f"{D}/Gone"))
sc = b.scan_library(dcfg.data)
with contextlib.redirect_stdout(io.StringIO()):
    code, body = b.library_delete(dcfg, Sup(), eid("Root-Q2_K"), "DELETE")
check("delete: model at the top of a model dir leaves that dir (even if it is now empty of models)", code == 200 and not body["dir_removed"] and os.path.isdir(D))
R = os.path.join(T, "solo"); w(f"{R}/Lone-Q4_K_M.gguf", 9)
rc = mk([R]); sc = b.scan_library(rc.data)
with contextlib.redirect_stdout(io.StringIO()):
    code, body = b.library_delete(rc, Sup(), eid("Lone-Q4_K_M"), "DELETE")
check("delete: emptied model dir itself is never removed", code == 200 and os.path.isdir(R) and os.listdir(R) == [])
# download dir counts as a root too
w(f"{DL}/only/Dl2-Q4_K_M.gguf", 9)
dc = mk([M], download_dir=DL); sc = b.scan_library(dc.data)
with contextlib.redirect_stdout(io.StringIO()):
    b.library_delete(dc, Sup(), eid("Dl2-Q4_K_M"), "DELETE"); b.library_delete(dc, Sup(), eid("Dl-Q8_0"), "DELETE")
check("delete: download dir entries deletable, the download dir itself stays", os.path.isdir(DL) and not os.path.exists(f"{DL}/only") and not os.path.exists(f"{DL}/new"))
# symlinked model file: deleting removes the link target? only the file inside the root; a link out of the roots is never listed
sc = b.scan_library(cfg.data)
check("delete: outside-root symlink target is not addressable", all("secret" not in m["first_shard"] for m in sc["models"]) and os.path.exists(f"{OUT}/secret.gguf"))
# partial entry refused
with contextlib.redirect_stdout(io.StringIO()):
    code, _ = b.library_delete(cfg, Sup(), by_name(sc, "half")["id"], "DELETE")
check("delete: partial downloads refused", code == 400 and os.path.exists(f"{M}/half.gguf.part"))
# read-only folder: error reported, nothing claimed
if os.geteuid() != 0:
    w(f"{T}/ro/Ro-Q4_K_M.gguf", 9); roc = mk([T + "/ro"]); sc = b.scan_library(roc.data); os.chmod(f"{T}/ro", 0o555)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code, body = b.library_delete(roc, Sup(), by_name(sc, "Ro-Q4_K_M")["id"], "DELETE")
        check("delete: permission error -> 500 with the reason, file still there", code == 500 and body["errors"] and os.path.exists(f"{T}/ro/Ro-Q4_K_M.gguf") and body["freed_bytes"] == 0)
    finally:
        os.chmod(f"{T}/ro", 0o755)

# ---- draft profile
P = os.path.join(T, "dr")
for i in (1, 2):
    w(f"{P}/Qwen3 235B/Big-Q4_K_M-0000{i}-of-00002.gguf", 1024 * 1024)
w(f"{P}/Small-Q4_K_M.gguf", 1024 * 1024); w(f"{P}/Gap-Q4_K_M-00001-of-00003.gguf", 10)
w(f"{P}/Small-Q4_K_M.part", 1)
cfgd = mk([P], [{"name": "Small-Q4_K_M", "model": "/x.gguf"}, {"name": "Small-Q4_K_M-2", "model": "/y.gguf"}])
before = json.dumps(cfgd.data, sort_keys=True)
sc = b.scan_library(cfgd.data)
sid = by_name(sc, "Small-Q4_K_M")["id"]
code, body = b.library_draft(cfgd, Sup(), {"id": sid})
pr = body["profile"]
check("draft: built from PROFILE_DEFAULTS, model = first shard, fit true, extra_args default",
      code == 200 and body["draft"] is True and pr["model"] == by_name(sc, "Small-Q4_K_M")["first_shard"] and pr["fit"] is True
      and pr["extra_args"] == b.PROFILE_DEFAULTS["extra_args"] and set(pr) == set(b.PROFILE_DEFAULTS) and pr["ctx"] == b.PROFILE_DEFAULTS["ctx"])
check("draft: name unique against existing profiles (-3 after -, -2)", pr["name"] == "Small-Q4_K_M-3")
check("draft: not saved (config untouched)", json.dumps(cfgd.data, sort_keys=True) == before)
check("draft: small model on 2 GPUs -> auto:1", pr["gpus"] == "auto:1")
b.live_gpu_mems = lambda: [1, 1]        # 1 MiB cards: the 2 MiB set needs more than half the VRAM
code, body = b.library_draft(cfgd, Sup(), {"id": by_name(sc, "Big-Q4_K_M")["id"]})
check("draft: needs more than half of total VRAM -> all GPUs; name sanitised from model name", body["profile"]["gpus"] == "auto:2" and body["profile"]["name"] == "Big-Q4_K_M")
b.live_gpu_mems = lambda: [24576, 24576]
code, body = b.library_draft(cfgd, Sup(), {"id": sid, "name": "my model!"})
check("draft: invalid explicit name -> 400", code == 400)
code, body = b.library_draft(cfgd, Sup(), {"id": sid, "name": "Small-Q4_K_M", "gpus": "0,1", "ctx": 8192})
check("draft: explicit gpus/ctx honoured, taken name uniquified with a note", code == 200 and body["profile"]["gpus"] == "0,1" and body["profile"]["ctx"] == 8192 and body["profile"]["name"] == "Small-Q4_K_M-3" and body["notes"])
check("draft: bad gpus / ctx -> 400", b.library_draft(cfgd, Sup(), {"id": sid, "gpus": "x"})[0] == 400 and b.library_draft(cfgd, Sup(), {"id": sid, "ctx": "big"})[0] == 400 and b.library_draft(cfgd, Sup(), {"id": sid, "ctx": 5})[0] == 400)
check("draft: unknown id 404, incomplete model 409", b.library_draft(cfgd, Sup(), {"id": "nope"})[0] == 404 and b.library_draft(cfgd, Sup(), {"id": by_name(sc, "Gap-Q4_K_M")["id"]})[0] == 409)
check("draft: unique_profile_name edge cases", b.unique_profile_name("a b/c", set()) == "a-b-c" and b.unique_profile_name("", set()) == "model" and b.unique_profile_name("x", {"x", "x-2"}) == "x-3")
try:
    b.Config.validate(dict(b.Config.normalize({"profiles": [dict(pr)], "active_profile": pr["name"], "api_keys": ["k"], "admin_password": "pw12345678"})))
    ok = True
except Exception as e:  # noqa: BLE001
    ok = False; print("validate:", repr(e))
check("draft: passes the same validation the save path uses", ok)

# ---- UI: fake-DOM render of the Library pane and the draft flow in the Profiles editor (node)
UI = os.path.join(HERE, "..", "ui.html")
if shutil.which("node"):
    html = open(UI).read()
    js = html[html.index("<script>") + 8:html.rindex("</script>")]
    lib = js[js.index("let libData = null"):js.index("/* ---- tabs: selected tab")]
    ed = js[js.index("function renderTabs()"):js.index("async function preview()")]
    harness = r'''
const els = {}; const $ = id => els[id] || (els[id] = { id, value: '', textContent: '', innerHTML: '', style: {}, disabled: false, addEventListener() {}, focus() {} });
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const tabHooks = {}; let cfg = { active_profile: 'a', profiles: [{ name: 'a', model: '/m/a.gguf' }] }, sel = 'a', status = null, sysInfo = null;
let draftProf = null; const loadSystem = () => {}; const refresh = () => {}; const act = async () => {}; const loadRouter = () => {};
const location = { hash: '' }; const toasts = []; const toast = (m, e) => toasts.push([m, !!e]);
const structuredClone = o => JSON.parse(JSON.stringify(o));
const formEls = [{ dataset: { k: 'name' }, type: 'text', value: '' }, { dataset: { k: 'model' }, type: 'text', value: '' }, { dataset: { k: 'ctx' }, type: 'number', value: '' }, { dataset: { k: 'fit' }, type: 'checkbox', checked: false }];
const document = { querySelectorAll: () => formEls };
let calls = [];
let reply = async () => ({});
const api = async (p, o) => { calls.push([p, o && o.body ? JSON.parse(o.body) : null, o && o.method]); return reply(p, o); };
const loadConfig = async () => { cfg = structuredClone(serverCfg); if (!sel || (!(draftProf && draftProf.name === sel) && !cfg.profiles.find(p => p.name === sel))) sel = cfg.active_profile; renderTabs(); fillForm(); };
let serverCfg = structuredClone(cfg);
''' + lib + ed + r'''
(async () => {
const out = {};
// empty state
libData = { models: [], ts: 1 }; renderLibrary();
out.empty = [$('libList').innerHTML, $('libHead').innerHTML];
// populated
const m = (o) => Object.assign({ id: 'id1', name: 'Q-Q4_K_M', quant: 'Q4_K_M', dir: '/m/q', first_shard: '/m/q/Q-Q4_K_M-00001-of-00002.gguf', shards: [], shards_found: 2, shards_expected: 2, complete: true, bytes: 3 * 2 ** 30, mtime: 1, mmproj: null, used_by: [], loaded: false, partial: false, hint: null, fit: { verdict: 'fits-vram', text: 'tip <b>' } }, o);
sysInfo = { disks: [{ role: 'download', path: '/srv/dl', free_gib: 12.5 }], settings: { min_free_gib_for_download: 20 } };
libData = { ts: 1, models: [m({}), m({ id: 'id2', name: 'Gap<x>', quant: null, shards_found: 2, shards_expected: 3, complete: false, hint: '1 of 3 shards missing', fit: { verdict: null, text: 'n/a' } }),
  m({ id: 'id3', name: 'Used', used_by: ['main'], loaded: true, fit: { verdict: 'too-big', text: 't' } }), m({ id: 'id4', name: 'Big', bytes: 200 * 2 ** 20, fit: { verdict: 'needs-ram', text: 'r' } })] };
renderLibrary();
out.rows = $('libList').innerHTML; out.head = $('libHead').innerHTML;
// delete modal
libDeleteOpen('id1'); out.modalOpen = [$('libModal').style.display, $('libModalGo').disabled, $('libModalName').textContent];
$('libModalIn').value = 'wrong'; libModalCheck(); out.wrong = $('libModalGo').disabled;
$('libModalIn').value = 'Q-Q4_K_M'; libModalCheck(); out.right = $('libModalGo').disabled;
reply = async (p) => p === '/api/library/delete' ? { freed_bytes: 3 * 2 ** 30 } : { models: [], ts: 2 };
calls = []; await libDeleteGo(); out.delCall = calls[0]; out.closed = $('libModal').style.display;
// add as profile -> unsaved draft in the editor
reply = async () => ({ profile: { name: 'Q-Q4_K_M', model: '/m/q/Q-Q4_K_M-00001-of-00002.gguf', gpus: 'auto:1', ctx: 32768, fit: true }, notes: ['n1'] });
calls = []; await libAddProfile('id1');
out.add = { call: calls[0], hash: location.hash, sel, tabs: $('tabs').innerHTML, note: $('pnote').textContent, form: formEls.map(e => e.type === 'checkbox' ? e.checked : e.value), cfgNames: cfg.profiles.map(p => p.name), cur: curProfile().name };
// another save (behaviour) style reload keeps the draft selected
await loadConfig(); out.afterReload = sel;
// save draft: appended to the config, no draft marker leaks, draft cleared
formEls[0].value = 'Q-Q4_K_M'; formEls[1].value = '/m/q/Q-Q4_K_M-00001-of-00002.gguf'; formEls[2].value = '32768'; formEls[3].checked = true;
calls = []; reply = async (p, o) => { if (o && o.method === 'PUT') serverCfg = JSON.parse(o.body); return {}; };
await saveProfile(false);
const put = calls.find(c => c[2] === 'PUT');
out.save = { names: put[1].profiles.map(p => p.name), active: put[1].active_profile, keys: Object.keys(put[1].profiles[1]).sort(), draftCleared: draftProf === null, sel, tabs: $('tabs').innerHTML };
// discard a draft with Delete
reply = async () => ({ profile: { name: 'tmp', model: '/x', gpus: 'auto:1', ctx: 1, fit: true }, notes: [] });
await libAddProfile('id1'); await delProfile(); out.discard = { draft: draftProf, sel, toast: toasts[toasts.length - 1][0] };
// failed save keeps the draft
await libAddProfile('id1'); formEls[0].value = 'tmp'; reply = async (p, o) => { if (o && o.method === 'PUT') throw new Error('boom'); return {}; };
await saveProfile(false); out.failed = { draftKept: !!draftProf, sel };
console.log(JSON.stringify(out));
})().catch(e => { console.log(JSON.stringify({ error: String(e && e.stack || e) })); });
'''
    hpth = os.path.join(T, "harness.js"); open(hpth, "w").write(harness)
    r = subprocess.run(["node", hpth], capture_output=True, text=True, timeout=60)
    try:
        o = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        o = None
        print(r.stderr[-600:])
    check("ui: library render does not throw", o is not None and "error" not in o)
    if o and "error" in o:
        print(o["error"][:600])
    if o and "error" not in o:
        check("ui: calm empty state points at the System tab", "No models found" in o["empty"][0] and 'href="#system"' in o["empty"][0] and "0 models" in o["empty"][1])
        rows = o["rows"]
        check("ui: header totals and free space on the download dir (amber below the minimum)", "4 models" in o["head"] and "12.5 GiB free in /srv/dl" in o["head"] and "chip warn" in o["head"])
        check("ui: quant chip, size, fit chips with tooltips", "Q4_K_M" in rows and "3.0 GiB" in rows and "200 MiB" in rows and "fits VRAM" in rows and "needs RAM" in rows and "too big" in rows and 'title="tip &lt;b>"' in rows)
        check("ui: incomplete set shows amber 2/3 and the hint; names escaped", 'chip warn" title="1 of 3 shards missing">2/3' in rows and "Gap&lt;x>" in rows and "Gap<x>" not in rows)
        check("ui: used-by chip and loaded badge", 'chip done">main<' in rows and "chip ok\">loaded" in rows)
        used_row = rows[rows.index("Used"):]
        check("ui: delete disabled with a tooltip for used/loaded models, enabled for idle ones",
              'class="danger" disabled title="Loaded right now' in used_row and 'class="danger"  title=""' in rows)
        check("ui: delete modal needs the exact name", o["modalOpen"] == ["flex", True, "Q-Q4_K_M"] and o["wrong"] is True and o["right"] is False)
        check("ui: delete call sends id + confirm token only", o["delCall"][0] == "/api/library/delete" and o["delCall"][1] == {"id": "id1", "confirm": "DELETE"} and o["closed"] == "none")
        a = o["add"]
        check("ui: Add as profile posts the id, jumps to #profiles, shows an unsaved draft tab",
              a["call"][0] == "/api/library/profile" and a["call"][1] == {"id": "id1"} and a["hash"] == "#profiles" and a["sel"] == "Q-Q4_K_M"
              and "unsaved" in a["tabs"] and "Unsaved draft" in a["note"] and a["cur"] == "Q-Q4_K_M")
        check("ui: draft form filled, and the draft is NOT in the loaded config", a["form"][0] == "Q-Q4_K_M" and a["form"][2] == 32768 and a["form"][3] is True and a["cfgNames"] == ["a"])
        check("ui: a config reload keeps the draft selected", o["afterReload"] == "Q-Q4_K_M")
        sv = o["save"]
        check("ui: saving the draft appends it through PUT /api/config, no draft marker saved, draft cleared",
              sv["names"] == ["a", "Q-Q4_K_M"] and sv["active"] == "a" and "draft" not in " ".join(sv["keys"]) and sv["draftCleared"] and "unsaved" not in sv["tabs"])
        check("ui: Delete on a draft discards it without touching the server", o["discard"]["draft"] is None and o["discard"]["sel"] == "a" and "discarded" in o["discard"]["toast"])
        check("ui: a failed save keeps the draft", o["failed"]["draftKept"] and o["failed"]["sel"] == "tmp")
else:
    print("SKIP ui render (node missing)")

shutil.rmtree(T, ignore_errors=True)
print("FAILED: " + ", ".join(fails) if fails else "ALL library checks passed")
sys.exit(1 if fails else 0)
