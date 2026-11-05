# bizen-ctl

One-port controller for llama-server on a shared GPU box (written on a Bizon workstation, any Linux box works). Python stdlib only.

- `http://host:8080/v1/...`: OpenAI-compatible API, needs `Authorization: Bearer <key>`.
  The first request loads the active profile (about 1 min from the NAS). After `idle_ttl_min`
  with no requests it unloads and frees the GPUs.
- `http://host:8080/`: web UI (admin password). Edit and switch profiles, start/stop/restart,
  per-GPU VRAM, live log, tok/s, test prompt, API keys.

## Install (on the GPU box, as your user)

```
tar xzf bizen-ctl.tgz -C ~ && cd ~/bizen-ctl
# stop the llama-server you started by hand first (it holds port 8080)
./install.sh
```

The installer writes `config.json` with a new API key and admin password and prints both once.
It also installs and starts the `systemd --user` service `bizen-ctl` and tries to enable
lingering so the service survives logout.

## Day to day

- Everything is in the UI. "Save, make active & restart" applies a profile.
- `systemctl --user status|restart|stop bizen-ctl`
- `journalctl --user -u bizen-ctl -f` shows controller logs. The llama-server log is in the UI.

## Updating from the portal

The header shows the short commit hash and a chip: `up to date`, `N behind` (as of the last fetch; the page never fetches on a timer), `local changes`, `diverged`, or `not a git checkout`. **Update** opens a dialog that runs a check and then offers **Update now**.

- `GET /api/controller/version` (admin): `{is_git, commit, commit_full, branch, dirty, remote_host, upstream, ahead, behind, last_fetch}`. `remote_host` is the host only; credentials in a remote URL are stripped from everything the controller returns or logs.
- `POST /api/controller/update` `{"action":"check"}` runs `git fetch --quiet` (60 s limit) and returns `{ok, reason, commits[<=20], files[], restart_needed, model_loaded, would_unload, ahead, behind, ...}`. `restart_needed` is true when `bizenctl.py` or `install.sh` changed; UI/docs/tests-only updates need no restart. `ok:false` carries a readable `reason`: not a git checkout, no upstream, local changes in the tree, diverged / ahead of the upstream, or a failed fetch.
- `{"action":"apply","confirm":true}` repeats the checks, runs `git pull --ff-only` (90 s limit) and returns `{ok, from, to, restart_needed, restarting}`. Updates are serialized (a second one gets 409). When a restart is needed the response goes out first, then after about 1.5 s the controller shuts down through the normal SIGTERM path (llama-server is stopped, so a loaded model is unloaded and takes minutes to reload) and **systemd `Restart=always` starts it again**; the page polls `/api/status` until it answers and reloads. Without a supervisor the process just stops and you start it by hand. With only UI/docs changes nothing restarts and the page reloads.
- It never runs `git reset`, `checkout -f`, `clean`, `update.sh` or `systemctl` (a script started inside the service would die with the unit's cgroup). `git` runs with `GIT_TERMINAL_PROMPT=0`, so a remote that needs credentials fails with a short error instead of hanging; the checkout must be able to fetch non-interactively (deploy key or credential helper).
- `update.sh` still works from a shell and stays the way to update when the portal is down.

## Profile fields

| field | meaning |
|---|---|
| gpus | `auto:N` picks the N GPUs with the least VRAM in use at start (ties go to the highest index, away from other users' default GPU 0), or a fixed list like `6,7` |
| fit / fit_target_mib | let `--fit` size the GPU/CPU split, leaving this much VRAM free per GPU |
| n_cpu_moe | used only when fit is off |
| pin / threads | `auto` = one thread per physical core on the socket where the container owns the most cores, with memory bound to it (numactl). This is recomputed at every start, so it survives LXD reshuffling the cpuset. |
| extra_args | appended as-is (`-fa on --load-mode none` by default) |

## Router (Kev)

Kev (a small decision model) picks which profile should answer a prompt.

1. Install Kev once in `~/kev` (`uv sync --extra serve`) and stop any Kev you started by hand (port 8009).
2. In the UI, Router section: add one route per profile with a plain description of when to use it,
   choose a default profile, tick the enable box, Save. bizen-ctl starts Kev itself on first use
   (on a GPU that llama-server isn't using) and unloads it after its own idle timer.
3. Use it:
   - `POST /v1/chat/completions` with `"model": "auto"` (the alias is configurable). Kev picks the
     profile, the controller switches to it, and you get the answer.
   - `POST /v1/route` with `{"prompt": "..."}` returns Kev's decision only. Nothing is loaded.
   - `GET /v1/models` lists the profiles and the alias.
4. Response headers on routed requests: `X-Bizen-Route` (profile), `X-Bizen-Route-P` (probability),
   `X-Bizen-Route-Fallback` (why the default was used), `X-Bizen-Route-Served` (a different profile was
   kept because other requests were in flight).

### Changing Kev's model

Router section, "Kev server settings": the Kev model field is a picker (published checkpoints with size
hints) that also takes any Hugging Face id or local run directory. Pick the precision next to it; the line
below the field says whether weights plus working memory fit on one card. Kev loads on a single GPU only.

- "Save & restart Kev" applies it at once. A plain Save only flags "restart needed"; the running Kev keeps
  its old model until it restarts or unloads on idle.
- The first start of a new model downloads its base weights (kev-4b: about 8 GB, roughly 12 minutes at your
  11.7 MB/s). The Kev start timeout is 900 s; raise `kev.start_timeout_s` for a bigger download.
- CUDA graphs: Kev replays its passes as CUDA graphs by default, which needs a state buffer of about 2 GB
  on kev-4b and kev-9b. On an 11 GB card kev-4b (8.4 GiB fp16 weights) only fits with "CUDA graphs: off"
  (`KEV_CUDA_GRAPHS=0`); the picker sets that for you. Requests are slower with graphs off.
- On the original box (Turing, 2080 Ti) there is no bf16, so use fp32, or fp16 for models that need it (kev-4b). fp16 was not
  compared with fp32 here, so check a few answers.

Rules: the loaded profile is only replaced when no other request is in flight; a switch costs a model
load (about a minute). If Kev is unsure (below the minimum probability) or unavailable, the default
profile is used. Requests whose `model` is not the alias are never routed.

## Admin page tabs

The page at `/` is a tabbed dashboard; the selected tab is kept in the URL hash (`#chat`, `#profiles`, ...), so a reload stays put.

- **Dashboard**: KPIs, per-GPU cards (memory bar plus a 5-minute util/memory sparkline; blue = this server's GPUs), a RAM panel (stacked used / cache / free bars for the container and for each NUMA node, 10-minute sparkline, llama-server RSS and bytes read from disk), a CPU grid (one cell per core, pinned cores outlined) and the generation state. Charts are inline SVG fed by `/api/metrics`.
- **Chat**: streaming chat against the running profile (system prompt, max tokens, temperature, time-to-first-token and live tok/s, Stop, Clear). A cold model shows a "loading, first start can take minutes" banner.
- **Profiles**: profile editor and Behaviour. **Library**: models on disk, plus Browse and Downloads for Hugging Face (see below). **Kev**: router settings and router test. **Logs**: llama-server log. **System**: detected hardware and portable settings (see below). **Access**: API keys, admin password, endpoints.

RAM note: "available" counts reclaimable page cache, and an mmap'd model *is* page cache, so it barely moves while a model loads or runs. The KPI therefore shows used (total - available) / total, and the RAM panel splits used from cache, per NUMA node. Two views, labelled in the panel: the **this container** bar comes from `/proc/meminfo` (inside LXD that is the container's own lxcfs view, e.g. 1 GiB used), while the **host (all users)** node bars come from `/sys/devices/system/node`, which is host-wide and includes every other user (e.g. 126 of 252 GiB used). They are not comparable. The **host free** KPI tile is free + reclaimable file cache summed over the nodes (`mem.host_free_mib`, null and hidden when there is no node data; each node also carries `avail_mib`).

### New admin endpoints (cookie login, like the rest of `/api/*`)

- `GET /api/metrics?since=<unix ts>`: samples newer than `since` from a 1 s background sampler (ring buffer, last 600). Each sample: `t`, `ram_used`, `ram_cached`, `ram_free`, `nodes` (used MiB per NUMA node), `gpus` (`[index, mem_used_mib, util]`), `cpu` (busy % per core, same order as the top-level `cpu_ids`), `rss` (llama-server RSS MiB), `read_bytes`. Also returns `now`, `interval_s`, `cpu_ids`, `cpus_pinned`.
- `POST /api/chat` with `{messages, max_tokens?, temperature?}`: streams `text/event-stream`. Starts with `data: {"bizen":"loading"}` keepalives while the model loads, then `{"bizen":"ready"}`, then llama-server's own chat-completion chunks, then `[DONE]`. Errors arrive in-stream as `data: {"error": ...}`. Closing the connection stops generation.
- `GET /api/status` keeps its shape; `mem` gained `used_mib`, `free_mib`, `buffers_mib`, `cached_mib`, `swap_*`, `nodes[]`, `cgroup` and `child` (`rss_mib`, `read_bytes`).

### Model load and Generation panels (Dashboard tab)

- **Model load**: progress bar with MB/s and ETA, phase chips (starting, reading tensors, allocating/warming, ready), a stacked bar per device (weights / KV / compute) for each GPU and the host (CPU) buffer, and a one-row layer strip coloured by device when the log prints per-layer assignments. If the llama.cpp build prints none of that (newer builds only log `load_model`/`threadpool init`/`listening`), `/api/loadmap` and `server.load` still carry `placement` (`source: "vram-delta"`): per profile GPU the VRAM added since the profile started (`nvidia-smi` `memory.used` now minus the snapshot taken just before launch, clamped at 0), the child's RSS as `host_mib`, `total_mib`, `gpu_share_pct`, and a copy frozen when the server became ready (`frozen: true`, with the live numbers under `live`), which the panel draws as one bar per GPU plus a host bar. Caveats: other users' processes on the same GPUs inflate the delta, and RSS counts mmapped weights. Phase chips advance on `threadpool init` / `initializing, n_slots` too. Idle shows a calm "no model loading" state and the last load summary stays after ready. The **Logs** tab has a *Download startup log* button (text of the current log from the last `starting profile` line).
- **Generation**: live tok/s, a 3-minute tok/s sparkline, and a per-GPU utilisation strip (time on x, shaded by util %, generation-active spans marked above). Tokens are counted from streamed SSE chunks passing through `/v1` and `/api/chat` (one chunk ~ one token), so no `--metrics` flag or `/slots` is needed.
- `GET /api/status` gained `server.load` = `{phase, bytes_read, bytes_total, pct, eta_s, mbps}` (shard-aware model size; `bytes_read` from the child's `/proc/<pid>/io` `rchar`, which is what `--load-mode none` does, with an RSS-growth fallback for mmap; `pct` is monotonic and capped at 99 until ready) and `server.gen` = `{active, tokens, ttft_s, tps, last_tps}`. Both are also on every `/api/metrics` sample.
- `GET /api/loadmap` (login required): `{profile, state, layers_total, layers_offloaded, devices:[{name, weights_mib, kv_mib, compute_mib}], cpu_mib, per_layer:[{layer, device}], fit:{raw:[...]}}` parsed from the startup log. Every pattern is optional, so a build that prints less just gives fewer fields. `test/fixtures/startup-q3.log` is a SYNTHETIC illustration of llama.cpp's format, not a real log; replace it with a real startup log when you have one.
- Not verified on the real box: llama.cpp's actual log wording for this build, `rchar` behaviour on the NAS mount, browser rendering.

## Library tab

Models already on disk, for tidying up and for creating profiles without typing paths. Open `#library`.

- Scans every `model_dirs` entry and `download_dir` (deduped, recursive, read-only until you delete something). Multi-shard sets (`-0000N-of-0000M.gguf`) are one row; `mmproj` files are not rows, they attach to the models in the same folder. Symlinks that resolve outside the model directories are ignored. Partial downloads (`.part`, `.incomplete`) and sets with a missing shard show an amber shard count (`2/3`) and a hint instead of being hidden.
- Columns: name and folder, quant chip (`UD-Q3_K_XL`, `Q4_K_M`, `IQ2_XXS`, `BF16`... parsed from the file name, blank when none), size, shards, **used by** (profiles whose model is the first shard or any shard), a **loaded** badge, and a fit chip. The header shows model count, total size and free space on the download directory (amber below `min_free_gib_for_download`).
- **Fit** is plain arithmetic on file size: it fits VRAM when the weights are at most 90% of the summed VRAM of all GPUs, needs RAM when they fit in 90% of VRAM + 90% of system RAM, otherwise too big. No GPU data means no verdict. The tooltip repeats that this is a rough size-based estimate, not a speed prediction.
- **Delete**: asks you to type the model name, then calls `POST /api/library/delete` with `{"id", "confirm": "DELETE"}`. The id is resolved from a fresh scan (a client path is never accepted). Refused with 409 while a profile uses the model or it is loaded, 400 without the confirm token or for partial downloads. It `os.remove`s only that set's own `.gguf` shards, plus its `mmproj` when no other model shares the folder and no profile names it in `extra_args`; never `rmtree`, other files in the folder stay. The folder is removed only if it is then empty and is a strict subdirectory of a model directory. One line per delete goes to the controller log.
- **Add as profile**: `POST /api/library/profile` with `{id, name?, gpus?, ctx?}` returns a draft (nothing is saved): `PROFILE_DEFAULTS` with `model` = first shard, a unique `[A-Za-z0-9._-]` name from the model name, `gpus` `auto:N` (all GPUs when the model needs more than half of the total VRAM, else 1 or 2), `fit` true. The UI opens it as a dashed "unsaved" tab in the Profiles editor; **Save** goes through the normal save path, **Delete** discards it.
- `GET /api/library` (login required) returns `{models:[{id, name, quant, dir, first_shard, shards:[{path,bytes}], shards_found, shards_expected, complete, bytes, gib, mtime, mmproj, used_by, loaded, partial, hint, fit:{vram_total_mib, ram_total_mib, verdict, text}}], roots, ts}`, newest first. `GET /api/models` is unchanged (the profile editor's list).
- Not verified on the real box: browser rendering, delete permissions and speed on the NAS CIFS mount, scan time on a large share.

## Browse and download (Library tab, Browse / Downloads sub-views)

Find a GGUF on Hugging Face and pull one quant into `download_dir` from the portal. `#library/browse` and `#library/downloads`; Installed (`#library`) is the table above.

- **Browse**: search box plus sort (downloads, likes, recently updated). A result opens a detail pane with licence / base model chips, a quant table (name, quant chip, size, shards, **fit** chip with the Library's size-only verdict, "already downloaded" when a complete model of that name is on disk, **Download**) and the model card, shown as plain text (cut at 20000 chars, front matter stripped). Multi-shard sets are one row and download together; a repo-root `mmproj` (or one in the same folder) is fetched with the set. Gated models need a token under System; with none set the page says so.
- **Downloads**: one card per download with progress bar, MB/s and ETA (10 s window), bytes done / total, Pause / Resume / Cancel / Remove (and "Remove + delete partial files"). A finished one has **Add as profile** (same unsaved-draft flow as Installed). While something downloads, the Dashboard KPI row shows a download tile (hidden otherwise).
- **Where files go**: `download_dir/<repo name>/<quant folder, or "files" for repo-root files>/<file>`, always written as `<file>.part` and renamed once the size equals the repo's listed size. Anything else (wrong size, short body) is an error and the `.part` stays for a resume.
- **Queue**: one transfer at a time (reads of up to 1 MiB that block on the socket, so no busy loop and little CPU), up to 3 waiting, FIFO. A request is refused with 409 when `remaining bytes + min_free_gib_for_download` does not fit on the download filesystem (`not enough free space: need X GiB, have Y GiB`), 400 when the directory is not writable.
- **Pause / resume / cancel**: all stop the transfer and keep the `.part`. Resume sends `Range: bytes=N-` (plus `If-Range` with the ETag from the first response); a server that ignores Range, or a changed file, makes it start over. A `.part` with no remembered ETag (hand-copied, or from before a history clear) is continued by size only, so delete it if you doubt where it came from. Cancel/error entries can be resumed too. State is kept in `downloads.json` next to `config.json` (no token in it); after a controller restart interrupted downloads are listed as **paused**, nothing starts by itself.
- **Security**: the token is sent only as `Authorization: Bearer` to the host (scheme, host, port) of `hf_endpoint`; redirects are followed by hand and the header is dropped on any hop to another host, e.g. the CDN. Repo ids must match `^[A-Za-z0-9][\w.\-]*/[\w.\-]+$`, file names are plain relative `.gguf` paths (no `..`, no absolute paths, no `.`/empty parts), and the destination is checked with `realpath` to stay inside `download_dir` (also at write time and for delete-partial). The token is never logged or returned.
- **API** (login required): `GET /api/hf/search?q=&sort=downloads|likes|lastModified&limit=20` (cached 60 s), `GET /api/hf/repo?id=owner/name`, `GET /api/downloads`, `POST /api/downloads {repo, files:[paths of one quant set]}`, `POST /api/downloads/<id>/cancel|pause|resume|remove` (`remove` takes `{"delete_partial": true}`). `GET /api/status` gained `downloads_active` (`{id,name,pct,mbps,eta_s}` or null). Every Hugging Face call has a 15 s timeout and runs on a helper thread; failures come back as a readable `error` (404 -> "repo not found or gated (set a token in System)", 401/403 with a token -> "token rejected").
- **Bandwidth and NAS notes**: there is no rate limit; a download uses whatever the link gives. On a shared box point `download_dir` at a local disk and move finished files to the NAS mount afterwards if the NAS is slow for random writes, or expect the write speed of the NAS to be the limit when it is the target. The final rename stays inside the same folder, so it is a cheap metadata operation on any filesystem. `hf_endpoint` can be a mirror. Large downloads (tens of GiB) fill the disk gradually; the free-space check counts every queued transfer.
- Tests: `test/mock-hf.py` (HF API + a second "CDN" port, Range, ETag, gated repo, slow mode) and `test/test_download.py`; `run-tests.sh` has an end-to-end section against the mock. Never against the real huggingface.co.
- Not verified against the real service: Hugging Face's exact field names (`lastModified`, `lfs.size`, `gated`, `gguf.architecture`, `expand[]` on the list call), tree pagination via the `Link` header, the real CDN redirect and ETag behaviour, speed on the box, and browser rendering.

## System tab and portability

The **System** tab (`#system`, between Logs and Access) shows what bizen-ctl detected on the machine and holds the
paths and defaults that used to be fixed to the original box. Everything is optional: a `config.json` without any
of the keys below behaves exactly as before.

- Hardware (`GET /api/system`, admin login): OS/kernel/Python/hostname, CPU model, sockets, physical cores, threads, the CPUs
  this process may use, NUMA nodes (cpus, memory) and the default pin; RAM total, container limit and per-node memory;
  GPUs (name, VRAM, driver, compute capability, CUDA version when `nvidia-smi` reports it); free space for the download dir,
  model dirs and HF cache; tools (`llama-server` path, executable flag and `--version`, cached 60 s; `numactl`, `taskset`,
  `nvidia-smi`, `hf`/`huggingface-cli`). Anything that cannot be read is `null` and shows as "–". Every probe has a 3 s timeout.
  "Rescan hardware" re-reads it.
- Settings (`POST /api/system/settings`, same persistence as `PUT /api/config`: lock, `config.json.bak`, mode 600). Send only the
  keys you want to change. `"dry_run": true` validates without saving (the UI's Check button). Saving never restarts a running
  model, and changing `llama_server` or `model_dirs` does not flag a restart.

| key | default | notes |
|---|---|---|
| `download_dir` | `""` = first `model_dirs` entry | absolute; created if missing and proven writable with a temp file |
| `model_dirs` | `["/srv/fs/models"]` | absolute; a missing dir only warns |
| `kev.hf_home` | `"/srv/fs/hf"` | the HF cache; editable on the System tab as "Hugging Face cache" |
| `llama_server` | `~/llama.cpp/build/bin/llama-server` | must be an executable file; a failing `--version` only warns |
| `hf_endpoint` | `https://huggingface.co` | http(s) URL, e.g. a mirror |
| `hf_token` | `""` | write-only: `GET /api/config` and `/api/system` return `has_hf_token` only. An empty or omitted token keeps the saved one; `"clear_hf_token": true` removes it |
| `min_free_gib_for_download` | `20` | used by the download stages that build on this tab |
| `gpu_bucket_mib` | `512` | `auto:N` ranks GPUs by used VRAM in buckets of this size; ties go to the highest index |
| `gpu_fallback_count` | `8` | GPU indices assumed for `auto:N` when `nvidia-smi` is missing |
| `download_mbps` | `11.7` | only the "about N min" estimate in the Kev hint |

The Kev fit hint now uses the VRAM of the GPU Kev runs on (else the explicit `kev.gpus`, else the smallest GPU llama-server is not using)
and says "GPU size unknown" instead of guessing when `nvidia-smi` gives nothing. Not verified on the real box: the exact `nvidia-smi`
fields on driver 535 and browser rendering of the tab.

## Notes

- llama-server itself listens only on 127.0.0.1 with a random internal key. All outside access goes through the controller.
- `config.json` holds secrets (mode 600). Each save keeps the previous version as `config.json.bak`.
- Tests: `test/run-tests.sh` runs the controller against a mock llama-server.
