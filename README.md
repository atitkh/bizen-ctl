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
- **Profiles**: profile editor and Behaviour. **Kev**: router settings and router test. **Logs**: llama-server log. **Access**: API keys, admin password, endpoints.

RAM note: "available" counts reclaimable page cache, and an mmap'd model *is* page cache, so it barely moves while a model loads or runs. The KPI therefore shows used (total - available) / total, and the RAM panel splits used from cache, per NUMA node (read from `/sys/devices/system/node`, which is host-wide even inside LXD, unlike lxcfs's `/proc/meminfo`).

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

## Notes

- llama-server itself listens only on 127.0.0.1 with a random internal key. All outside access goes through the controller.
- `config.json` holds secrets (mode 600). Each save keeps the previous version as `config.json.bak`.
- Tests: `test/run-tests.sh` runs the controller against a mock llama-server.
