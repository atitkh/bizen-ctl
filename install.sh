#!/usr/bin/env bash
# Install bizen-ctl as a systemd --user service (no root needed).
# Re-running is safe: keeps an existing config.json.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$DIR/config.json"
UNIT_DIR="$HOME/.config/systemd/user"
PY="$(command -v python3)"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!!\033[0m  %s\n' "$*"; }

# 1. config with fresh secrets
if [[ ! -f "$CFG" ]]; then
  say "Creating config.json with a new API key and admin password"
  "$PY" - "$DIR/config.example.json" "$CFG" <<'EOF'
import json, secrets, sys, os
src, dst = sys.argv[1:3]
c = json.load(open(src))
c["api_keys"] = ["bz-" + secrets.token_hex(24)]
c["admin_password"] = secrets.token_urlsafe(12)
with open(dst, "w") as f:
    json.dump(c, f, indent=2)
os.chmod(dst, 0o600)
print(f"\n   admin password: {c['admin_password']}\n   API key:        {c['api_keys'][0]}\n")
EOF
  warn "Save those two values; you can change both in the web UI."
else
  say "Keeping existing config.json"
fi

# 2. sanity checks
LS="$("$PY" -c "import json,os;print(os.path.expanduser(json.load(open('$CFG'))['llama_server']))")"
[[ -x "$LS" ]] || warn "llama-server not found at $LS (fix 'llama_server' in the UI or config.json)"
command -v numactl >/dev/null || warn "numactl not installed: CPU pinning falls back to taskset, without memory binding"
command -v nvidia-smi >/dev/null || warn "nvidia-smi not found"
PORT="$("$PY" -c "import json;print(json.load(open('$CFG'))['listen_port'])")"
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  warn "Something is already listening on port $PORT. Stop your manual llama-server first (or change listen_port)."
fi

# 3. systemd user unit
if ! systemctl --user show-environment >/dev/null 2>&1; then
  warn "systemd --user is not available in this session."
  echo "    Run it in tmux instead:  tmux new -d -s bizen '$PY -u $DIR/bizenctl.py --config $CFG'"
  exit 0
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/bizen-ctl.service" <<EOF
[Unit]
Description=bizen-ctl (llama-server controller + web UI)
After=network-online.target

[Service]
ExecStart=$PY -u $DIR/bizenctl.py --config $CFG
Restart=always
RestartSec=3
KillMode=mixed
TimeoutStopSec=40

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now bizen-ctl.service
sleep 1
systemctl --user --no-pager --lines=5 status bizen-ctl.service || true

# 4. keep running after logout
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]]; then
  loginctl enable-linger "$USER" 2>/dev/null || true
fi
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" == "yes" ]]; then
  say "Lingering enabled: the service keeps running after you log out and starts on boot."
else
  warn "Could not enable lingering. The service stops when your last SSH session ends."
  echo "    Ask an admin to run:  sudo loginctl enable-linger $USER"
  echo "    Until then, keep one session open (e.g. tmux) or use the tmux command above."
fi

say "Done. Web UI: http://<this-host>:$PORT/   API: http://<this-host>:$PORT/v1"
echo "    Logs:    journalctl --user -u bizen-ctl -f"
echo "    Control: systemctl --user {status|restart|stop} bizen-ctl"
