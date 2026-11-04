#!/usr/bin/env bash
# Pull the latest bizen-ctl and restart the service only if the Python changed.
# ui.html is read per request, so UI-only updates need just a browser refresh.
# Usage: ./update.sh [--force-restart]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

before="$(git rev-parse HEAD)"
git pull --ff-only
after="$(git rev-parse HEAD)"

if [[ "$before" == "$after" ]]; then
  echo "Already up to date ($(git rev-parse --short HEAD))."
  [[ "${1:-}" == "--force-restart" ]] || exit 0
else
  git --no-pager log --oneline "$before..$after" | head -10
fi

if [[ "${1:-}" == "--force-restart" ]] || ! git diff --quiet "$before" "$after" -- bizenctl.py; then
  echo "bizenctl.py changed: restarting (this unloads the running model)."
  systemctl --user restart bizen-ctl
  sleep 1
  systemctl --user --no-pager --lines=3 status bizen-ctl | head -6
else
  echo "UI/docs only: no restart needed, refresh the browser."
fi
