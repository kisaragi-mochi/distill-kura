#!/usr/bin/env bash
# One command, no model, no config, nothing installed: watch a kura come into being.
#
# It runs the whole loop against a scripted fake model, so the result is identical on
# every machine — which is the point. If this script fails on a clean Ubuntu box, the
# README's first ten minutes are broken and nobody should have to discover that by hand.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'kill "${FAKE_PID:-}" 2>/dev/null || true; rm -rf "$WORK"' EXIT
export PYTHONPATH="$ROOT"
cd "$WORK"

say() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

say "a fake model, so the demo is the same everywhere"
python3 "$ROOT/scripts/fake_llm.py" 18099 & FAKE_PID=$!
ready=""
for _ in $(seq 50); do
  if curl -sf "http://127.0.0.1:18099/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.1
done
if [ -z "$ready" ]; then
  echo "the fake model never became reachable on port 18099 (is it already in use?)" >&2
  exit 1
fi

say "a journal to distil"
mkdir -p journal
python3 - <<'PY'
import json
lines = [
    ("user", "put the archive on the slow disk, the fast one is for scratch"),
    ("assistant-tool", "df -h /data"),
    ("tool", "/data 3.2T used 1.1T avail"),
    ("assistant", "I think we should mirror it too, but that is only my hunch."),
]
with open("journal/session.jsonl", "w", encoding="utf-8") as f:
    for kind, text in lines:
        if kind == "user":
            f.write(json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": text}]}}) + "\n")
        elif kind == "tool":
            f.write(json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content": [{"type": "text", "text": text}]}]}}) + "\n")
        elif kind == "assistant-tool":
            f.write(json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": text}}]}}) + "\n")
        else:
            f.write(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}) + "\n")
    # enough material that the distiller considers the batch worth drinking
    f.write(json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "padding " * 2000}]}}) + "\n")
PY

say "a kura and its config"
python3 -m distill_kura.cli init main --path "$WORK/kura" >/dev/null
cat > kura.toml <<TOML
[server]
default = "main"
[models.thinker]
url = "http://127.0.0.1:18099/v1"
model = "fake"
[stores.main]
path = "$WORK/kura"
label = "demo kura"
[distill.journals]
claude = "$WORK/journal"
TOML

say "distil: drink -> spot -> gate -> draft"
python3 -m distill_kura.cli -c kura.toml distill run
python3 -m distill_kura.cli -c kura.toml distill drafts

say "drain: the scribe re-reads each draft and pours or tosses"
python3 -m distill_kura.cli -c kura.toml distill drain

say "the store now holds"
python3 -m distill_kura.cli -c kura.toml doctor > doctor.json
python3 - <<'PY'
import json
d = json.load(open("doctor.json"))
print(f"  {d['memories']} memories, {d['links_resolved']} links, "
      f"{len(d['islands'])} islands, {len(d['links_dead'])} dead links")
PY
cat kura/MEMORY.md

say "weave the resident map, and print the block a host would inject"
python3 -m distill_kura.cli -c kura.toml weave --no-model
python3 -m distill_kura.cli -c kura.toml prefill

say "recall, by meaning"
python3 -m distill_kura.cli -c kura.toml recall "where does the archive live?"

say "what it cost"
python3 -m distill_kura.cli -c kura.toml bench compress

say "provenance: why does this memory exist?"
ls kura/_evidence/ 2>/dev/null | head -3 || echo "  (none: nothing was poured)"

printf '\n\033[1;32m✓ clean-room demo finished\033[0m — everything above ran with no model and no network.\n'
