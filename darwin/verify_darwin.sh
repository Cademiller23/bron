#!/usr/bin/env bash
# DARWIN end-to-end verification.
# Run from your repo root (the folder that contains darwin/):
#     bash verify_darwin.sh
#
# It (1) loads your .env, (2) checks deps, (3) runs the preflight, (4) runs the
# full test suite -- gated integration tests included, since the .env is loaded.
 
set -uo pipefail
 
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
echo "Python: $($PY --version 2>&1)  ($PY)"
 
# Load .env into this shell so (a) the app sees secrets and (b) any pytest
# skipif(env-var) gates open and the gated integration tests actually run.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env 2>/dev/null || echo "NOTE: some .env lines didn't parse in bash -- quote any value containing & or spaces."
  set +a
  echo "Loaded .env into the environment."
else
  echo "WARNING: no .env file in this directory."
fi
echo
 
echo "==> 1/4  Dependency check"
$PY - <<'PYEOF'
import importlib, sys
need = ["pydantic", "pymongo", "motor", "google.genai", "voyageai", "ortools", "pytest", "hypothesis"]
missing = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, f"{type(e).__name__}: {e}"))
if missing:
    print("Missing/broken dependencies:")
    for m, e in missing:
        print(f"  - {m}: {e}")
    sys.exit(1)
print("OK -- all core dependencies import.")
PYEOF
if [ $? -ne 0 ]; then
  echo "Install them, e.g.:  $PY -m pip install pydantic pymongo motor google-genai voyageai ortools pytest hypothesis python-dotenv --break-system-packages"
  exit 1
fi
echo
 
echo "==> 2/4  Preflight (.env wiring + MongoDB + Gemini + Voyage + fleet credential audit)"
$PY darwin_preflight.py
PRE=$?
echo
if [ $PRE -ne 0 ]; then
  echo "Preflight found BLOCKERS (above). Offline tests will still run, but a real solve fails until these are fixed."
fi
 
echo "==> 3/4  Full test suite (offline ~565 tests; gated integration tests run too, .env is loaded)"
$PY -m pytest -q
TESTS=$?
echo
if [ $TESTS -ne 0 ]; then
  echo "Tests failed. If failures mention MAX / MiniMax / a 404 model id, switch the fleet to Gemini-only (see notes) and re-run."
  exit 1
fi
 
echo "==> 4/4  Result"
if [ $PRE -eq 0 ]; then
  echo "GREEN: brain is built, every test passes, and MongoDB/Gemini/Voyage are wired. [OK]"
else
  echo "Tests pass, but fix the preflight blockers above before a live end-to-end solve."
fi
 