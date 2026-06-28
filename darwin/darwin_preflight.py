#!/usr/bin/env python3
"""
DARWIN preflight — verifies the .env is wired in and every external service works.

Run from your repo root (the folder that contains darwin/):
    python darwin_preflight.py

The CORE checks (env vars + MongoDB + Gemini + Voyage) do NOT import your darwin
package, so they run even if the package has an unrelated import error. The
OPTIONAL section at the end imports your fleet/registry and Conductor and audits
which models you actually have credentials for.

Override the test model names if needed:
    DARWIN_GEMINI_TEST_MODEL=gemini-3.5-flash  DARWIN_VOYAGE_TEST_MODEL=voyage-3.5  python darwin_preflight.py
"""
import os
import sys
import traceback

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_ICON = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[INFO]"}
results = []


def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"{_ICON[status]} {name}" + (f" -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# 0. Load the .env  (the heart of "is the .env wired in")
# --------------------------------------------------------------------------
def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()  # searches CWD and parents
        if os.path.exists(".env"):
            load_dotenv(".env", override=False)
        record("python-dotenv load_dotenv()", PASS, "found and called")
    except ImportError:
        if os.path.exists(".env"):
            for raw in open(".env"):
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            record("manual .env parse", WARN,
                   "python-dotenv NOT installed -- I parsed .env by hand. "
                   "Install it (pip install python-dotenv) and call load_dotenv() at your app's entrypoint, "
                   "or your real process won't see these vars.")
        else:
            record(".env loading", FAIL, "no python-dotenv and no .env file in this directory")


load_env()

# --------------------------------------------------------------------------
# 1. Required environment variables
# --------------------------------------------------------------------------
REQUIRED = ["MONGODB_URI", "MONGODB_DB", "GEMINI_API_KEY", "VOYAGE_API_KEY"]
for var in REQUIRED:
    val = os.environ.get(var, "")
    if val:
        record(f"env {var}", PASS, (val[:10] + "...") if len(val) > 10 else "set")
    elif var == "MONGODB_URI":
        record(f"env {var}", FAIL,
               "MISSING. NOTE: ATLAS_API_KEY is NOT a connection string. You need "
               "MONGODB_URI=mongodb+srv://USER:PASS@cluster.xxx.mongodb.net/?retryWrites=true&w=majority")
    else:
        record(f"env {var}", FAIL, "missing or empty")

# Informational: optional / not-needed vars
for var in ("DigitalOceanModel", "DIGITALOCEAN_MODEL", "DIGITALOCEAN_MODEL_KEY", "DO_MODEL_KEY"):
    if os.environ.get(var):
        record(f"env {var} (optional)", INFO, "DigitalOcean model key present (optional 4th model)")
        break
if os.environ.get("ATLAS_API_KEY"):
    record("env ATLAS_API_KEY (optional)", INFO, "present but NOT needed by Darwin (Atlas cluster-admin only)")

# --------------------------------------------------------------------------
# 2. MongoDB connection
# --------------------------------------------------------------------------
try:
    from pymongo import MongoClient
    uri = os.environ.get("MONGODB_URI", "")
    db = os.environ.get("MONGODB_DB", "darwin")
    if not uri:
        record("MongoDB ping", FAIL, "no MONGODB_URI to connect with")
    else:
        c = MongoClient(uri, serverSelectionTimeoutMS=6000)
        c.admin.command("ping")
        cols = c[db].list_collection_names()
        record("MongoDB ping", PASS,
               f"connected; db '{db}' has {len(cols)} collection(s): {cols if cols else '(none yet -- normal until first write)'}")
except Exception as e:
    record("MongoDB ping", FAIL,
           f"{type(e).__name__}: {e}  -> check Network Access IP allowlist and URL-encode any symbols in the password")

# --------------------------------------------------------------------------
# 3. Gemini API
# --------------------------------------------------------------------------
gemini_model = os.environ.get("DARWIN_GEMINI_TEST_MODEL", "gemini-3.5-flash")
try:
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    try:
        resp = client.models.generate_content(model=gemini_model, contents="Reply with the single word: ok")
        txt = (getattr(resp, "text", "") or "").strip()[:40]
        record(f"Gemini call ({gemini_model})", PASS, f"responded: {txt!r}")
    except Exception as e:
        record(f"Gemini call ({gemini_model})", FAIL,
               f"{type(e).__name__}: {e}  -> if NOT_FOUND/404 the model id is wrong; "
               f"valid GA ids: gemini-3.5-flash, gemini-3.1-pro, gemini-3.1-flash-lite")
except ImportError as e:
    record("Gemini SDK import", FAIL, f"google-genai not installed: {e}")
except Exception as e:
    record("Gemini client", FAIL, f"{type(e).__name__}: {e}")

# --------------------------------------------------------------------------
# 4. Voyage embeddings
# --------------------------------------------------------------------------
voyage_model = os.environ.get("DARWIN_VOYAGE_TEST_MODEL", "voyage-3.5")
try:
    import voyageai
    vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
    r = vo.embed(["preflight ping"], model=voyage_model, input_type="document")
    record(f"Voyage embed ({voyage_model})", PASS, f"got {len(r.embeddings[0])}-dim vector")
except ImportError as e:
    record("Voyage SDK import", FAIL, f"voyageai not installed: {e}")
except Exception as e:
    record("Voyage embed", FAIL, f"{type(e).__name__}: {e}  -> check VOYAGE_API_KEY / model name")

# --------------------------------------------------------------------------
# 5. OPTIONAL: darwin package -- fleet credential audit + Conductor + F1 probe
#    (Adjust import paths to match your tree; failures here are non-fatal.)
# --------------------------------------------------------------------------
print("\n--- optional darwin-package checks (adjust paths if they differ) ---")

# 5a. Fleet credential audit -- THE check that reveals models you can't authenticate
try:
    from darwin.routing.fleet import get_fleet
    for m in get_fleet():
        mid = getattr(m, "model_id", getattr(m, "id", "?"))
        key_env = getattr(m, "api_key_env", "") or ""
        if not key_env:
            record(f"fleet '{mid}'", INFO, "no api_key_env -- looks like a local/self-hosted endpoint (e.g. Modular MAX). "
                                           "Needs a running server URL, not just a key.")
        elif os.environ.get(key_env):
            record(f"fleet '{mid}' ({key_env})", PASS, "credential present")
        else:
            record(f"fleet '{mid}' ({key_env})", FAIL,
                   f"env var {key_env} is NOT set -- every agent routed to this model will fail to authenticate at runtime")
except Exception as e:
    record("fleet credential audit", WARN, f"could not import darwin.routing.fleet ({e})")

# 5b. Conductor import (the entry point)
try:
    from darwin.escalation.conductor import Conductor  # noqa: F401
    record("Conductor import", PASS, "darwin.escalation.conductor.Conductor")
except Exception as e:
    record("Conductor import", WARN, f"adjust path: {e}")

# 5c. F1 integration probe
f1_found = False
for modpath in ("darwin.problem.adapters.f1", "darwin.f1", "darwin.problem.f1", "darwin.f1.scorer"):
    try:
        __import__(modpath)
        record("F1 integration", PASS, f"found {modpath}")
        f1_found = True
        break
    except Exception:
        continue
if not f1_found:
    record("F1 integration", WARN,
           "no F1 problem/scorer module found inside darwin/. The brain currently solves supply-chain "
           "ProblemInstances; F1 needs an integration layer (calendar problem repr + a scorer that returns a "
           "B1 ScoreBreakdown + a calendar output schema + a reference optimum). See the notes in chat.")

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
fails = [r for r in results if r[1] == FAIL]
print("\n" + "=" * 64)
print(f"SUMMARY: {sum(r[1] == PASS for r in results)} pass | "
      f"{len(fails)} fail | {sum(r[1] == WARN for r in results)} warn")
if fails:
    print("\nBLOCKERS to fix before a real end-to-end solve:")
    for n, _, d in fails:
        print(f"  [FAIL] {n} -- {d}")
    sys.exit(1)
print("\nAll hard checks passed -- external services are wired. [OK]")