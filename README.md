# Lebronsseiur (Darwin + F1 Visualization)

Monorepo for the AIEWF hackathon submission:

- **`darwin/`** — Python optimization brain (F1 calendar scoring, multi-agent evolution, WebSocket bridge)
- **`frontend/`** — Next.js app (speech intake → agent network → side-by-side F1 flight map)

## Quick start (demo)

### 1. Backend (Darwin)

```bash
cd darwin
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m pytest           # optional: verify backend (~4s)
```

Start the WebSocket bridge (streams evolution events to the UI):

```bash
cd darwin
source venv/bin/activate
# Demo mode (no API keys): recorded replay
DARWIN_DEMO=1 python -m darwin.observability.serve_frontend
# Live F1 solve (requires configured model keys in darwin/.env)
python -m darwin.observability.serve_frontend
```

Serves on **`ws://localhost:8765`**.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
```

In `.env.local`, set:

```bash
NEXT_PUBLIC_AGENT_RUN_WS=ws://localhost:8765   # live bron; omit for built-in simulation
```

```bash
npm run dev
```

Open **http://localhost:3000**

| Route | What it shows |
|-------|----------------|
| `/` | Speech / voice intake (landing) |
| `/agents` | Live agent topology evolution |
| `/flight` | Side-by-side current vs proposed F1 calendar |

## Repo layout

```
darwin/           # Python backend (B1–B8)
frontend/         # Next.js visualization app
requirements.txt  # Root pytest deps (mirrors darwin/requirements.txt)
pyproject.toml    # Darwin package metadata
conftest.py       # Pytest configuration
```

## Tests

```bash
# Backend (from repo root)
pip install -r requirements.txt
python -m pytest

# Frontend stream verifier (with serve_frontend running)
cd frontend && node scripts/verify-darwin-stream.mjs
```

## Environment variables

- **Backend:** copy `darwin/.env.example` → `darwin/.env` (API keys for live runs; not required for `DARWIN_DEMO=1`)
- **Frontend:** copy `frontend/.env.example` → `frontend/.env.local` (LiveKit + DigitalOcean for voice; WebSocket URL for agents)

Never commit `.env` or `.env.local` files.

## F1 calendar data

Sample run artifacts live in `frontend/public/data/` (`f1_baseline.json`, `f1_optimized.json`, `f1_run.json`).

## Architecture

Darwin emits events → `darwin/observability/frontend_bridge.py` translates them to the frontend `RunEvent` protocol → `frontend/src/lib/agent-run.ts` reducer drives the agents page. The flight map visualizes baseline vs optimized calendar routes with cumulative carbon, cost, and revenue counters.

See `darwin/observability/CONTRACT.md` and phase `CONTRACT.md` files under `darwin/` for backend design details.
