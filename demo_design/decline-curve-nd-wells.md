# Demo Summary: ND Wells Decline Curve Analysis App

**Repo**: `decline-curve-nd-wells` (github.com/k-jones_data/decline-curve-nd-wells)
**Status**: Decommissioned — kept for reference only
**Live deployment (historical)**: Databricks Apps (decommissioned)

## What it was

A Databricks App (Flask backend + static JS frontend) for exploring North Dakota
oil & gas well production data:

- **Interactive map** (MapLibre GL) of well locations, colored by operator/formation
- **Decline curve analysis** — hyperbolic Arps fit per well, plotted against
  historical production
- **Operator performance rankings** — cumulative oil, well counts, average
  initial rate (qi) by operator
- **Geographic polygon filtering** for selecting wells by area
- **"Genie" AI chat** placeholder for natural-language queries over the data
- **Data export** capability

## Architecture

- `app.py` / `main.py` / `run.py` — three overlapping Flask entrypoints (never
  consolidated) serving `/api/wells`, `/api/operators`, `/api/wells/<uwi>`
- `databricks_query.py` — Databricks SQL warehouse client; fell back to
  hardcoded mock data because warehouse SQL execution was never wired up
- `js/dca.js` — client-side Arps hyperbolic decline fit (grid-search +
  coordinate-descent refinement) and cumulative-production formulas
- `js/map.js`, `js/operators.js`, `js/charting.js`, `js/chat.js` — frontend modules
- `notebooks/process-production-data.py`, `process-workspace-data.py` —
  Databricks notebooks intended to load Excel production files (from a
  Databricks Volume) into Delta tables (catalog/schema)
- `deploy.sh`, `start-dev-sync.sh`, `sync-local-to-databricks.sh`,
  `upload-*.sh` — Databricks Apps deploy/sync tooling

## Status at decommission

- App deployed and running on Databricks Apps, but **all served data was
  hardcoded mock JSON** (a handful of fake Bakken wells like "BAKKEN 1H",
  operators "Continental Resources", "Whiting Petroleum", etc.) — the Delta
  table integration was never completed.
- ~107 Excel production files (~274MB) were staged for upload but not fully
  processed.
- Three duplicate Flask entrypoints existed (`app.py`, `main.py`, `run.py`)
  from iterative debugging, never cleaned up.

## Why decommissioned

Maintaining a live Databricks Apps deployment + sync tooling for a demo that
never got past mock data wasn't sustainable. The core algorithmic value —
Arps hyperbolic decline curve fitting (grid-search + refinement,
qi/Di/b parameters, cumulative production formulas for b≠1 and harmonic
b=1 cases) — duplicates what already exists, more rigorously (scipy-based
optimization, full model suite), in **`ressmith.primitives.decline`** and
**`ressmith.primitives.models.arps`**. No unique reservoir-engineering logic
was lost.

## If recreating this demo

1. Use `ressmith` for the actual decline curve fitting (Python, server-side)
   instead of reimplementing Arps math in `js/dca.js`.
2. Pick one Flask entrypoint (consolidate `app.py`/`main.py`/`run.py`).
3. Wire `databricks_query.py` to a real SQL warehouse / Delta table before
   building UI around it — don't ship mock data as a placeholder for
   production data.
4. Reuse `js/map.js` / `js/charting.js` / `js/operators.js` for the frontend —
   those are generic visualization modules independent of the decline math.
