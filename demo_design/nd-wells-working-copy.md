# Demo Summary: ND Wells Decline Curve App (working copy)

**Repo**: local working copy of `decline-curve-nd-wells`, no remote configured
**Status**: Decommissioned — kept for reference only
**Live deployment (historical)**: Databricks Apps (decommissioned)
**Relation to `decline-curve-nd-wells`**: a later, locally-modified checkout
of the same project (HEAD `f4aa516 "Complete data integration setup - Flask
app ready for real Delta tables, full upload script ready"`, vs.
`decline-curve-nd-wells` at `d8eb7c3`). Diverged with uncommitted local edits
and was never pushed anywhere — effectively a scratch/working duplicate.

## What it was

Same Databricks App as `decline-curve-nd-wells`: Flask backend + MapLibre GL
frontend for North Dakota Bakken well production data, with hyperbolic Arps
decline curve fitting, operator rankings, polygon filtering, and a "Genie"
chat placeholder. See `decline-curve-nd-wells/DEMO_SUMMARY.md` for the full
architecture writeup — it applies here too.

## What's different in this copy

- One commit ahead, focused on "data integration setup": real `requirements.txt`
  (instead of `pyproject.toml`), a `.code-workspace` VS Code workspace
  file, an additional notebook (`01_load_xlsx_to_delta.py`) for loading Excel
  files into Delta tables, and a sample data file `2023_12.xlsx`.
- `app.py` mock well/operator data was edited (different placeholder dataset
  than the other repo's version), still mock data — never connected to a real
  warehouse.
- Local-only `__pycache__` and `node_modules` build artifacts present.

## Why decommissioned

This was a diverging, unpushed working copy of `decline-curve-nd-wells` with
no unique reservoir-engineering logic — same Arps hyperbolic fit in
`js/dca.js`, superseded by `ressmith.primitives.decline` /
`ressmith.primitives.models.arps`. The only genuinely new artifact (a sample
production Excel file and a Delta-load notebook) was infrastructure for a
data pipeline that was never completed.

## If recreating this demo

Don't maintain two parallel checkouts. Follow the recommendations in
`decline-curve-nd-wells/DEMO_SUMMARY.md` — use `ressmith` for decline curve
math, consolidate the Flask entrypoints, and build the Delta table pipeline
(`01_load_xlsx_to_delta.py` is a reasonable starting point) before wiring up
the UI.
