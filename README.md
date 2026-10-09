# SkyCity Channel Atlas — Streamlit dashboard

Order Channel Performance & Market Share Analytics for SkyCity Auckland restaurants & bars.
Dark "night harbour" design with skyline imagery, seven clearly separated tabs, and a filter bar above the tabs (no sidebar).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tabs
Channel overview · Geography · Cuisine & segment · Dependency risk · Profitability · Outlook · Data & methodology

## Filter bar (above the tabs)
Channel view toggle · Subregion / Cuisine / Segment filters with a live restaurant count and Reset button · collapsible **Settings** panel (data source, risk rules, urban-core definition).
Default risk rule: High risk = Uber Eats + DoorDash ≥ 70% of orders, or any one platform ≥ 50%.

## Adding new data
* Drop any CSV with the same columns into `data/` — files are merged (newest row wins per RestaurantID).
* Or use the Settings-panel uploader (replace, or append new entries). Try `samples/sample_new_entries.csv`.
* New cuisines / subregions / segments appear automatically in filters and charts.

## Layout
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (styling, tabs, charts) |
| `analytics.py` | Data cleaning, KPIs, risk flags, profit and projection logic (no UI code) |
| `.streamlit/config.toml` | Dark theme |
| `assets/` | Background and hero images |
| `tests/` | `python tests/test_analytics.py` (logic), `python tests/smoke_app.py` (UI wiring smoke test) |

Shares are always recomputed from order counts; the supplied `InStoreShare` column is not trusted.
