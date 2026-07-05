# 🏊‍♂️🚴‍♂️🏃‍♂️ Triathlon Race Engine

A local-first machine-learning system that predicts my finish time for my **first sprint triathlon
(750m swim / 20km bike / 5km run)** from my own Garmin training data — and reads out my actual fitness.

Built to apply the *Supervised ML: Regression and Classification* certification to real, personal data.

## What it does

- **Predicts my race finish time** (with a confidence range) by modelling each leg from my training.
- **Interprets every workout** — aerobic / threshold / intervals / brick / recovery — to use the right
  data and show my training distribution (the **classification** half).
- **Fits a pace-duration curve** per discipline from my best efforts (the **regression** half) to read
  off sustainable race pace.
- **Surfaces insights** Garmin doesn't: weakest leg, brick fatigue, "am I training smart?", fitness
  trend, and what-if pace sliders.

## How it works

```
Garmin → GarminDB (local SQLite) → data layer → per-workout classifier
       → pace-duration curves (×3) + brick calibration → Race Engine → insights → dashboard
```

## Stack

Python · GarminDB · pandas · scikit-learn · Streamlit

## Setup (data foundation — GRE-5)

Requires **Python 3.11+** (`brew install python@3.12` on macOS).

```bash
# 1. Environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure GarminDB (one-time)
#    Copy the example config into place, then edit it with your Garmin Connect
#    email/password and data start dates:
mkdir -p ~/.GarminDb
cp .venv/lib/python3.12/site-packages/garmindb/GarminConnectConfig.json.example \
   ~/.GarminDb/GarminConnectConfig.json

# 3. Full historical download + build the SQLite DBs (~/HealthData/DBs)
#    Run interactively — Garmin Connect will prompt for an MFA code on first
#    login; tokens are then cached locally so later syncs don't re-prompt.
garmindb_cli.py --all --download --import --analyze

# 4. Incremental sync (run anytime to pull only new activities)
garmindb_cli.py --all --download --import --analyze --latest
# or: python -m triathlon_engine.data.sync          # incremental
#     python -m triathlon_engine.data.sync full     # full re-download
```

Notes:
- Garmin archives intraday data older than ~6 months ("cold storage") — sync regularly to accumulate history.
- All Garmin data and credentials stay outside the repo (`~/HealthData`, `~/.GarminDb`) and are gitignored anyway.

## Status

In development — tracked in Linear (`GRE-5` … `GRE-12`). See [CLAUDE.md](./CLAUDE.md) for architecture,
build order, and working conventions.

## Privacy

Runs entirely on my machine. Garmin login and data stay local — **no personal data is committed to
this repo** (see `.gitignore`).
