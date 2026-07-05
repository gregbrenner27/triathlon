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

## Status

In development — tracked in Linear (`GRE-5` … `GRE-12`). See [CLAUDE.md](./CLAUDE.md) for architecture,
build order, and working conventions.

## Privacy

Runs entirely on my machine. Garmin login and data stay local — **no personal data is committed to
this repo** (see `.gitignore`).
