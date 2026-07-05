# Triathlon Race Engine

A **local-first ML system** that predicts Greg's finish time for his **first sprint triathlon
(750m swim / 20km bike / 5km run)** from his own Garmin training data, plus a fitness readout and
coaching-style insights. Portfolio project demonstrating the *Supervised ML: Regression and
Classification* certification on real personal data.

Runs entirely on Greg's machine — Garmin login and data stay local, nothing stored externally.
Scoped to Greg for v1; designed cleanly enough to open up to other athletes later.

---

## Linear integration (read this before starting work)

- **Workspace / team:** `Greg Brenner` (key `GRE`)
- **Project:** `Triathlon Race Engine`
- **Tickets:** `GRE-5` → `GRE-12`
- Use the Linear MCP (`linear-server`) to read tickets and update status. Don't re-discover team/project IDs — they're above.

### Build order & dependencies (STRICT — this is a dependency chain, work sequentially)

```
GRE-5  Set up GarminDB + sync to local SQLite        (foundation — unblocks everything)
  └─ GRE-6  Data-access layer → pandas
       ├─ GRE-7  Per-workout interpreter (classifier)
       │    ├─ GRE-8  Pace-duration curves (regression ×3)   ─┐
       │    └─ GRE-9  Brick fatigue calibration               ─┤
       │                                                        └─ GRE-10  Race Engine (compose)
       │                                                              ├─ GRE-11  Insights & fitness readout
       │                                                              └─ GRE-12  Dashboard UI (Streamlit + shadcn)
```

Do **not** work tickets in parallel or out of order — later tickets build on earlier code/data.

---

## Workflow rules (per ticket)

1. **One ticket per session.** Work only the ticket the user names. When done, **stop and tell the user** before starting the next.
2. **Plan mode first.** Read the Linear ticket + this file, propose a plan, and wait for approval **before editing**.
3. **Build**, then **verify against the ticket's acceptance criteria** — that's the quality gate.
4. **Commit and push to `origin`** (`https://github.com/gregbrenner27/triathlon`). Solo project → committing straight to `main` is fine; use a ticket branch + PR only if the user asks.
5. **Set the Linear ticket to Done**, then stop.

### ⚠️ Manual gate — GRE-5 (Garmin login)
GRE-5 requires **Greg's** Garmin credentials + MFA code. Claude sets up GarminDB and hands Greg the
exact commands to run; **Greg** performs the login/sync interactively. All later tickets use data
already on disk, so Claude runs freely from GRE-6 onward.

---

## Tech stack

- **Python** (3.11+)
- **[GarminDB](https://github.com/tcgoetz/GarminDB)** — pulls Garmin history into local SQLite (data layer)
- **pandas** — data wrangling
- **scikit-learn / numpy** — regression + classification models
- **Streamlit** — dashboard UI (style with the **shadcn MCP** for a polished, non-boring look)
- **matplotlib / plotly** — charts

## Repo structure (folder → ticket)

```
src/triathlon_engine/
  data/       # GRE-5 sync helpers, GRE-6 data-access layer (activities/laps/records → pandas)
  interpret/  # GRE-7 per-workout session-type classifier
  models/     # GRE-8 pace-duration curves, GRE-9 brick calibration
  engine/     # GRE-10 Race Engine (composition → finish time + range)
  insights/   # GRE-11 fitness readout + insights
app/          # GRE-12 Streamlit dashboard
notebooks/    # exploration / sanity-checking (not the deliverable)
data/         # local Garmin data — GITIGNORED, never commit
```

## 🔒 Data & privacy (non-negotiable)

- **Never commit** Garmin data, exported files, SQLite DBs, FIT/JSON files, or auth tokens.
- Garmin credentials/tokens live in the user's home dir (GarminDB default), never in the repo.
- `.gitignore` already excludes `data/`, `*.db`, `*.sqlite`, `*.fit`, `.env`, etc. — keep it that way.

## Key design decisions (already made — don't relitigate)

- **Data layer:** GarminDB (persistent local SQLite), not live API pulls — respects Garmin rate limits, richer history.
- **Prediction method:** power-law / Riegel pace-duration curves on *best efforts*, NOT naive whole-session averaging or raw Critical Speed (CS overpredicts efforts > ~20 min; the bike leg ~35–45 min sits in that zone).
- **Session typing:** rule-based + file-native signals first (multisport tag → brick; structured-workout steps; else pace-variability + HR-zones + Training-Effect features). Upgradeable to a trained classifier.
- **Watch baseline:** Forerunner 255 Music — has VO2max, race predictor, Training Readiness/Status, HRV status, Body Battery, aerobic/anaerobic Training Effect; no running power / endurance / hill score.
- **Data caveat:** ~1 yr run history, ~3 mo swim/bike → swim/bike models are thin; output **ranges, not false precision**.
