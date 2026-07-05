"""Per-workout interpreter — session-type classifier (GRE-7).

Labels every activity with what kind of session it was, using the hierarchy
from the ticket:

1. bike→run back-to-back (within BRICK_GAP_MIN minutes) → ``brick``
2. structured-workout steps embedded in the FIT file → read intent directly
3. otherwise rule-based on features: pace CV, HR-zone distribution, lap
   structure, Training Effect, Z4+ excursions.

Rule-based by design (transparent, no training labels needed); upgradeable to
a trained classifier once hand labels exist.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from ..data.access import (
    ACTIVITIES_DB,
    get_activities,
    get_activity_meta,
    get_workout_steps,
)

LABELS = [
    "aerobic-base",
    "threshold-tempo",
    "intervals-vo2",
    "brick",
    "technique-drill",
    "recovery-easy",
]

BRICK_GAP_MIN = 15  # max minutes between bike end and run start
_EXCURSION_MIN_S = 45  # a Z4+ surge must last this long to count

# Tunable rule thresholds, calibrated against the feature distributions of
# Garmin's own trainingEffectLabel groups over Greg's history (2024-08→):
# RECOVERY sits at TE ≤ ~2.3 with ~zero Z4+ time (but plenty of Z3, so Z3 is
# not a recovery discriminator); TEMPO/LACTATE_THRESHOLD at Z4+ ≥ ~0.3 vs
# ~0.01 for AEROBIC_BASE. High aerobic TE alone does NOT imply threshold —
# long easy runs reach TE 3.5+.
_TH = {
    "recovery_max_te": 2.5,
    "recovery_max_moving_s": 60 * 60,
    "recovery_max_z4plus_pct": 0.05,
    "intervals_min_anaerobic_te": 1.0,
    "intervals_min_pace_cv": 0.28,
    "intervals_min_excursions": 3,
    "threshold_min_z4plus_pct": 0.30,
    "threshold_alt_te": 3.5,
    "threshold_alt_z4plus_pct": 0.20,
    "drill_max_te": 2.5,
    "drill_pace_vs_median": 1.15,
}


def _stream_features() -> pd.DataFrame:
    """Per-activity features from the per-second records, in one bulk query.

    Returns pace_cv (CV of nonzero speed) and z4_excursions (# of sustained
    surges above the activity's Z4 floor).
    """
    with sqlite3.connect(ACTIVITIES_DB) as con:
        recs = pd.read_sql_query(
            "select r.activity_id, r.hr, r.speed, a.hrz_4_hr "
            "from activity_records r join activities a using (activity_id)",
            con,
        )

    def per_activity(g: pd.DataFrame) -> pd.Series:
        speed = g["speed"].dropna()
        speed = speed[speed > 0]
        pace_cv = speed.std() / speed.mean() if len(speed) > 30 else np.nan
        excursions = 0
        if g["hrz_4_hr"].notna().any() and g["hr"].notna().any():
            hot = (g["hr"] >= g["hrz_4_hr"]).to_numpy()
            # count runs of consecutive True at least _EXCURSION_MIN_S long
            # (records are ~1/s)
            run_len = 0
            for flag in hot:
                run_len = run_len + 1 if flag else 0
                if run_len == _EXCURSION_MIN_S:
                    excursions += 1
        return pd.Series({"pace_cv": pace_cv, "z4_excursions": excursions})

    return recs.groupby("activity_id").apply(per_activity, include_groups=False)


def _lap_features() -> pd.DataFrame:
    """Lap-structure features: lap count and distance uniformity (CV)."""
    with sqlite3.connect(ACTIVITIES_DB) as con:
        laps = pd.read_sql_query(
            "select activity_id, distance from activity_laps", con
        )
    grouped = laps.groupby("activity_id")["distance"]
    out = pd.DataFrame(
        {
            "n_laps": grouped.size(),
            "lap_dist_cv": grouped.std() / grouped.mean(),
        }
    )
    return out


def _detect_bricks(acts: pd.DataFrame) -> pd.Series:
    """True where an activity is part of a brick (bike→run adjacency or
    explicit multisport sport)."""
    is_brick = pd.Series(False, index=acts.index)
    is_brick |= acts["sport"].eq("multisport")

    ordered = acts.sort_values("start_time")
    end_time = ordered["start_time"] + pd.to_timedelta(
        ordered["elapsed_s"], unit="s"
    )
    nxt = ordered.shift(-1)
    gap_min = (
        nxt["start_time"] - end_time
    ).dt.total_seconds() / 60.0
    pair = (
        ordered["sport"].eq("cycling")
        & nxt["sport"].eq("running")
        & gap_min.between(-1, BRICK_GAP_MIN)
    )
    brick_ids = set(ordered.loc[pair, "activity_id"]) | set(
        nxt.loc[pair, "activity_id"].dropna()
    )
    is_brick |= acts["activity_id"].isin(brick_ids)
    return is_brick


def _label_from_steps(steps: pd.DataFrame) -> str | None:
    """Map structured-workout intent to a label (ticket hierarchy rule 2)."""
    if steps.empty:
        return None
    intensities = steps["intensity"].str.lower()
    has_rest = intensities.str.contains("rest|recovery").any()
    n_active = intensities.str.contains("active|interval").sum()
    if has_rest and n_active >= 2:
        return "intervals-vo2"
    return "threshold-tempo"


def _classify_row(row: pd.Series, swim_median_pace: float) -> tuple[str, str]:
    """Feature rules (hierarchy rule 3). Returns (label, reason)."""
    def val(key: str) -> float:
        v = row.get(key)
        return 0.0 if v is None or pd.isna(v) else float(v)

    z3p = sum(val(f"hrz_{z}_pct") for z in (3, 4, 5))
    z4p = sum(val(f"hrz_{z}_pct") for z in (4, 5))
    te = val("training_effect")
    ana = val("anaerobic_training_effect")
    cv = row.get("pace_cv")
    exc = val("z4_excursions")

    if (
        ana >= _TH["intervals_min_anaerobic_te"]
        and exc >= _TH["intervals_min_excursions"]
    ) or (
        cv is not None
        and cv == cv
        and cv >= _TH["intervals_min_pace_cv"]
        and exc >= _TH["intervals_min_excursions"]
    ):
        return "intervals-vo2", (
            f"anaerobic TE {ana:.1f}, {exc:.0f} sustained Z4+ surges, "
            f"pace CV {cv if cv == cv else float('nan'):.2f}"
        )

    if z4p >= _TH["threshold_min_z4plus_pct"] or (
        te >= _TH["threshold_alt_te"] and z4p >= _TH["threshold_alt_z4plus_pct"]
    ):
        return "threshold-tempo", (
            f"{z4p:.0%} of time in Z4+, aerobic TE {te:.1f}"
        )

    if row["sport"] == "swimming" and (
        te <= _TH["drill_max_te"]
        and swim_median_pace == swim_median_pace
        and (row.get("pace_s_per_100m") or 0)
        > _TH["drill_pace_vs_median"] * swim_median_pace
    ):
        return "technique-drill", (
            f"swim {row.get('pace_s_per_100m', float('nan')):.0f}s/100m "
            f">{_TH['drill_pace_vs_median']:.2f}x median, TE {te:.1f}"
        )

    if (
        te < _TH["recovery_max_te"]
        and (row.get("moving_s") or 0) <= _TH["recovery_max_moving_s"]
        and z4p <= _TH["recovery_max_z4plus_pct"]
    ):
        return "recovery-easy", (
            f"TE {te:.1f}, {row.get('moving_s', 0) / 60:.0f} min, "
            f"{z4p:.0%} in Z4+"
        )

    return "aerobic-base", (
        f"steady: {z3p:.0%} above Z2, TE {te:.1f}, "
        f"pace CV {cv if cv is not None and cv == cv else float('nan'):.2f}"
    )


def label_activities(
    acts: pd.DataFrame | None = None, with_steps: bool = True
) -> pd.DataFrame:
    """Label every activity. Returns activities plus ``label``, ``reason``
    and the features used (``pace_cv``, ``z4_excursions``, ``n_laps``,
    ``lap_dist_cv``)."""
    if acts is None:
        acts = get_activities()
    acts = acts.merge(
        _stream_features(), left_on="activity_id", right_index=True, how="left"
    ).merge(_lap_features(), left_on="activity_id", right_index=True, how="left")

    swim = acts[acts["sport"] == "swimming"]
    swim_median_pace = swim["pace_s_per_100m"].median()

    # Parsing every FIT for workout steps is slow; Connect metadata says which
    # activities actually followed a structured workout (workoutId set).
    structured_ids: set[str] = set()
    if with_steps:
        meta = get_activity_meta()
        structured_ids = set(
            meta.loc[meta["workoutId"].notna(), "activityId"].astype(str)
        )

    is_brick = _detect_bricks(acts)
    labels, reasons = [], []
    for idx, row in acts.iterrows():
        if is_brick.loc[idx]:
            labels.append("brick")
            reasons.append("bike→run back-to-back (or multisport file)")
            continue
        step_label = (
            _label_from_steps(get_workout_steps(row["activity_id"]))
            if with_steps and str(row["activity_id"]) in structured_ids
            else None
        )
        if step_label:
            labels.append(step_label)
            reasons.append("structured workout steps in FIT")
            continue
        label, reason = _classify_row(row, swim_median_pace)
        labels.append(label)
        reasons.append(reason)

    acts["label"] = labels
    acts["reason"] = reasons
    return acts


def training_distribution(labeled: pd.DataFrame) -> pd.DataFrame:
    """Session counts, time share and easy/hard split per label."""
    hard = {"threshold-tempo", "intervals-vo2", "brick"}
    out = (
        labeled.groupby("label")
        .agg(sessions=("activity_id", "size"), hours=("moving_s", lambda s: s.sum() / 3600))
        .reset_index()
    )
    out["time_share"] = out["hours"] / out["hours"].sum()
    out["intensity"] = np.where(out["label"].isin(hard), "hard", "easy")
    return out.sort_values("sessions", ascending=False).reset_index(drop=True)
