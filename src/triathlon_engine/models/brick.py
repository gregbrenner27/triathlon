"""Brick fatigue calibration — run slowdown off the bike (GRE-9).

Method: fit fresh-run pace as a linear function of average HR (standalone
runs only), then for each GRE-7 ``brick``-labelled run compare its actual
pace to the fresh pace the model expects at the same HR. The median ratio is
the slowdown factor the Race Engine applies to the run leg.

With fewer than MIN_BRICKS_DATA brick runs, falls back to the
sports-science rule of thumb (~5% slower off the bike for novice sprint
triathletes) and says so — the output always states its source and sample
size. More brick sessions before race day sharpen this automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MIN_BRICKS_DATA = 3
RULE_OF_THUMB_FACTOR = 1.05
# Fresh-run comparison set: steady labels only — intervals/threshold pace
# doesn't reflect the pace~HR relation of a continuous run.
_FRESH_LABELS = {"aerobic-base", "recovery-easy"}
_MIN_FRESH_RUNS = 10


@dataclass
class BrickCalibration:
    factor: float  # APPLIED slowdown: multiply fresh run time by this
    measured_factor: float  # raw data measurement (can be < 1)
    source: str  # "data-derived" | "rule-of-thumb"
    n_bricks: int
    spread: tuple[float, float] = (np.nan, np.nan)  # IQR of per-session factors
    per_session: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""


def _fresh_pace_model(fresh: pd.DataFrame) -> tuple[LinearRegression, float]:
    """pace_s_per_km ~ avg_hr over standalone steady runs."""
    X = fresh[["avg_hr"]].to_numpy(dtype=float)
    y = fresh["pace_s_per_km"].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    resid_std = float((y - model.predict(X)).std(ddof=2))
    return model, resid_std


def brick_calibration(labeled: pd.DataFrame | None = None) -> BrickCalibration:
    """Compute the brick slowdown factor from labelled activities.

    ``labeled`` is the output of ``interpret.label_activities()`` (computed
    here if not supplied — pass it in when you already have it).
    """
    if labeled is None:
        from ..interpret import label_activities

        labeled = label_activities()

    runs = labeled[
        (labeled["sport"] == "running")
        & labeled["pace_s_per_km"].notna()
        & labeled["avg_hr"].notna()
    ]
    brick_runs = runs[runs["label"] == "brick"]
    fresh = runs[runs["label"].isin(_FRESH_LABELS)]
    # Distance-match the comparison: long easy runs carry HR drift that makes
    # short brisk runs look artificially fast at the same average HR.
    if not brick_runs.empty:
        med_d = brick_runs["distance"].median()
        matched = fresh[fresh["distance"].between(0.5 * med_d, 2.5 * med_d)]
        if len(matched) >= _MIN_FRESH_RUNS:
            fresh = matched

    note = (
        "More brick sessions before race day sharpen this estimate — "
        "each bike→run within 15 min counts automatically."
    )
    if len(brick_runs) < MIN_BRICKS_DATA or len(fresh) < _MIN_FRESH_RUNS:
        return BrickCalibration(
            factor=RULE_OF_THUMB_FACTOR,
            measured_factor=np.nan,
            source="rule-of-thumb",
            n_bricks=len(brick_runs),
            note=(
                f"only {len(brick_runs)} brick runs / {len(fresh)} fresh runs "
                f"on record — using the ~5% novice-sprint-tri rule. " + note
            ),
        )

    # Compare each brick run to the HR-matched fresh expectation, but only
    # within the HR range the fresh model actually saw (no extrapolation).
    model, _ = _fresh_pace_model(fresh)
    hr_lo, hr_hi = fresh["avg_hr"].min(), fresh["avg_hr"].max()
    rows = []
    for _, b in brick_runs.iterrows():
        if not (hr_lo <= b["avg_hr"] <= hr_hi):
            continue
        expected = float(model.predict([[float(b["avg_hr"])]])[0])
        rows.append(
            {
                "activity_id": b["activity_id"],
                "date": b["date"],
                "distance_km": b["distance"],
                "avg_hr": b["avg_hr"],
                "pace_s_per_km": b["pace_s_per_km"],
                "expected_fresh_pace": expected,
                "factor": b["pace_s_per_km"] / expected,
            }
        )
    per_session = pd.DataFrame(rows)
    if len(per_session) < MIN_BRICKS_DATA:
        return BrickCalibration(
            factor=RULE_OF_THUMB_FACTOR,
            measured_factor=np.nan,
            source="rule-of-thumb",
            n_bricks=len(per_session),
            per_session=per_session,
            note=(
                "brick runs fall outside the fresh-run HR range — "
                "using the ~5% rule. " + note
            ),
        )

    factors = per_session["factor"]
    measured = float(factors.median())
    # A measured factor < 1 means no penalty is detectable (Greg runs his
    # bricks brisk); applying a speed-up to the race run leg would
    # double-count effort, so the applied factor floors at 1.0.
    applied = max(measured, 1.0)
    if measured < 1.0:
        note = (
            f"no brick penalty detectable — brick runs were "
            f"{(1 - measured):.0%} faster than HR- and distance-matched fresh "
            f"runs across {len(per_session)} sessions; applying 1.00 (no "
            f"slowdown), race-day range still allows up to "
            f"{RULE_OF_THUMB_FACTOR:.2f}. " + note
        )
    return BrickCalibration(
        factor=applied,
        measured_factor=measured,
        source="data-derived",
        n_bricks=len(per_session),
        spread=(float(factors.quantile(0.25)), float(factors.quantile(0.75))),
        per_session=per_session,
        note=note,
    )
