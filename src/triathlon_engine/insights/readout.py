"""Fitness readout & insights layer (GRE-11).

Every insight derives from the models — nothing hardcoded. Each is one plain
sentence plus its supporting numbers, with honesty about data thinness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..engine.race_engine import RaceForecast, predict_race
from ..interpret import label_activities, training_distribution
from ..models.brick import brick_calibration
from ..models.pace_curves import RACE_DISTANCES_KM, run_effort_matrix

POLARIZED_EASY_SHARE = 0.80  # 80/20 benchmark
# Critical-speed fit uses efforts in the ~3–20 min window (the CS model's
# validity zone); for Greg that's 1–5 km.
_CS_TARGETS_KM = (1.0, 2.0, 3.0, 5.0)
_CS_WINDOW_WEEKS = 12
_CS_STEP_WEEKS = 4
_IMPROVEMENT = 0.05  # what-if sensitivity step: 5% per leg


@dataclass
class Insight:
    title: str
    sentence: str
    data: dict = field(default_factory=dict)
    caveat: str = ""


def critical_speed(efforts: pd.DataFrame) -> tuple[float, float]:
    """(CS m/s, D′ m) from the linear distance = CS·time + D′ model.

    Needs ≥3 distinct distances in the CS validity window; NaNs otherwise.
    """
    pts = efforts[efforts["distance_km"].isin(_CS_TARGETS_KM)]
    pts = pts.loc[pts.groupby("distance_km")["time_s"].idxmin()]
    if len(pts) < 3:
        return (np.nan, np.nan)
    X = pts[["time_s"]].to_numpy()
    y = (pts["distance_km"] * 1000).to_numpy()
    model = LinearRegression().fit(X, y)
    return float(model.coef_[0]), float(model.intercept_)


def cs_trend(
    window_weeks: int = _CS_WINDOW_WEEKS,
    step_weeks: int = _CS_STEP_WEEKS,
    min_n: int = 30,
) -> pd.DataFrame:
    """Run critical speed per rolling training block — the fitness trend.

    Blocks with fewer than ``min_n`` effort points produce junk CS fits
    (few hard efforts → degenerate regression) and are dropped.
    """
    matrix = run_effort_matrix(list(_CS_TARGETS_KM))
    if matrix.empty:
        return pd.DataFrame(columns=["block_end", "cs_kmh", "d_prime_m", "n"])
    end = matrix["start_time"].max()
    start = matrix["start_time"].min() + pd.Timedelta(weeks=window_weeks)
    rows = []
    for block_end in pd.date_range(start, end, freq=f"{step_weeks}W"):
        block = matrix[
            matrix["start_time"].between(
                block_end - pd.Timedelta(weeks=window_weeks), block_end
            )
        ]
        cs, d_prime = critical_speed(block)
        if cs == cs and len(block) >= min_n:
            rows.append(
                {
                    "block_end": block_end,
                    "cs_kmh": cs * 3.6,
                    "d_prime_m": d_prime,
                    "n": len(block),
                }
            )
    return pd.DataFrame(rows)


def weakest_leg(forecast: RaceForecast | None = None) -> Insight:
    """Which leg pays back training most: minutes saved by a 5% improvement,
    weighted by how uncertain the leg is."""
    forecast = forecast or predict_race()
    legs = {l.name: l for l in forecast.legs if l.name in ("Swim", "Bike", "Run")}
    gain = {name: l.time_s * _IMPROVEMENT for name, l in legs.items()}
    spread = {name: l.hi_s - l.lo_s for name, l in legs.items()}
    weakest = max(gain, key=lambda n: gain[n] + 0.25 * spread[n])
    return Insight(
        title="Weakest leg",
        sentence=(
            f"{weakest} is where training pays off most: 5% faster saves "
            f"{gain[weakest] / 60:.1f} min, and it carries the widest "
            f"uncertainty (±{spread[weakest] / 120:.1f} min)."
        ),
        data={
            "gain_s_per_5pct": gain,
            "range_width_s": spread,
        },
        caveat=(
            legs[weakest].note
            if legs[weakest].source != "curve"
            else ""
        ),
    )


def brick_reality_check() -> Insight:
    cal = brick_calibration()
    if cal.source == "data-derived" and cal.measured_factor < 1.0:
        sentence = (
            f"No brick slowdown is detectable in your {cal.n_bricks} brick "
            f"sessions — you actually ran {(1 - cal.measured_factor):.0%} "
            "faster off the bike than in matched fresh runs; the engine "
            "applies no penalty but allows up to 5% on race day."
        )
    elif cal.source == "data-derived":
        sentence = (
            f"Your run slows ~{(cal.factor - 1):.0%} off the bike "
            f"(measured across {cal.n_bricks} brick sessions)."
        )
    else:
        sentence = (
            f"Not enough brick sessions to measure your slowdown — assuming "
            f"the ~{(cal.factor - 1):.0%} novice rule until you log more."
        )
    return Insight(
        title="Brick reality-check",
        sentence=sentence,
        data={
            "measured_factor": cal.measured_factor,
            "applied_factor": cal.factor,
            "n_bricks": cal.n_bricks,
        },
        caveat=cal.note,
    )


def training_smart(labeled: pd.DataFrame | None = None) -> Insight:
    labeled = labeled if labeled is not None else label_activities()
    dist = training_distribution(labeled)
    easy = dist.loc[dist["intensity"] == "easy", "time_share"].sum()
    verdict = (
        "right in the polarized sweet spot"
        if abs(easy - POLARIZED_EASY_SHARE) <= 0.05
        else "more easy volume than the 80/20 benchmark"
        if easy > POLARIZED_EASY_SHARE
        else "harder than the 80/20 benchmark — watch recovery"
    )
    return Insight(
        title="Are you training smart?",
        sentence=(
            f"Your last-year split is {easy:.0%} easy / {1 - easy:.0%} hard — "
            f"{verdict}."
        ),
        data={"distribution": dist, "easy_share": easy},
    )


def fitness_trend() -> Insight:
    trend = cs_trend()
    if len(trend) < 2:
        return Insight(
            title="Fitness trend",
            sentence="Not enough run history to chart a critical-speed trend yet.",
            data={"trend": trend},
        )
    first, last = trend.iloc[0], trend.iloc[-1]
    months = (last["block_end"] - first["block_end"]).days / 30.4
    direction = "improved" if last["cs_kmh"] > first["cs_kmh"] else "declined"
    return Insight(
        title="Fitness trend (run critical speed)",
        sentence=(
            f"Your aerobic engine has {direction}: run critical speed went "
            f"{first['cs_kmh']:.1f} → {last['cs_kmh']:.1f} km/h over the last "
            f"{months:.0f} months (D′ finishing reserve now "
            f"{last['d_prime_m']:.0f} m)."
        ),
        data={"trend": trend},
        caveat="Run only — swim/bike best-effort history is too thin for a trend.",
    )


def what_if(
    swim_pace_delta_s_per_100m: float = 0.0,
    bike_speed_delta_kmh: float = 0.0,
    run_pace_delta_s_per_km: float = 0.0,
    t1_delta_s: float = 0.0,
    t2_delta_s: float = 0.0,
    forecast: RaceForecast | None = None,
) -> Insight:
    """Recompute the finish time from adjusted leg inputs (negative = faster)."""
    forecast = forecast or predict_race()
    legs = {l.name: l.time_s for l in forecast.legs}
    d = RACE_DISTANCES_KM

    swim = legs["Swim"] + swim_pace_delta_s_per_100m * d["swim"] * 10
    old_v = d["bike"] / (legs["Bike"] / 3600.0)
    bike = d["bike"] / max(old_v + bike_speed_delta_kmh, 1e-6) * 3600.0
    run = legs["Run"] + run_pace_delta_s_per_km * d["run"]
    t1 = max(legs["T1"] + t1_delta_s, 0.0)
    t2 = max(legs["T2"] + t2_delta_s, 0.0)

    new_finish = swim + t1 + bike + t2 + run
    delta = new_finish - forecast.finish_s
    changes = {
        "swim_pace_delta_s_per_100m": swim_pace_delta_s_per_100m,
        "bike_speed_delta_kmh": bike_speed_delta_kmh,
        "run_pace_delta_s_per_km": run_pace_delta_s_per_km,
        "t1_delta_s": t1_delta_s,
        "t2_delta_s": t2_delta_s,
    }
    described = ", ".join(f"{k} {v:+g}" for k, v in changes.items() if v)
    return Insight(
        title="What-if",
        sentence=(
            f"With {described or 'no changes'}: finish "
            f"{'improves' if delta < 0 else 'slips'} by "
            f"{abs(delta) / 60:.1f} min to "
            f"{int(new_finish // 3600)}:{int(new_finish % 3600 // 60):02d}:"
            f"{int(new_finish % 60):02d}."
        ),
        data={"new_finish_s": new_finish, "delta_s": delta, **changes},
    )


def fitness_readout() -> list[Insight]:
    """All insights, ready for the dashboard (GRE-12)."""
    labeled = label_activities()
    forecast = predict_race()
    return [
        weakest_leg(forecast),
        brick_reality_check(),
        training_smart(labeled),
        fitness_trend(),
        what_if(swim_pace_delta_s_per_100m=-20, forecast=forecast),
    ]
