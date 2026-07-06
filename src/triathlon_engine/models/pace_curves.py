"""Pace-duration curves — sustainable pace per discipline (GRE-8).

Method: extract best sustained efforts across all sessions, then fit a
power-law / Riegel curve ``log(time) = a + b·log(distance)`` — plain linear
regression in log-log space. Race pace is read off the curve at the sprint
distances (750m swim / 20km bike / 5km run) with a t-based prediction
interval, because with thin data honesty beats precision.

Best-effort sources per discipline:
- run:  per-second records (cumulative distance) — fastest window per target
- swim: FIT set structure via ``get_swim_sets`` (per-second swim records
  carry no distance), fastest set per distance
- bike: none — Greg's rides to date are all indoor with no speed/distance;
  ``fit_bike`` returns an explicit no-data result rather than a fake curve.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from ..data.access import ACTIVITIES_DB, get_activities, get_swim_sets

RACE_DISTANCES_KM = {"swim": 0.75, "bike": 20.0, "run": 5.0}

# Longest target 10k = 2× race distance; beyond that Greg's history only has
# easy long runs, whose submaximal "best efforts" drag the curve upward.
RUN_TARGETS_KM = [0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
MIN_POINTS_TO_FIT = 4
_MIN_SWIM_SET_M = 50  # ignore micro-sets; too noisy
# Sets (or whole swims) faster than this are watch artifacts — mis-configured
# pool length (50m instead of 25m halves apparent pace) or missed lengths.
# Greg's fastest genuine hard sets are ~110 s/100m.
_SWIM_PACE_PLAUSIBLE = (90.0, 400.0)  # s/100m


@dataclass
class PaceCurve:
    """A fitted log-log pace-duration curve for one discipline."""

    sport: str
    ok: bool
    message: str = ""
    slope: float = np.nan  # Riegel exponent b
    intercept: float = np.nan
    resid_std: float = np.nan
    r2: float = np.nan
    n_points: int = 0
    points: pd.DataFrame = field(default_factory=pd.DataFrame)

    def predict_time_s(self, distance_km: float) -> float:
        """Predicted sustainable time (s) to cover distance_km."""
        if not self.ok:
            return np.nan
        return float(
            np.exp(self.intercept + self.slope * np.log(distance_km))
        )

    def prediction_interval_s(
        self, distance_km: float, level: float = 0.9
    ) -> tuple[float, float]:
        """(low, high) time bounds at ``level`` from the t-based interval on
        log-time residuals."""
        if not self.ok:
            return (np.nan, np.nan)
        mid = self.predict_time_s(distance_km)
        dof = max(self.n_points - 2, 1)
        x = np.log(self.points["distance_km"].to_numpy())
        x_new = np.log(distance_km)
        se = self.resid_std * np.sqrt(
            1 + 1 / self.n_points + (x_new - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
        )
        t = stats.t.ppf(0.5 + level / 2, dof)
        return (mid * np.exp(-t * se), mid * np.exp(t * se))


def _fit_power_law(points: pd.DataFrame, sport: str) -> PaceCurve:
    """log-log linear regression over (distance_km, time_s) best efforts."""
    points = points.dropna(subset=["distance_km", "time_s"])
    points = points[(points["distance_km"] > 0) & (points["time_s"] > 0)]
    if len(points) < MIN_POINTS_TO_FIT:
        return PaceCurve(
            sport=sport,
            ok=False,
            message=(
                f"only {len(points)} usable best-effort points "
                f"(need ≥{MIN_POINTS_TO_FIT})"
            ),
            points=points,
        )
    X = np.log(points[["distance_km"]].to_numpy())
    y = np.log(points["time_s"].to_numpy())
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    resid = y - pred
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else np.nan
    return PaceCurve(
        sport=sport,
        ok=True,
        slope=float(model.coef_[0]),
        intercept=float(model.intercept_),
        resid_std=float(resid.std(ddof=2)) if len(points) > 2 else 0.0,
        r2=float(r2),
        n_points=len(points),
        points=points.reset_index(drop=True),
    )


def run_effort_matrix(targets_km: list[float] = RUN_TARGETS_KM) -> pd.DataFrame:
    """Per-activity fastest sustained effort per target distance.

    Two-pointer sweep over each run's cumulative-distance stream: for each
    target, the minimum time of any contiguous stretch covering it. One row
    per (activity, target); includes ``start_time`` so callers can slice by
    training block (GRE-11 fitness trend).
    """
    runs = get_activities("run")
    starts = dict(
        zip(runs["activity_id"].astype(str), runs["start_time"], strict=True)
    )
    ids = tuple(starts)
    with sqlite3.connect(ACTIVITIES_DB) as con:
        recs = pd.read_sql_query(
            "select activity_id, timestamp, distance from activity_records "
            f"where activity_id in ({','.join('?' * len(ids))}) "
            "and distance is not null order by activity_id, record",
            con,
            params=ids,
            parse_dates=["timestamp"],
        )

    rows = []
    for aid, g in recs.groupby("activity_id"):
        dist = g["distance"].to_numpy()  # cumulative km
        t = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
        if len(dist) < 30:
            continue
        for target in targets_km:
            if dist[-1] < target:
                continue
            lo = 0
            fastest = np.inf
            for hi in range(len(dist)):
                while dist[hi] - dist[lo] >= target:
                    fastest = min(fastest, t[hi] - t[lo])
                    lo += 1
            if np.isfinite(fastest):
                rows.append(
                    {
                        "activity_id": aid,
                        "start_time": starts[aid],
                        "distance_km": target,
                        "time_s": fastest,
                    }
                )
    return pd.DataFrame(rows)


def run_best_efforts(targets_km: list[float] = RUN_TARGETS_KM) -> pd.DataFrame:
    """Fastest sustained effort per target distance across all runs."""
    matrix = run_effort_matrix(targets_km)
    if matrix.empty:
        return matrix
    idx = matrix.groupby("distance_km")["time_s"].idxmin()
    return (
        matrix.loc[idx, ["distance_km", "time_s", "activity_id"]]
        .sort_values("distance_km")
        .reset_index(drop=True)
    )


def swim_best_efforts() -> pd.DataFrame:
    """Fastest set per distinct set distance across all swims.

    Technique/drill sessions are excluded via the GRE-7 labels — drill sets
    (kick, paddles) aren't representative efforts.
    """
    from ..interpret import label_activities

    swims = label_activities(get_activities("swim"), with_steps=False)
    swims = swims[swims["label"] != "technique-drill"]
    # Whole activities with implausible average pace have a mis-configured
    # pool length — every set in them is wrong, so drop at activity level.
    lo, hi = _SWIM_PACE_PLAUSIBLE
    swims = swims[swims["pace_s_per_100m"].between(lo, hi)]
    sets = []
    for aid in swims["activity_id"].astype(str):
        s = get_swim_sets(aid)
        if not s.empty:
            s = s.assign(activity_id=aid)
            sets.append(s)
    if not sets:
        return pd.DataFrame(columns=["distance_km", "time_s", "activity_id"])
    all_sets = pd.concat(sets, ignore_index=True)
    all_sets = all_sets[all_sets["distance_m"] >= _MIN_SWIM_SET_M]
    all_sets = all_sets[all_sets["pace_s_per_100m"].between(lo, hi)]
    idx = all_sets.groupby("distance_m")["time_s"].idxmin()
    best = all_sets.loc[idx].copy()
    best["distance_km"] = best["distance_m"] / 1000.0
    return best[["distance_km", "time_s", "activity_id"]].sort_values(
        "distance_km"
    ).reset_index(drop=True)


def fit_run() -> PaceCurve:
    return _fit_power_law(run_best_efforts(), "run")


def fit_swim() -> PaceCurve:
    return _fit_power_law(swim_best_efforts(), "swim")


def fit_bike() -> PaceCurve:
    """All rides on record are indoor with no speed/distance stream, so a
    pace-duration curve cannot be fit. Returns a no-data curve; the race
    engine must take the bike leg from another estimate (manual pace range or
    future outdoor rides)."""
    bikes = get_activities("bike")
    with_pace = bikes[bikes["distance"] > 0]
    if with_pace.empty:
        return PaceCurve(
            sport="bike",
            ok=False,
            message=(
                f"{len(bikes)} rides on record, none with speed/distance "
                "(all indoor without a sensor) — record outdoor rides or add "
                "a speed sensor to enable this curve"
            ),
        )
    # If outdoor rides appear later, treat each ride's (distance, moving
    # time) as an effort point; per-second refinement can come later.
    points = pd.DataFrame(
        {
            "distance_km": with_pace["distance"],
            "time_s": with_pace["moving_s"],
            "activity_id": with_pace["activity_id"].astype(str),
        }
    )
    return _fit_power_law(points, "bike")


def race_predictions(level: float = 0.9) -> pd.DataFrame:
    """Predicted leg time + pace + uncertainty at sprint-race distances.

    Legs with a fitted curve use it (``source='curve'``). The bike leg falls
    back to the physics estimate from Greg's self-reported indoor watts
    (``source='power-model'``) until outdoor ride data exists.
    """
    rows = []
    for sport, fit_fn in {"swim": fit_swim, "bike": fit_bike, "run": fit_run}.items():
        curve = fit_fn()
        d = RACE_DISTANCES_KM[sport]
        mid = curve.predict_time_s(d)
        lo, hi = curve.prediction_interval_s(d, level)
        source, note = "curve", curve.message
        ok = curve.ok
        if sport == "bike" and not curve.ok:
            from .bike_estimate import bike_leg_estimate

            est = bike_leg_estimate(d)
            mid, lo, hi = est.time_s, est.time_lo_s, est.time_hi_s
            source = est.source
            note = f"{curve.message}; using power-model estimate meanwhile"
            ok = True
        pace = (
            mid / (d * 10) if sport == "swim" else mid / d
        )  # s/100m or s/km
        rows.append(
            {
                "sport": sport,
                "distance_km": d,
                "ok": ok,
                "source": source,
                "time_s": mid,
                "time_lo_s": lo,
                "time_hi_s": hi,
                "pace": pace,
                "pace_unit": "s/100m" if sport == "swim" else "s/km",
                "n_points": curve.n_points,
                "r2": curve.r2,
                "riegel_exponent": curve.slope,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def plot_curves(out_dir: str = "data/reports") -> list[str]:
    """Scatter best efforts + fitted curve per discipline → PNGs (gitignored
    dir — plots contain personal training data). Returns file paths."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for curve in (fit_swim(), fit_run(), fit_bike()):
        if not curve.ok:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        pts = curve.points
        ax.scatter(pts["distance_km"], pts["time_s"] / 60, zorder=3)
        xs = np.geomspace(pts["distance_km"].min(), max(
            pts["distance_km"].max(), RACE_DISTANCES_KM[curve.sport]
        ), 100)
        mid = [curve.predict_time_s(x) / 60 for x in xs]
        band = [curve.prediction_interval_s(x) for x in xs]
        ax.plot(xs, mid, label=f"fit (b={curve.slope:.3f}, R²={curve.r2:.3f})")
        ax.fill_between(
            xs, [b[0] / 60 for b in band], [b[1] / 60 for b in band], alpha=0.2,
            label="90% prediction interval",
        )
        d_race = RACE_DISTANCES_KM[curve.sport]
        ax.axvline(d_race, ls="--", c="gray", label=f"race {d_race} km")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distance (km)")
        ax.set_ylabel("time (min)")
        ax.set_title(f"{curve.sport}: pace-duration curve ({curve.n_points} best efforts)")
        ax.legend()
        path = out / f"pace_curve_{curve.sport}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths
