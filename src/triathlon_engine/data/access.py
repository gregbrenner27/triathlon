"""Data-access layer: GarminDB SQLite → tidy pandas DataFrames (GRE-6).

Downstream code (interpreter, models, engine) consumes these loaders and never
touches SQLite directly. Conventions:

- distances km, speeds km/h, ascent/descent m (GarminDB stores metric per config)
- pace exposed as sec/km (run/bike) and sec/100m (swim)
- durations exposed as float seconds (``*_s`` columns)
- missing metrics are NaN, never an exception
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .sync import DB_DIR, HEALTH_DATA_DIR

ACTIVITIES_DB = DB_DIR / "garmin_activities.db"
FIT_DIR = HEALTH_DATA_DIR / "FitFiles" / "Activities"

# Garmin FIT uses 127 as the "invalid" sentinel for temperature bytes.
_TEMP_SENTINEL = 127.0

_SPORT_ALIASES = {
    "run": "running",
    "running": "running",
    "bike": "cycling",
    "cycling": "cycling",
    "swim": "swimming",
    "swimming": "swimming",
    "multisport": "multisport",
}


def _connect() -> sqlite3.Connection:
    if not ACTIVITIES_DB.exists():
        raise FileNotFoundError(
            f"{ACTIVITIES_DB} not found — run the GarminDB sync first (see README)."
        )
    return sqlite3.connect(ACTIVITIES_DB)


def _seconds(series: pd.Series) -> pd.Series:
    """'HH:MM:SS.ffffff' strings → float seconds (NaN-safe)."""
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def _clean_temperature(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if "temperature" in col:
            df[col] = df[col].replace(_TEMP_SENTINEL, np.nan)
    return df


def _derive_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Add pace_s_per_km / pace_s_per_100m.

    Prefer Garmin's avg_speed (km/h): for pool swims it excludes rest between
    sets, whereas GarminDB's moving_time includes it. Fall back to
    moving_s/distance when avg_speed is missing.
    """
    speed = df["avg_speed"].where(df["avg_speed"] > 0)
    dist = df["distance"].where(df["distance"] > 0)
    fallback = df["moving_s"] / dist
    df["pace_s_per_km"] = (3600.0 / speed).fillna(fallback)
    df["pace_s_per_100m"] = df["pace_s_per_km"] / 10.0
    return df


def normalize_sport(sport: str) -> str:
    """Map friendly names (run/bike/swim) to GarminDB sport values."""
    try:
        return _SPORT_ALIASES[sport.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown sport {sport!r}; expected one of {sorted(set(_SPORT_ALIASES))}"
        ) from None


def get_activities(sport: str | None = None) -> pd.DataFrame:
    """One row per activity, newest first.

    Adds derived columns: ``elapsed_s``, ``moving_s``, ``pace_s_per_km``,
    ``pace_s_per_100m``, ``hrz_{1..5}_s`` and ``hrz_{1..5}_pct`` (share of
    in-zone time), plus run-specific fields (vo2_max, cadence detail) where
    GarminDB has them.
    """
    query = (
        "select a.*, s.vo2_max, s.avg_steps_per_min, s.max_steps_per_min, "
        "s.avg_step_length "
        "from activities a left join steps_activities s using (activity_id)"
    )
    with _connect() as con:
        df = pd.read_sql_query(query, con, parse_dates=["start_time", "stop_time"])

    df["elapsed_s"] = _seconds(df["elapsed_time"])
    df["moving_s"] = _seconds(df["moving_time"])
    for z in range(1, 6):
        df[f"hrz_{z}_s"] = _seconds(df[f"hrz_{z}_time"])
    zone_cols = [f"hrz_{z}_s" for z in range(1, 6)]
    zone_total = df[zone_cols].sum(axis=1)
    for z in range(1, 6):
        df[f"hrz_{z}_pct"] = np.where(
            zone_total > 0, df[f"hrz_{z}_s"] / zone_total, np.nan
        )

    df = _derive_pace(df)
    df = _clean_temperature(df)
    df["date"] = df["start_time"].dt.date

    if sport is not None:
        df = df[df["sport"] == normalize_sport(sport)]
    return df.sort_values("start_time", ascending=False).reset_index(drop=True)


def get_laps(activity_id: str) -> pd.DataFrame:
    """Per-lap breakdown for one activity, in lap order."""
    with _connect() as con:
        df = pd.read_sql_query(
            "select * from activity_laps where activity_id = ? order by lap",
            con,
            params=(str(activity_id),),
            parse_dates=["start_time", "stop_time"],
        )
    df["elapsed_s"] = _seconds(df["elapsed_time"])
    df["moving_s"] = _seconds(df["moving_time"])
    df = _derive_pace(df)
    return _clean_temperature(df)


def get_records(activity_id: str) -> pd.DataFrame:
    """Per-second stream for one activity.

    Columns include ``timestamp``, ``elapsed_s`` (since first record), ``hr``,
    ``speed`` (km/h), ``pace_s_per_km``, ``cadence``, ``altitude``,
    ``distance`` (cumulative km), lat/long.
    """
    with _connect() as con:
        df = pd.read_sql_query(
            "select * from activity_records where activity_id = ? order by record",
            con,
            params=(str(activity_id),),
            parse_dates=["timestamp"],
        )
    if not df.empty:
        df["elapsed_s"] = (
            df["timestamp"] - df["timestamp"].iloc[0]
        ).dt.total_seconds()
        speed = df["speed"].where(df["speed"] > 0)
        df["pace_s_per_km"] = 3600.0 / speed
    return _clean_temperature(df)


def get_activity_meta(activity_id: str | None = None) -> pd.DataFrame:
    """Garmin Connect summary metadata (one row per activity) from the JSON
    files GarminDB downloads alongside the FIT files.

    Exposes fields the SQLite schema drops — notably ``trainingEffectLabel``
    (Garmin's own session-type call: AEROBIC_BASE, RECOVERY, TEMPO,
    LACTATE_THRESHOLD, SPEED) and ``workoutId`` (set only when a structured
    workout was followed).
    """
    fields = [
        "activityId",
        "activityName",
        "startTimeLocal",
        "trainingEffectLabel",
        "aerobicTrainingEffectMessage",
        "anaerobicTrainingEffectMessage",
        "workoutId",
        "vO2MaxValue",
        "avgStrokes",
        "poolLength",
        "lapCount",
    ]
    paths = (
        [FIT_DIR / f"activity_{activity_id}.json"]
        if activity_id is not None
        else sorted(FIT_DIR.glob("activity_*.json"))
    )
    rows = []
    for path in paths:
        if not path.exists() or path.name.startswith("activity_details_"):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        rows.append({f: data.get(f) for f in fields})
    df = pd.DataFrame(rows, columns=fields)
    df["activityId"] = df["activityId"].astype("string")
    return df


def get_swim_sets(activity_id: str) -> pd.DataFrame:
    """Swim sets (active laps) parsed from the activity FIT file.

    GarminDB's activity_laps rows are empty for pool swims, but the FIT lap
    messages carry the real set structure: distance (km), swim time, lengths,
    stroke. Rest laps (zero distance) are dropped. Columns: ``set`` (order),
    ``distance_m``, ``time_s``, ``n_lengths``, ``pace_s_per_100m``.
    """
    columns = ["set", "distance_m", "time_s", "n_lengths", "pace_s_per_100m"]
    fit_path = FIT_DIR / f"{activity_id}_ACTIVITY.fit"
    if not fit_path.exists():
        return pd.DataFrame(columns=columns)

    from fitfile import File  # deferred: fitfile import is slow

    try:
        fit = File(str(fit_path))
    except Exception:
        return pd.DataFrame(columns=columns)
    rows = []
    for msg_type in fit.message_types:
        if not str(msg_type).endswith(".lap"):
            continue
        for i, msg in enumerate(fit[msg_type]):
            f = msg.fields
            dist_km = f.get("total_distance") or 0.0
            timer = f.get("total_timer_time")
            time_s = (
                timer.hour * 3600 + timer.minute * 60 + timer.second
                + timer.microsecond / 1e6
                if timer is not None
                else np.nan
            )
            if dist_km <= 0 or not time_s or time_s != time_s:
                continue  # rest lap or unusable
            dist_m = dist_km * 1000.0
            rows.append(
                {
                    "set": i,
                    "distance_m": dist_m,
                    "time_s": time_s,
                    "n_lengths": f.get("num_active_lengths"),
                    "pace_s_per_100m": time_s / (dist_m / 100.0),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def get_workout_steps(activity_id: str) -> pd.DataFrame:
    """Structured-workout steps embedded in the activity FIT file.

    Returns an empty DataFrame when the activity wasn't a structured workout
    (the common case — free workouts carry no workout/workout_step messages).
    """
    columns = ["step", "intensity", "duration_type", "duration_value", "target_type"]
    fit_path = FIT_DIR / f"{activity_id}_ACTIVITY.fit"
    if not fit_path.exists():
        return pd.DataFrame(columns=columns)

    from fitfile import File  # deferred: fitfile import is slow

    try:
        fit = File(str(fit_path))
    except Exception:
        return pd.DataFrame(columns=columns)
    rows = []
    for msg_type in fit.message_types:
        if "workout_step" not in str(msg_type):
            continue
        for i, msg in enumerate(fit[msg_type]):
            f = msg.fields
            rows.append(
                {
                    "step": f.get("message_index", i),
                    "intensity": str(f.get("intensity", "")),
                    "duration_type": str(f.get("duration_type", "")),
                    "duration_value": f.get("duration_value"),
                    "target_type": str(f.get("target_type", "")),
                }
            )
    return pd.DataFrame(rows, columns=columns)
