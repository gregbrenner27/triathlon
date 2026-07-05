"""Pace-duration curves (GRE-8) and brick calibration (GRE-9)."""

from .pace_curves import (
    PaceCurve,
    fit_bike,
    fit_run,
    fit_swim,
    plot_curves,
    race_predictions,
)

__all__ = [
    "PaceCurve",
    "fit_bike",
    "fit_run",
    "fit_swim",
    "plot_curves",
    "race_predictions",
]
