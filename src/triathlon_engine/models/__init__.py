"""Pace-duration curves (GRE-8) and brick calibration (GRE-9)."""

from .bike_estimate import BikeEstimate, bike_leg_estimate
from .brick import BrickCalibration, brick_calibration
from .pace_curves import (
    PaceCurve,
    fit_bike,
    fit_run,
    fit_swim,
    plot_curves,
    race_predictions,
)

__all__ = [
    "BikeEstimate",
    "BrickCalibration",
    "PaceCurve",
    "bike_leg_estimate",
    "brick_calibration",
    "fit_bike",
    "fit_run",
    "fit_swim",
    "plot_curves",
    "race_predictions",
]
