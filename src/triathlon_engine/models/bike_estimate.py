"""Bike-leg estimate from self-reported indoor power (GRE-8 adaptation).

Greg's rides are all indoor with no speed/distance stream, so a pace-duration
curve can't be fit (see ``pace_curves.fit_bike``). Until outdoor rides exist,
the bike leg is estimated from physics: reported sustainable watts → flat-road
speed via the standard cubic power balance

    P·η = ½·ρ·CdA·v³ + Crr·m·g·v

Inputs are Greg's self-reported numbers (2026-07-05): steady spin sessions at
150–170 W (console readout, HR 120–130 = comfortably aerobic), interval days
~180–220 W (rough). A ~40 min race effort near threshold sits between those
bands. Spin-console power is ±15%; race position is a road bike on the hoods.

This is an ESTIMATE, not a fitted model — bounds are scenario-based (worst
power × worst aero vs best power × best aero) and deliberately wide. It should
be replaced by the fitted curve as soon as outdoor ride data syncs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Rider + environment
RIDER_KG = 73.3  # from Garmin profile
BIKE_KIT_KG = 9.0
AIR_DENSITY = 1.225  # kg/m³, sea level
CRR = 0.005  # road tires, decent tarmac
DRIVETRAIN_EFF = 0.975
GRAVITY = 9.81

# Road bike on the hoods; band covers hoods-to-slightly-upright
CDA_MID = 0.32
CDA_RANGE = (0.30, 0.36)

# Race-effort power band from Greg's reported numbers (steady 150–170 W
# aerobic, intervals ~180–220 W): threshold-ish 40 min effort
RACE_POWER_W = (170.0, 210.0)
CONSOLE_ERROR = 0.15  # spin consoles are ±15%


@dataclass
class BikeEstimate:
    distance_km: float
    time_s: float
    time_lo_s: float
    time_hi_s: float
    speed_kmh: float
    power_w_mid: float
    source: str = "power-model (self-reported watts)"


def speed_from_power(power_w: float, cda: float = CDA_MID) -> float:
    """Flat-road speed (m/s) sustained at power_w — real root of the cubic."""
    a = 0.5 * AIR_DENSITY * cda
    b = CRR * (RIDER_KG + BIKE_KIT_KG) * GRAVITY
    p = power_w * DRIVETRAIN_EFF
    roots = np.roots([a, 0.0, b, -p])
    real = roots[np.isreal(roots)].real
    return float(real[real > 0].min())


def bike_leg_estimate(distance_km: float = 20.0) -> BikeEstimate:
    """Race bike-leg time with scenario bounds.

    mid  = mid power, mid CdA
    low  = high power +console error, best CdA (fast scenario)
    high = low power −console error, worst CdA (slow scenario)
    """
    p_lo, p_hi = RACE_POWER_W
    p_mid = (p_lo + p_hi) / 2

    v_mid = speed_from_power(p_mid, CDA_MID)
    v_fast = speed_from_power(p_hi * (1 + CONSOLE_ERROR), CDA_RANGE[0])
    v_slow = speed_from_power(p_lo * (1 - CONSOLE_ERROR), CDA_RANGE[1])

    d_m = distance_km * 1000.0
    return BikeEstimate(
        distance_km=distance_km,
        time_s=d_m / v_mid,
        time_lo_s=d_m / v_fast,
        time_hi_s=d_m / v_slow,
        speed_kmh=v_mid * 3.6,
        power_w_mid=p_mid,
    )
