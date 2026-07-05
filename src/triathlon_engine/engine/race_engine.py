"""Race Engine — compose the sprint-tri finish-time prediction (GRE-10).

    swim + T1 + bike + T2 + (run × brick factor)  →  finish time + range

Leg times and per-leg uncertainty come from GRE-8 (``race_predictions``),
the brick factor from GRE-9 (``brick_calibration``), and the range is
widened by a readiness assessment built on GRE-7's session labels. Every
assumption is carried in the output — nothing tuned silently.

Range composition is a straight sum of per-leg lows/highs (legs correlate on
race day — heat, pacing errors, nerves — so independence-style sqrt-shrinking
would overstate confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models.brick import RULE_OF_THUMB_FACTOR, brick_calibration
from ..models.pace_curves import race_predictions

# First-timer transition assumptions (pool sprint, no wetsuit strip): tweak
# freely — they're surfaced in the output.
T1_S = (120.0, 180.0, 300.0)  # (fast, expected, slow)
T2_S = (60.0, 90.0, 150.0)

# Readiness → range widening on each leg's half-width
_READINESS_WIDEN = {"solid": 1.0, "fair": 1.15, "thin": 1.30}
_WEEKS_LOOKBACK = 8


@dataclass
class Leg:
    name: str
    time_s: float
    lo_s: float
    hi_s: float
    source: str
    note: str = ""


@dataclass
class RaceForecast:
    finish_s: float
    finish_lo_s: float
    finish_hi_s: float
    legs: list[Leg]
    readiness: str
    readiness_detail: str
    assumptions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        def fmt(s: float) -> str:
            m, sec = divmod(int(round(s)), 60)
            h, m = divmod(m, 60)
            return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

        lines = [
            f"Predicted finish: {fmt(self.finish_s)} "
            f"[{fmt(self.finish_lo_s)} – {fmt(self.finish_hi_s)}]",
            f"Readiness: {self.readiness} — {self.readiness_detail}",
            "",
        ]
        for leg in self.legs:
            lines.append(
                f"  {leg.name:<5} {fmt(leg.time_s):>8}  "
                f"[{fmt(leg.lo_s)} – {fmt(leg.hi_s)}]  ({leg.source})"
            )
        lines.append("")
        lines.extend(f"  • {a}" for a in self.assumptions)
        return "\n".join(lines)


def _readiness(labeled: pd.DataFrame) -> tuple[str, str]:
    """Grade training readiness from the last _WEEKS_LOOKBACK weeks.

    solid: ≥2 sessions/wk in every discipline; fair: every discipline
    trained but one is <2/wk; thin: a discipline missing or <1/wk.
    """
    recent = labeled[
        labeled["start_time"]
        >= labeled["start_time"].max() - pd.Timedelta(weeks=_WEEKS_LOOKBACK)
    ]
    names = {"swimming": "swim", "cycling": "bike", "running": "run"}
    per_wk = {
        sport: len(recent[recent["sport"] == sport]) / _WEEKS_LOOKBACK
        for sport in names
    }
    detail = ", ".join(f"{names[s]} {v:.1f}/wk" for s, v in per_wk.items())
    if min(per_wk.values()) >= 2.0:
        return "solid", detail
    if min(per_wk.values()) >= 1.0:
        return "fair", detail
    return "thin", detail


def predict_race() -> RaceForecast:
    """The master prediction: legs + transitions + brick + readiness."""
    from ..interpret import label_activities

    labeled = label_activities()
    preds = race_predictions().set_index("sport")
    brick = brick_calibration(labeled)
    readiness, detail = _readiness(labeled)
    widen = _READINESS_WIDEN[readiness]

    assumptions = [
        f"T1 {T1_S[1]:.0f}s (range {T1_S[0]:.0f}–{T1_S[2]:.0f}) and "
        f"T2 {T2_S[1]:.0f}s (range {T2_S[0]:.0f}–{T2_S[2]:.0f}): "
        "first-timer estimates, edit engine/race_engine.py to tweak",
        f"brick factor {brick.factor:.2f} applied to run "
        f"({brick.source}, n={brick.n_bricks}"
        + (
            f", measured {brick.measured_factor:.2f} floored at 1.00"
            if brick.measured_factor == brick.measured_factor
            and brick.measured_factor < 1.0
            else ""
        )
        + f"); run upper bound allows {RULE_OF_THUMB_FACTOR:.2f}",
        f"readiness '{readiness}' widens leg ranges ×{widen:.2f}",
    ]

    legs: list[Leg] = []
    for sport, label in (("swim", "Swim"), ("bike", "Bike"), ("run", "Run")):
        p = preds.loc[sport]
        mid, lo, hi = p["time_s"], p["time_lo_s"], p["time_hi_s"]
        if sport == "run":
            mid, lo = mid * brick.factor, lo * brick.factor
            hi = hi * max(brick.factor, RULE_OF_THUMB_FACTOR)
        # readiness widening around the mid
        lo, hi = mid - (mid - lo) * widen, mid + (hi - mid) * widen
        legs.append(
            Leg(label, float(mid), float(lo), float(hi), p["source"], p["note"])
        )
        if p["note"]:
            assumptions.append(f"{label}: {p['note']}")
    legs.insert(1, Leg("T1", T1_S[1], T1_S[0], T1_S[2], "assumption"))
    legs.insert(3, Leg("T2", T2_S[1], T2_S[0], T2_S[2], "assumption"))

    return RaceForecast(
        finish_s=sum(l.time_s for l in legs),
        finish_lo_s=sum(l.lo_s for l in legs),
        finish_hi_s=sum(l.hi_s for l in legs),
        legs=legs,
        readiness=readiness,
        readiness_detail=detail,
        assumptions=assumptions,
    )
