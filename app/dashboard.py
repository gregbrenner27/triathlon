"""Triathlon Race Engine dashboard (GRE-12).

Run:  streamlit run app/dashboard.py

Design: light theme, shadcn-style cards (hairline ring, 12px radius, surface
on plane), dataviz-skill palette — swim #2a78d6 / bike #1baf7a / run #eda100
(validated categorical trio; aqua & yellow are sub-3:1 on the light surface so
every mark carries a direct label), transitions in neutral gray.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from triathlon_engine.engine import predict_race
from triathlon_engine.insights import (
    brick_reality_check,
    cs_trend,
    training_smart,
    weakest_leg,
)
from triathlon_engine.interpret import label_activities, training_distribution
from triathlon_engine.models import fit_run, fit_swim
from triathlon_engine.models.pace_curves import RACE_DISTANCES_KM

# ---------------------------------------------------------------- palette
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
RING = "rgba(11,11,11,0.10)"
SWIM, BIKE, RUN = "#2a78d6", "#1baf7a", "#eda100"
NEUTRAL = "#c3c2b7"
EASY, HARD = "#2a78d6", "#e34948"  # diverging pair for intensity polarity
LEG_COLOR = {"Swim": SWIM, "T1": NEUTRAL, "Bike": BIKE, "T2": NEUTRAL, "Run": RUN}

st.set_page_config(page_title="Triathlon Race Engine", page_icon="🏊", layout="wide")

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 3.4rem; max-width: 1200px; }}
      div[data-testid="stVerticalBlockBorderWrapper"] {{
          background: {SURFACE}; border: 1px solid {RING};
          border-radius: 12px; box-shadow: 0 1px 2px rgba(11,11,11,0.04);
      }}
      .hero-time {{ font-size: 4.2rem; font-weight: 700; line-height: 1;
                    letter-spacing: -0.02em; color: {INK}; }}
      .hero-range {{ font-size: 1.05rem; color: {INK_2}; margin-top: .4rem; }}
      .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
               font-size: .8rem; font-weight: 600; border: 1px solid {RING}; }}
      .badge-good {{ background: #e9f6e9; color: #006300; }}
      .kicker {{ text-transform: uppercase; letter-spacing: .08em;
                font-size: .72rem; font-weight: 600; color: {MUTED}; }}
      .card-title {{ font-weight: 650; font-size: 1.0rem; color: {INK}; }}
      .card-body {{ color: {INK_2}; font-size: .92rem; line-height: 1.45; }}
      .caveat {{ color: {MUTED}; font-size: .78rem; line-height: 1.35; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data (cached)
@st.cache_data(show_spinner="Reading your Garmin history…")
def _load(db_mtime: float):
    """Cache keyed on the activities DB's mtime — a GarminDB sync changes it,
    so a page refresh after a sync recomputes everything automatically."""
    labeled = label_activities()
    forecast = predict_race()
    return {
        "labeled": labeled,
        "forecast": forecast,
        "dist": training_distribution(labeled),
        "trend": cs_trend(),
        "swim_curve": fit_swim(),
        "run_curve": fit_run(),
        "weakest": weakest_leg(forecast),
        "brick": brick_reality_check(),
        "smart": training_smart(labeled),
    }


def fmt(s: float) -> str:
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def base_layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  color=INK_2, size=13),
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=NEUTRAL, zeroline=False,
                     tickcolor=MUTED, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=NEUTRAL, zeroline=False,
                     tickcolor=MUTED, tickfont=dict(color=MUTED))
    return fig


def load():
    from triathlon_engine.data.access import ACTIVITIES_DB

    return _load(ACTIVITIES_DB.stat().st_mtime)


D = load()
forecast = D["forecast"]
legs = {l.name: l for l in forecast.legs}

# ---------------------------------------------------------------- hero
st.markdown('<div class="kicker">Sprint triathlon · 750 m / 20 km / 5 km</div>',
            unsafe_allow_html=True)
st.title("Race Engine")

with st.container(border=True):
    left, right = st.columns([1.1, 1.6], gap="large")
    with left:
        st.markdown('<div class="kicker">Predicted finish</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-time">{fmt(forecast.finish_s)}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="hero-range">range {fmt(forecast.finish_lo_s)} – '
            f'{fmt(forecast.finish_hi_s)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="badge badge-good">readiness: {forecast.readiness}</span> '
            f'<span class="caveat">{forecast.readiness_detail}</span>',
            unsafe_allow_html=True,
        )
    with right:
        # one stacked timeline bar, 2px surface gaps, direct labels
        fig = go.Figure()
        for name in ("Swim", "T1", "Bike", "T2", "Run"):
            leg = legs[name]
            small = leg.time_s < 0.05 * forecast.finish_s
            fig.add_bar(
                y=[""], x=[leg.time_s], orientation="h", name=name,
                marker=dict(color=LEG_COLOR[name],
                            line=dict(color=SURFACE, width=2)),
                text=f"{name} {fmt(leg.time_s)}" if not small else "",
                textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#ffffff", size=13),
                hovertemplate=(f"{name}: {fmt(leg.time_s)} "
                               f"[{fmt(leg.lo_s)}–{fmt(leg.hi_s)}]<extra></extra>"),
            )
        fig.update_layout(barmode="stack", bargap=0)
        fig.update_yaxes(visible=False)
        fig.update_xaxes(visible=False)
        base_layout(fig, height=88)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        rows = []
        for name in ("Swim", "T1", "Bike", "T2", "Run"):
            leg = legs[name]
            rows.append({"Leg": name, "Time": fmt(leg.time_s),
                         "Range": f"{fmt(leg.lo_s)} – {fmt(leg.hi_s)}",
                         "Basis": leg.source})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- insights
st.markdown("### Insights")
cols = st.columns(3, gap="medium")
for col, ins in zip(cols, (D["weakest"], D["brick"], D["smart"]), strict=True):
    with col, st.container(border=True):
        st.markdown(f'<div class="card-title">{ins.title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-body">{ins.sentence}</div>', unsafe_allow_html=True)
        if ins.caveat:
            st.markdown(f'<div class="caveat">{ins.caveat}</div>', unsafe_allow_html=True)

# training distribution: easy/hard polarity, direct labels
with st.container(border=True):
    st.markdown('<div class="card-title">Training distribution (last year, share of hours)</div>',
                unsafe_allow_html=True)
    dist = D["dist"].sort_values("time_share")
    fig = go.Figure(
        go.Bar(
            y=dist["label"], x=dist["time_share"], orientation="h",
            marker=dict(color=[EASY if i == "easy" else HARD for i in dist["intensity"]],
                        line=dict(color=SURFACE, width=2)),
            text=[f"{v:.0%}" for v in dist["time_share"]],
            textposition="outside", textfont=dict(color=INK_2),
            hovertemplate="%{y}: %{x:.1%}, %{customdata:.1f} h<extra></extra>",
            customdata=dist["hours"],
        )
    )
    fig.update_xaxes(tickformat=".0%", range=[0, dist["time_share"].max() * 1.18])
    base_layout(fig, height=240)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<span style="color:{EASY}">■</span> <span class="caveat">easy</span> '
        f'&nbsp;<span style="color:{HARD}">■</span> <span class="caveat">hard</span>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- fitness
st.markdown("### Fitness readout")
c1, c2 = st.columns([1.5, 1], gap="medium")

with c1, st.container(border=True):
    st.markdown('<div class="card-title">Run critical speed — rolling 12-week blocks</div>',
                unsafe_allow_html=True)
    trend = D["trend"]
    fig = go.Figure(
        go.Scatter(
            x=trend["block_end"], y=trend["cs_kmh"], mode="lines+markers",
            line=dict(color=SWIM, width=2), marker=dict(size=7, color=SWIM),
            hovertemplate="%{x|%b %Y}: %{y:.1f} km/h (n=%{customdata})<extra></extra>",
            customdata=trend["n"],
        )
    )
    fig.update_yaxes(title_text="critical speed (km/h)", title_font=dict(color=MUTED))
    base_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="caveat">Run only — swim/bike effort history is too thin '
                'for a trend. Early dips are low-effort blocks (small n).</div>',
                unsafe_allow_html=True)

with c2, st.container(border=True):
    st.markdown('<div class="card-title">Pace-duration curves (best efforts)</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    for curve, color, label in ((D["run_curve"], RUN, "run"),
                                (D["swim_curve"], SWIM, "swim")):
        pts = curve.points
        xs = np.geomspace(pts["distance_km"].min(),
                          max(pts["distance_km"].max(),
                              RACE_DISTANCES_KM[curve.sport]), 60)
        fig.add_scatter(x=xs, y=[curve.predict_time_s(x) / 60 for x in xs],
                        mode="lines", line=dict(color=color, width=2),
                        hoverinfo="skip")
        fig.add_scatter(x=pts["distance_km"], y=pts["time_s"] / 60, mode="markers",
                        marker=dict(size=8, color=color,
                                    line=dict(color=SURFACE, width=2)),
                        hovertemplate=f"{label} " + "%{x:.2g} km: %{y:.1f} min<extra></extra>")
        mid_i = len(xs) // 2
        fig.add_annotation(x=np.log10(xs[mid_i]),
                           y=np.log10(curve.predict_time_s(xs[mid_i]) / 60),
                           text=label, showarrow=False, yshift=-16,
                           font=dict(color=color, size=12))
    fig.update_xaxes(type="log", title_text="distance (km)",
                     title_font=dict(color=MUTED),
                     tickvals=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
                     ticktext=["0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10"])
    fig.update_yaxes(type="log", title_text="time (min)",
                     title_font=dict(color=MUTED),
                     tickvals=[1, 2, 5, 10, 20, 40],
                     ticktext=["1", "2", "5", "10", "20", "40"])
    base_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="caveat">Bike: no fitted curve yet — all rides indoor '
                'without speed data; leg uses the power model.</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- what-if
st.markdown("### What-if")
st.markdown(
    '<div class="card-body">Drag a slider and watch each leg update: '
    "what your swim, bike and run become, and what that does to the finish.</div>",
    unsafe_allow_html=True,
)
with st.container(border=True):
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    swim_d = s1.slider("Swim pace change (s/100m)", -30, 15, 0, 5)
    bike_d = s2.slider("Bike speed change (km/h)", -4.0, 6.0, 0.0, 0.5)
    run_d = s3.slider("Run pace change (s/km)", -30, 15, 0, 5)
    tran_d = s4.slider("Transition time change (s)", -120, 60, 0, 15)

    # baseline per-leg paces
    swim_pace_0 = legs["Swim"].time_s / (RACE_DISTANCES_KM["swim"] * 10)  # s/100m
    bike_v_0 = RACE_DISTANCES_KM["bike"] / (legs["Bike"].time_s / 3600.0)  # km/h
    run_pace_0 = legs["Run"].time_s / RACE_DISTANCES_KM["run"]  # s/km
    tran_0 = legs["T1"].time_s + legs["T2"].time_s

    # adjusted paces → leg times
    swim_pace = swim_pace_0 + swim_d
    bike_v = max(bike_v_0 + bike_d, 1e-6)
    run_pace = run_pace_0 + run_d
    swim_t = swim_pace * RACE_DISTANCES_KM["swim"] * 10
    bike_t = RACE_DISTANCES_KM["bike"] / bike_v * 3600.0
    run_t = run_pace * RACE_DISTANCES_KM["run"]
    tran_t = max(tran_0 + tran_d, 0.0)
    new_finish = swim_t + bike_t + run_t + tran_t
    delta = new_finish - forecast.finish_s

    def leg_metric(col, name, pace_txt, new_t, old_t):
        d = new_t - old_t
        col.metric(
            f"{name} · {pace_txt}",
            fmt(new_t),
            delta=None if abs(d) < 1 else f"{'+' if d > 0 else '-'}{fmt(abs(d))}",
            delta_color="inverse",  # less time = green
            help=f"was {fmt(old_t)}",
        )

    st.markdown('<div class="kicker" style="margin-top:.6rem">Your race at these settings</div>',
                unsafe_allow_html=True)
    c_swim, c_bike, c_run, c_tran = st.columns(4, gap="medium")
    leg_metric(c_swim, "Swim 750m", f"{fmt(swim_pace)} /100m", swim_t, legs["Swim"].time_s)
    leg_metric(c_bike, "Bike 20k", f"{bike_v:.1f} km/h", bike_t, legs["Bike"].time_s)
    leg_metric(c_run, "Run 5k", f"{fmt(run_pace)} /km", run_t, legs["Run"].time_s)
    leg_metric(c_tran, "T1 + T2", "combined", tran_t, tran_0)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Adjusted finish", fmt(new_finish))
    if abs(delta) < 1:
        m2.metric("vs prediction", "±0:00")
    else:
        m2.metric("vs prediction", f"{'−' if delta < 0 else '+'}{fmt(abs(delta))}",
                  delta=f"{-delta / 60:.1f} min", delta_color="normal")

st.markdown(
    '<div class="caveat" style="margin-top:1rem">Assumptions: '
    + " · ".join(forecast.assumptions)
    + "</div>",
    unsafe_allow_html=True,
)
