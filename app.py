"""
Dynamic Pricing Lab — Streamlit Application
Premium dashboard with 6 tabs: Live, Think, Scoreboard, Belief, Update, Baselines.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from simulator import generate_historical, LiveWorld
from features import build_features, feature_columns, RollingFeatureStore
from model import DemandModel
from optimizer import PricingOptimizer

# ─────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Lab",
    page_icon="DPL",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
#  Custom CSS — clean light professional theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ── */
.stApp {
    background: #f8f9fc;
    color: #1e293b;
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
    background: #eef1f6 !important;
    border-right: 1px solid #d5dbe5;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #eef1f6;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: #1a73e8 !important;
    color: #fff !important;
}
h1, h2, h3 { color: #1e293b; }

/* ── KPI card ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #1a73e8;
}
.kpi-value.warn {
    color: #dc2626;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
}
.badge-explore {
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #fed7aa;
}
.badge-exploit {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
}
.badge-retrain {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}
.badge-shock {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}
.badge-drift {
    background: #fefce8;
    color: #a16207;
    border: 1px solid #fde68a;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] .stAppDeployButton { display: none; }
header[data-testid="stHeader"] button[kind="header"] { display: none; }
[data-testid="stToolbar"] { display: none !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="JetBrains Mono, monospace", color="#334155", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(255,255,255,0)", font_size=10),
)
COLOR_PRICE = "#1a73e8"
COLOR_REC = "#16a34a"
COLOR_REV = "#7c3aed"
COLOR_BAND = "rgba(26,115,232,0.10)"
COLOR_SHOCK = "#dc2626"
COLOR_BELIEF = "#ea580c"
COLOR_BASELINE_A = "#64748b"
COLOR_BASELINE_B = "#d97706"


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, warn: bool = False):
    cls = "kpi-value warn" if warn else "kpi-value"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>"""


def badge(text: str, kind: str = "exploit"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def speed_seconds(speed: str) -> float:
    return {"1x": 1.0, "5x": 0.2, "20x": 0.05}[speed]


# ─────────────────────────────────────────────────────────────────────
#  Session state initialization
# ─────────────────────────────────────────────────────────────────────

def init_state():
    if "initialized" in st.session_state:
        return

    cfg = Config()
    st.session_state.cfg = cfg
    st.session_state.running = False
    st.session_state.tick = 0
    st.session_state.log = []
    st.session_state.last_price = 100.0
    st.session_state.last_candidates = None
    st.session_state.last_mc = None
    st.session_state.retrain_ticks = []
    st.session_state.errors_rolling = []
    st.session_state.drift_alerts = []
    st.session_state.baseline_a_cum = 0.0
    st.session_state.baseline_b_cum = 0.0
    st.session_state.opt_cum = 0.0
    st.session_state.baseline_a_hist = []
    st.session_state.baseline_b_hist = []
    st.session_state.opt_hist = []

    # ── Generate historical data & train model ──
    with st.spinner("Generating 180 days of historical data..."):
        hist = generate_historical(cfg)
        hist_feat = build_features(hist, comp_enabled=cfg.comp_enabled)
        fcols = feature_columns(cfg.comp_enabled)
        X = hist_feat[fcols]
        y = hist_feat["sales"].values.astype(float)

        dm = DemandModel()
        dm.fit(X, y)
        st.session_state.model = dm
        st.session_state.hist = hist
        st.session_state.hist_feat = hist_feat
        st.session_state.fcols = fcols

    # ── Live world ──
    st.session_state.world = LiveWorld(cfg)
    st.session_state.feature_store = RollingFeatureStore(comp_enabled=cfg.comp_enabled)
    st.session_state.optimizer = PricingOptimizer(cfg)

    st.session_state.initialized = True


# ─────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────

def render_sidebar():
    cfg = st.session_state.cfg

    st.sidebar.markdown("## Dynamic Pricing Lab")
    st.sidebar.markdown("---")

    objective = st.sidebar.radio(
        "Objective", ["Revenue", "Profit"], index=0, horizontal=True,
    )
    st.session_state.objective = objective.lower()

    st.session_state.explore = st.sidebar.toggle("Exploration (Thompson)", value=False)

    st.session_state.speed = st.sidebar.select_slider(
        "Speed", options=["1x", "5x", "20x"], value="5x",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Parameters")

    cfg.unit_cost = st.sidebar.slider("Unit Cost", 30.0, 120.0, cfg.unit_cost, 1.0)
    cfg.max_price_change_per_tick = st.sidebar.slider(
        "Max Δ Price/Tick", 0.5, 10.0, cfg.max_price_change_per_tick, 0.5,
    )
    cfg.stock_initial = st.sidebar.slider("Initial Stock", 50, 1000, cfg.stock_initial, 10)
    cfg.comp_enabled = st.sidebar.toggle("Competition", value=cfg.comp_enabled)

    st.sidebar.markdown("---")

    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("Start", width="stretch"):
        st.session_state.running = True
    if col2.button("Pause", width="stretch"):
        st.session_state.running = False
    if col3.button("Reset", width="stretch"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # model info
    dm = st.session_state.model
    if dm.is_fitted:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Model MAE**: `{dm.mae_holdout:.2f}`")
        st.sidebar.markdown(f"**Residual σ**: `{dm.residual_std:.2f}`")
        if st.session_state.retrain_ticks:
            st.sidebar.markdown(
                badge("RETRAINED", "retrain") +
                f" at tick {st.session_state.retrain_ticks[-1]}",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────
#  Simulation step
# ─────────────────────────────────────────────────────────────────────

def run_tick():
    """Execute one simulation tick."""
    cfg = st.session_state.cfg
    world = st.session_state.world
    model = st.session_state.model
    fstore = st.session_state.feature_store
    opt = st.session_state.optimizer

    tick = st.session_state.tick
    last_price = st.session_state.last_price

    # ── Peek at current world state (price not yet decided) ──
    peek_obs = {
        "dow": ((world.start_hour * 60 + tick) // 1440) % 7,
        "hour": ((world.start_hour * 60 + tick) // 60) % 24,
        "traffic": 80.0,  # placeholder, world will generate real
        "promo": int(world.promo_active),
        "comp_price": world.comp_price,
        "stock": world.stock,
    }
    # get rolling features
    base_row = fstore.current_feature_row(peek_obs, price_override=last_price)

    # ── Optimize ──
    result = opt.optimize(
        model=model,
        base_row=base_row,
        last_price=last_price,
        objective=st.session_state.objective,
        explore=st.session_state.explore,
        comp_enabled=cfg.comp_enabled,
    )
    rec_price = result["recommended_price"]
    st.session_state.last_candidates = result["candidates"]
    st.session_state.last_mc = result["mc_result"]

    # ── Tick world ──
    obs = world.tick(rec_price)
    fstore.push(obs)

    # ── Prediction for the applied price ──
    real_row = fstore.current_feature_row(obs)
    pred_mean, pred_p10, pred_p90 = model.predict_single(real_row)

    # ── Compute objectives ──
    if st.session_state.objective == "profit":
        obj_real = (rec_price - cfg.unit_cost) * obs["sales"]
        obj_expected = (rec_price - cfg.unit_cost) * pred_mean
    else:
        obj_real = rec_price * obs["sales"]
        obj_expected = rec_price * pred_mean

    # ── Baselines ──
    baseline_a_sales = min(obs["demand"], obs["stock"] + obs["sales"])  # approx
    baseline_b_price = max(cfg.p_min, min(cfg.p_max, obs["comp_price"] - 1))
    if st.session_state.objective == "profit":
        bl_a = (100 - cfg.unit_cost) * obs["sales"]
        bl_b = (baseline_b_price - cfg.unit_cost) * obs["sales"]
    else:
        bl_a = 100 * obs["sales"]
        bl_b = baseline_b_price * obs["sales"]

    st.session_state.baseline_a_cum += bl_a
    st.session_state.baseline_b_cum += bl_b
    st.session_state.opt_cum += obj_real
    st.session_state.baseline_a_hist.append(st.session_state.baseline_a_cum)
    st.session_state.baseline_b_hist.append(st.session_state.baseline_b_cum)
    st.session_state.opt_hist.append(st.session_state.opt_cum)

    # ── Error tracking ──
    error = obs["sales"] - pred_mean
    st.session_state.errors_rolling.append(error)

    # ── Drift detection (simple: compare recent traffic dist) ──
    drift_flag = False
    if len(st.session_state.log) > 60:
        recent = [l["traffic"] for l in st.session_state.log[-30:]]
        older = [l["traffic"] for l in st.session_state.log[-60:-30]]
        psi = abs(np.mean(recent) - np.mean(older)) / (np.std(older) + 1e-6)
        drift_flag = psi > 1.5
    st.session_state.drift_alerts.append(drift_flag)

    # ── Detect active shocks ──
    active_shocks = []
    for s in cfg.shocks:
        if s.kind == "promo" and s.tick <= tick < s.tick + s.duration:
            active_shocks.append("PROMO")
        elif s.tick == tick:
            active_shocks.append(s.kind.upper())

    # ── Log ──
    log_entry = {
        **obs,
        "rec_price": rec_price,
        "pred_mean": round(pred_mean, 2),
        "pred_p10": round(pred_p10, 2),
        "pred_p90": round(pred_p90, 2),
        "obj_real": round(obj_real, 2),
        "obj_expected": round(obj_expected, 2),
        "error": round(error, 2),
        "drift": drift_flag,
        "shocks": active_shocks,
    }
    st.session_state.log.append(log_entry)
    st.session_state.last_price = rec_price
    st.session_state.tick += 1

    # ── Retrain model every N ticks ──
    if st.session_state.tick % cfg.retrain_every == 0 and st.session_state.tick > 0:
        retrain_model()
        st.session_state.retrain_ticks.append(st.session_state.tick)


def retrain_model():
    """Retrain model on historical + observed live data."""
    try:
        cfg = st.session_state.cfg
        hist = st.session_state.hist_feat.copy()

        # Extract only the columns build_features needs from live log
        live_raw = pd.DataFrame(st.session_state.log)
        keep_cols = ["day", "hour", "dow", "traffic", "promo", "comp_price",
                     "price", "stock", "demand", "sales"]
        live_cols = [c for c in keep_cols if c in live_raw.columns]
        live_df = live_raw[live_cols].copy()
        if "day" not in live_df.columns:
            live_df["day"] = 0

        live_feat = build_features(live_df, comp_enabled=cfg.comp_enabled)
        combined = pd.concat([hist, live_feat], ignore_index=True)
        fcols = st.session_state.fcols
        X = combined[fcols].fillna(0)
        y = combined["sales"].fillna(0).values.astype(float)

        new_model = DemandModel()
        new_model.fit(X, y)
        st.session_state.model = new_model
    except Exception:
        # If retraining fails, keep the old model
        pass


# ─────────────────────────────────────────────────────────────────────
#  Tab renderers
# ─────────────────────────────────────────────────────────────────────

def render_tab_live():
    """Tab 1 — Live Terminal"""
    log = st.session_state.log
    if not log:
        st.info("Press **Start** in the sidebar to begin the simulation.")
        return

    last = log[-1]
    cfg = st.session_state.cfg

    # ── KPI cards ──
    cols = st.columns(6)
    cards = [
        ("Tick", f"{last['tick']}/{cfg.live_ticks}"),
        ("Price", f"${last['rec_price']:.2f}"),
        (f"{st.session_state.objective.title()} (real)", f"${last['obj_real']:.0f}"),
        ("Sales", str(last["sales"])),
        ("Stock", str(last["stock"])),
        ("Traffic", f"{last['traffic']:.0f}"),
    ]
    for i, (lbl, val) in enumerate(cards):
        warn = (lbl == "Stock" and last["stock"] < 30)
        cols[i].markdown(kpi_card(lbl, val, warn), unsafe_allow_html=True)

    # ── Badges row ──
    badges_html = ""
    if st.session_state.explore:
        badges_html += badge("EXPLORE", "explore") + " "
    else:
        badges_html += badge("EXPLOIT", "exploit") + " "
    if last.get("shocks"):
        for s in last["shocks"]:
            badges_html += badge(f"SHOCK: {s}", "shock") + " "
    if last.get("drift"):
        badges_html += badge("DRIFT", "drift") + " "
    if st.session_state.tick in st.session_state.retrain_ticks:
        badges_html += badge("RETRAINED", "retrain") + " "
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ──
    window = log[-120:]
    ticks = [l["tick"] for l in window]

    # Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=ticks, y=[l["price"] for l in window],
        name="Applied Price", line=dict(color=COLOR_PRICE, width=2),
    ))
    fig_price.add_trace(go.Scatter(
        x=ticks, y=[l["rec_price"] for l in window],
        name="Recommended", line=dict(color=COLOR_REC, width=2, dash="dot"),
    ))
    if cfg.comp_enabled:
        fig_price.add_trace(go.Scatter(
            x=ticks, y=[l["comp_price"] for l in window],
            name="Competitor", line=dict(color=COLOR_BASELINE_B, width=1, dash="dash"),
        ))
    # shock markers
    for l in window:
        if l.get("shocks"):
            fig_price.add_vline(x=l["tick"], line_dash="dot",
                                line_color=COLOR_SHOCK, opacity=0.5)
    fig_price.update_layout(
        title="Price Evolution", height=300, template=PLOTLY_TEMPLATE,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_price, width="stretch")

    # Objective chart
    fig_obj = go.Figure()
    fig_obj.add_trace(go.Scatter(
        x=ticks, y=[l["obj_real"] for l in window],
        name=f"{st.session_state.objective.title()} (Real)",
        line=dict(color=COLOR_REV, width=2),
    ))
    fig_obj.add_trace(go.Scatter(
        x=ticks, y=[l["obj_expected"] for l in window],
        name="Expected",
        line=dict(color=COLOR_REC, width=1.5, dash="dot"),
    ))
    fig_obj.update_layout(
        title=f"{st.session_state.objective.title()} per Tick", height=280,
        template=PLOTLY_TEMPLATE, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_obj, width="stretch")


def render_tab_think():
    """Tab 2 — Think (demand + objective curves with uncertainty)"""
    candidates = st.session_state.last_candidates
    log = st.session_state.log
    if candidates is None or not log:
        st.info("Waiting for simulation data…")
        return

    last = log[-1]
    cfg = st.session_state.cfg
    prices = candidates["price"].values

    # ── Demand vs Price ──
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=prices, y=candidates["d_p90"],
        line=dict(width=0), showlegend=False,
    ))
    fig_d.add_trace(go.Scatter(
        x=prices, y=candidates["d_p10"],
        fill="tonexty", fillcolor="rgba(26,115,232,0.10)",
        line=dict(width=0), name="p10–p90 Band",
    ))
    fig_d.add_trace(go.Scatter(
        x=prices, y=candidates["d_mean"],
        name="E[Demand]", line=dict(color=COLOR_PRICE, width=2.5),
    ))
    fig_d.add_vline(x=last["price"], line_dash="solid",
                    line_color=COLOR_PRICE, annotation_text="Applied",
                    annotation_font_color=COLOR_PRICE)
    fig_d.add_vline(x=last["rec_price"], line_dash="dash",
                    line_color=COLOR_REC, annotation_text="Recommended",
                    annotation_font_color=COLOR_REC)
    fig_d.update_layout(
        title=f"Demand Curve — Tick {last['tick']}",
        xaxis_title="Price ($)", yaxis_title="Demand (units)",
        height=350, template=PLOTLY_TEMPLATE, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_d, width="stretch")

    # ── Objective vs Price ──
    fig_o = go.Figure()
    fig_o.add_trace(go.Scatter(
        x=prices, y=candidates["obj_p90"],
        line=dict(width=0), showlegend=False,
    ))
    fig_o.add_trace(go.Scatter(
        x=prices, y=candidates["obj_p10"],
        fill="tonexty", fillcolor="rgba(124,58,237,0.10)",
        line=dict(width=0), name="p10–p90 Band",
    ))
    fig_o.add_trace(go.Scatter(
        x=prices, y=candidates["obj_expected"],
        name=f"E[{st.session_state.objective.title()}]",
        line=dict(color=COLOR_REV, width=2.5),
    ))
    fig_o.add_vline(x=last["price"], line_dash="solid",
                    line_color=COLOR_PRICE, annotation_text="Applied",
                    annotation_font_color=COLOR_PRICE)
    fig_o.add_vline(x=last["rec_price"], line_dash="dash",
                    line_color=COLOR_REC, annotation_text="Recommended",
                    annotation_font_color=COLOR_REC)
    fig_o.update_layout(
        title=f"{st.session_state.objective.title()} Curve — Tick {last['tick']}",
        xaxis_title="Price ($)",
        yaxis_title=f"{st.session_state.objective.title()} ($)",
        height=350, template=PLOTLY_TEMPLATE, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_o, width="stretch")


def render_tab_scoreboard():
    """Tab 3 — Scoreboard: candidate evaluation bar chart."""
    candidates = st.session_state.last_candidates
    log = st.session_state.log
    if candidates is None or not log:
        st.info("Waiting for simulation data…")
        return

    last = log[-1]
    df = candidates.copy()

    colors = [
        COLOR_REC if abs(p - last["rec_price"]) < 0.01 else
        ("rgba(26,115,232,0.5)" if f else "rgba(200,210,220,0.5)")
        for p, f in zip(df["price"], df["feasible"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["price"], y=df["obj_expected"],
        marker_color=colors,
        hovertemplate=(
            "Price: $%{x:.1f}<br>"
            "E[Obj]: %{y:.1f}<br>"
            "D_mean: %{customdata[0]:.1f}<br>"
            "D_p10: %{customdata[1]:.1f}<br>"
            "D_p90: %{customdata[2]:.1f}<br>"
            "Risk width: %{customdata[3]:.1f}<extra></extra>"
        ),
        customdata=df[["d_mean", "d_p10", "d_p90", "risk_width"]].values,
    ))
    fig.update_layout(
        title=f"Candidate Scores — Tick {last['tick']}",
        xaxis_title="Price ($)",
        yaxis_title=f"E[{st.session_state.objective.title()}]",
        height=420, template=PLOTLY_TEMPLATE, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    # Quick stats
    best = df.loc[df["obj_expected"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Price", f"${best['price']:.1f}")
    c2.metric("E[Demand]", f"{best['d_mean']:.1f}")
    c3.metric("Risk Width", f"{best['risk_width']:.1f}")


def render_tab_belief():
    """Tab 4 — Belief: probability of being optimal."""
    candidates = st.session_state.last_candidates
    log = st.session_state.log
    if candidates is None or not log:
        st.info("Waiting for simulation data…")
        return

    last = log[-1]
    df = candidates.copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["price"], y=df["belief_prob"],
        marker=dict(
            color=df["belief_prob"],
            colorscale=[[0, "#dbeafe"], [0.5, "#f59e0b"], [1, "#dc2626"]],
            cmin=0, cmax=df["belief_prob"].max() + 0.01,
        ),
        hovertemplate="Price: $%{x:.1f}<br>P(optimal): %{y:.3f}<extra></extra>",
    ))
    fig.add_vline(x=last["rec_price"], line_dash="dash",
                  line_color=COLOR_REC, annotation_text="Chosen",
                  annotation_font_color=COLOR_REC)
    fig.update_layout(
        title=f"Belief Distribution — P(price is optimal) — Tick {last['tick']}",
        xaxis_title="Price ($)", yaxis_title="Probability",
        height=400, template=PLOTLY_TEMPLATE, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    mode_label = "EXPLORE" if st.session_state.explore else "EXPLOIT"
    mode_badge = badge(mode_label, "explore" if st.session_state.explore else "exploit")
    st.markdown(f"Mode: {mode_badge}", unsafe_allow_html=True)

    top3 = df.nlargest(3, "belief_prob")[["price", "belief_prob", "obj_expected"]]
    st.dataframe(top3.style.format({
        "price": "${:.1f}", "belief_prob": "{:.3f}", "obj_expected": "${:.1f}",
    }), hide_index=True, width="stretch")


def render_tab_update():
    """Tab 5 — Update Step: prediction vs actual + drift."""
    log = st.session_state.log
    if not log:
        st.info("Waiting for simulation data…")
        return

    last = log[-1]

    # ── Prediction vs Actual ──
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["p10", "Mean", "p90", "Actual"],
        y=[last["pred_p10"], last["pred_mean"], last["pred_p90"], last["sales"]],
        marker_color=[COLOR_PRICE, COLOR_REC, COLOR_PRICE, COLOR_REV],
        text=[f"{v:.1f}" for v in [last["pred_p10"], last["pred_mean"],
                                     last["pred_p90"], last["sales"]]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Prediction vs Actual — Tick {last['tick']} (Price ${last['price']:.1f})",
        yaxis_title="Demand (units)", height=320,
        template=PLOTLY_TEMPLATE, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    # ── Rolling error ──
    errors = st.session_state.errors_rolling
    if len(errors) > 5:
        window = errors[-120:]
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            y=window, mode="lines",
            line=dict(color=COLOR_PRICE, width=1.5),
            name="Error (actual − predicted)",
        ))
        # rolling MAE
        if len(window) > 10:
            rolling_mae = pd.Series(window).abs().rolling(10).mean().values
            fig_err.add_trace(go.Scatter(
                y=rolling_mae, mode="lines",
                line=dict(color=COLOR_REV, width=2, dash="dot"),
                name="Rolling MAE (10)",
            ))
        fig_err.add_hline(y=0, line_dash="dash", line_color="#cbd5e1")
        fig_err.update_layout(
            title="Rolling Prediction Error", height=280,
            template=PLOTLY_TEMPLATE, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_err, width="stretch")

    # ── Drift alert ──
    if last.get("drift"):
        st.markdown(
            badge("DRIFT DETECTED — traffic distribution shifted", "drift"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            badge("No drift detected", "exploit"),
            unsafe_allow_html=True,
        )

    # ── Retrain info ──
    if st.session_state.retrain_ticks:
        st.markdown(
            f"Last retrain at tick **{st.session_state.retrain_ticks[-1]}** "
            f"(every {st.session_state.cfg.retrain_every} ticks) — "
            + badge("RETRAINED", "retrain"),
            unsafe_allow_html=True,
        )


def render_tab_baselines():
    """Tab 6 — Baselines: cumulative comparison."""
    if not st.session_state.opt_hist:
        st.info("Waiting for simulation data…")
        return

    ticks = list(range(len(st.session_state.opt_hist)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ticks, y=st.session_state.opt_hist,
        name="Optimizer", line=dict(color=COLOR_REC, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=ticks, y=st.session_state.baseline_a_hist,
        name="Baseline A (Fixed $100)",
        line=dict(color=COLOR_BASELINE_A, width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=ticks, y=st.session_state.baseline_b_hist,
        name="Baseline B (Comp -$1)",
        line=dict(color=COLOR_BASELINE_B, width=2, dash="dot"),
    ))
    fig.update_layout(
        title=f"Cumulative {st.session_state.objective.title()} vs Baselines",
        xaxis_title="Tick", yaxis_title=f"Cumulative {st.session_state.objective.title()} ($)",
        height=420, template=PLOTLY_TEMPLATE, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    # ── Uplift metrics ──
    opt = st.session_state.opt_cum
    bl_a = st.session_state.baseline_a_cum
    bl_b = st.session_state.baseline_b_cum

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Optimizer Total", f"${opt:,.0f}"), unsafe_allow_html=True)

    uplift_a = ((opt - bl_a) / max(bl_a, 1)) * 100
    c2.markdown(kpi_card("vs Fixed $100", f"{uplift_a:+.1f}%",
                         warn=(uplift_a < 0)), unsafe_allow_html=True)

    uplift_b = ((opt - bl_b) / max(bl_b, 1)) * 100
    c3.markdown(kpi_card("vs Comp -$1", f"{uplift_b:+.1f}%",
                         warn=(uplift_b < 0)), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    init_state()
    render_sidebar()

    # ── Header ──
    st.markdown(
        "<h1 style='text-align:center; font-family: JetBrains Mono, monospace;'>"
        "Dynamic Pricing Lab</h1>"
        "<p style='text-align:center; color:#64748b; margin-bottom:24px;'>"
        "Real-time pricing optimization with demand uncertainty</p>",
        unsafe_allow_html=True,
    )

    # ── Tabs ──
    tabs = st.tabs([
        "Live", "Think", "Scoreboard",
        "Belief", "Update", "Baselines",
    ])

    with tabs[0]:
        render_tab_live()
    with tabs[1]:
        render_tab_think()
    with tabs[2]:
        render_tab_scoreboard()
    with tabs[3]:
        render_tab_belief()
    with tabs[4]:
        render_tab_update()
    with tabs[5]:
        render_tab_baselines()

    # ── Auto-tick loop ──
    if st.session_state.running and st.session_state.tick < st.session_state.cfg.live_ticks:
        run_tick()
        time.sleep(speed_seconds(st.session_state.speed))
        st.rerun()
    elif st.session_state.tick >= st.session_state.cfg.live_ticks:
        st.session_state.running = False
        st.toast("Simulation complete!")


if __name__ == "__main__":
    main()
