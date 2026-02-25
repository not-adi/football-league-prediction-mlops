"""
src/dashboard/app.py
Football League Predictor — Developed by Aditya Yadav
Fully cloud-compatible: all predictions computed from static JSON files.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL    = os.getenv("API_URL", "http://app:8000")
ARTS_DIR   = Path("mlflow/artifacts")
IS_CLOUD   = not ARTS_DIR.exists() or os.getenv("IS_STREAMLIT_CLOUD", "false") == "true"

# detect cloud by checking if API is reachable
def _api_available() -> bool:
    try:
        requests.get(f"{API_URL}/health", timeout=2)
        return True
    except Exception:
        return False

USE_API = not IS_CLOUD and _api_available()

st.set_page_config(
    page_title="⚽ League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
footer {visibility: hidden;}
.developer-tag {
    position: fixed; bottom: 12px; right: 16px;
    background: linear-gradient(135deg, #0f3460, #16213e);
    color: #a0c4ff; padding: 6px 14px; border-radius: 20px;
    font-size: 12px; border: 1px solid #1a4a8a; z-index: 9999;
}
.developer-tag span { color: #ffd700; font-weight: 600; }
</style>
<div class="developer-tag">⚽ Developed by <span>Aditya Yadav</span></div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ League Predictor")
    st.caption("Dixon-Coles + Monte Carlo Simulation")
    st.divider()

    league_options = {"Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿": "PL", "La Liga 🇪🇸": "PD"}
    selected_league_name = st.selectbox("League", list(league_options.keys()))
    league_code = league_options[selected_league_name]

    st.divider()
    if USE_API:
        if st.button("🔄 Refresh Predictions", use_container_width=True):
            with st.spinner("Refreshing..."):
                try:
                    requests.post(f"{API_URL}/predictions/refresh?league={league_code}", timeout=5)
                    st.cache_data.clear()
                    st.success("Refreshed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not connect: {e}")
    else:
        st.info("🌐 Running on Streamlit Cloud\nPredictions from latest model run", icon="ℹ️")

    st.divider()
    st.caption("Data: football-data.org")
    st.caption("Model: Dixon-Coles (1997)")
    st.caption("Simulations: 10,000 per run")
    st.divider()
    st.markdown("**Developed by**")
    st.markdown("### Aditya Yadav")


# ── Load static params for match predictor ────────────────────────────────
@st.cache_resource
def load_dc_params(league: str):
    """Load Dixon-Coles params from committed JSON — works on cloud."""
    params_path = ARTS_DIR / f"{league}_params.json"
    if not params_path.exists():
        return None
    with open(params_path) as f:
        d = json.load(f)
    return d  # keep as dict, compute inline


def dc_match_probs(params_dict: dict, home_team: str, away_team: str) -> dict:
    """Compute match outcome probabilities from params dict."""
    from scipy.stats import poisson

    attack  = params_dict["attack"]
    defence = params_dict["defence"]
    home_adv = params_dict["home_advantage"]
    rho      = params_dict["rho"]

    lam_h = np.exp(attack[home_team] - defence[away_team] + home_adv)
    lam_a = np.exp(attack[away_team] - defence[home_team])

    max_g = 10
    matrix = np.outer(
        poisson.pmf(range(max_g + 1), lam_h),
        poisson.pmf(range(max_g + 1), lam_a),
    )

    # Rho correction
    for hg, ag, corr_fn in [
        (0, 0, lambda: 1 - lam_h * lam_a * rho),
        (1, 0, lambda: 1 + lam_a * rho),
        (0, 1, lambda: 1 + lam_h * rho),
        (1, 1, lambda: 1 - rho),
    ]:
        matrix[hg, ag] *= corr_fn()

    matrix = np.maximum(matrix, 0)
    matrix /= matrix.sum()

    return {
        "home_win": float(np.tril(matrix, -1).sum()),
        "draw":     float(np.trace(matrix)),
        "away_win": float(np.triu(matrix, 1).sum()),
    }


def fuzzy_find(name: str, teams: list) -> str | None:
    exact = next((t for t in teams if t.lower() == name.lower()), None)
    if exact:
        return exact
    return next((t for t in teams if name.lower() in t.lower()), None)


# ── Fetch predictions ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_from_api(league: str) -> dict:
    resp = requests.get(f"{API_URL}/predictions/{league}", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def fetch_from_static(league: str) -> dict:
    sim_path   = ARTS_DIR / f"{league}_simulation.json"
    probs_path = ARTS_DIR / f"{league}_position_probs.csv"
    if not sim_path.exists() or not probs_path.exists():
        return None
    with open(sim_path) as f:
        sim_data = json.load(f)
    probs_df = pd.read_csv(probs_path, index_col="team")
    standings = []
    for team in sorted(probs_df.index,
                       key=lambda t: sim_data["expected_position"].get(t, 99)):
        row = probs_df.loc[team]
        standings.append({
            "team":              team,
            "expected_position": sim_data["expected_position"].get(team, 0),
            "expected_points":   sim_data["expected_points"].get(team, 0),
            "p_title":      float(row["P(title)"])      if "P(title)"      in probs_df.columns else None,
            "p_top4":       float(row["P(top4)"])       if "P(top4)"       in probs_df.columns else None,
            "p_top6":       float(row["P(top6)"])       if "P(top6)"       in probs_df.columns else None,
            "p_relegation": float(row["P(relegation)"]) if "P(relegation)" in probs_df.columns else None,
        })
    return {"league": league, "season": sim_data.get("season"),
            "generated_at": sim_data.get("generated_at"), "standings": standings}


def fetch_predictions(league: str) -> dict:
    if USE_API:
        try:
            return fetch_from_api(league)
        except Exception:
            pass
    data = fetch_from_static(league)
    if data:
        return data
    st.error("No prediction data found. Run the pipeline and push mlflow/artifacts/ to GitHub.")
    st.stop()


try:
    data = fetch_predictions(league_code)
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()

standings_raw = data["standings"]
generated_at  = data.get("generated_at", "Unknown")

df = pd.DataFrame([{
    "Pos":     i + 1,
    "Team":    t["team"],
    "xPts":    round(t["expected_points"], 1),
    "xPos":    round(t["expected_position"], 1),
    "Title %": round((t.get("p_title")      or 0) * 100, 1),
    "Top 4 %": round((t.get("p_top4")       or 0) * 100, 1),
    "Top 6 %": round((t.get("p_top6")       or 0) * 100, 1),
    "Rel %":   round((t.get("p_relegation") or 0) * 100, 1),
} for i, t in enumerate(standings_raw)])


# ── Header ─────────────────────────────────────────────────────────────────
league_name = "Premier League" if league_code == "PL" else "La Liga"
st.title(f"{league_name} — 2025/26 Final Standings Forecast")
st.caption(f"Last updated: {generated_at} UTC  |  Developed by **Aditya Yadav**")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🏆 Title Favourite",  df.iloc[0]["Team"], f"{df.iloc[0]['Title %']}% chance")
with c2:
    t4 = df.sort_values("Top 4 %", ascending=False).iloc[0]
    st.metric("🟢 Top 4 Favourite",  t4["Team"], f"{t4['Top 4 %']}%")
with c3:
    rl = df.sort_values("Rel %", ascending=False).iloc[0]
    st.metric("🔴 Relegation Risk",  rl["Team"], f"{rl['Rel %']}%")
with c4:
    st.metric("📊 Teams", len(df), "2025/26 season")

st.divider()


# ── Standings table ─────────────────────────────────────────────────────────
st.subheader("📋 Predicted Final Standings")

def style_row(row):
    pos, n = row["Pos"], len(df)
    if pos == 1:       return ["background-color: #ffd70033"] * len(row)
    elif pos <= 4:     return ["background-color: #4CAF5022"] * len(row)
    elif pos <= 6:     return ["background-color: #2196F322"] * len(row)
    elif pos >= n - 2: return ["background-color: #f4433622"] * len(row)
    return [""] * len(row)

st.dataframe(
    df.style.apply(style_row, axis=1).format({
        "xPts": "{:.1f}", "xPos": "{:.1f}",
        "Title %": "{:.1f}%", "Top 4 %": "{:.1f}%",
        "Top 6 %": "{:.1f}%", "Rel %": "{:.1f}%",
    }),
    use_container_width=True, height=700, hide_index=True
)
st.caption("🟡 Title  🟢 Champions League  🔵 Europa League  🔴 Relegation")

st.divider()


# ── Charts ──────────────────────────────────────────────────────────────────
cl, cr = st.columns(2)
with cl:
    st.subheader("🏆 Title Race")
    tdf = df[df["Title %"] > 0.1].sort_values("Title %", ascending=True)
    if tdf.empty: tdf = df.head(5).sort_values("Title %", ascending=True)
    fig = px.bar(tdf, x="Title %", y="Team", orientation="h",
                 color="Title %", color_continuous_scale=["#1a1a2e","#ffd700"], text="Title %")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, height=420, margin=dict(l=10,r=60,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

with cr:
    st.subheader("🔴 Relegation Risk")
    rdf = df[df["Rel %"] > 0.1].sort_values("Rel %", ascending=True)
    if rdf.empty: rdf = df.tail(5).sort_values("Rel %", ascending=True)
    fig2 = px.bar(rdf, x="Rel %", y="Team", orientation="h",
                  color="Rel %", color_continuous_scale=["#fff3e0","#f44336"], text="Rel %")
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       coloraxis_showscale=False, height=420, margin=dict(l=10,r=60,t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()


# ── Match Outcome Predictor ──────────────────────────────────────────────────
st.subheader("🎯 Match Outcome Predictor")
st.caption("Powered by Dixon-Coles model — works on cloud and locally")

params_dict = load_dc_params(league_code)
teams_list  = sorted(df["Team"].tolist())

mh, mv, ma = st.columns([5, 1, 5])
with mh:
    home = st.selectbox("🏠 Home Team", teams_list, key="home_sel")
with mv:
    st.markdown("<br><br><div style='text-align:center;font-size:24px;font-weight:bold;color:#ffd700'>VS</div>",
                unsafe_allow_html=True)
with ma:
    away = st.selectbox("✈️ Away Team", [t for t in teams_list if t != home], key="away_sel")

if st.button("⚽ Predict Match Outcome", type="primary", use_container_width=True):
    if params_dict is None:
        st.error("Model params not found. Make sure mlflow/artifacts/ is pushed to GitHub.")
    else:
        with st.spinner(f"Calculating {home} vs {away}..."):
            try:
                home_key = fuzzy_find(home, list(params_dict["attack"].keys()))
                away_key = fuzzy_find(away, list(params_dict["attack"].keys()))

                if not home_key or not away_key:
                    st.error(f"Could not find team in model params.")
                else:
                    probs = dc_match_probs(params_dict, home_key, away_key)
                    hw = probs["home_win"] * 100
                    dr = probs["draw"]     * 100
                    aw = probs["away_win"] * 100

                    st.markdown("### 📊 Match Outcome Probabilities")
                    r1, r2, r3 = st.columns(3)
                    r1.metric(f"🏠 {home}",  f"{hw:.1f}%", "Home Win")
                    r2.metric("🤝 Draw",     f"{dr:.1f}%", "")
                    r3.metric(f"✈️ {away}",  f"{aw:.1f}%", "Away Win")

                    fig3 = go.Figure(go.Bar(
                        x=[f"{home}\n(Home Win)", "Draw", f"{away}\n(Away Win)"],
                        y=[hw, dr, aw],
                        marker_color=["#4CAF50","#9E9E9E","#2196F3"],
                        text=[f"{hw:.1f}%", f"{dr:.1f}%", f"{aw:.1f}%"],
                        textposition="outside",
                        textfont=dict(size=16, color="white"),
                    ))
                    fig3.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(title="Probability (%)",
                                   range=[0, max(hw, dr, aw) + 15],
                                   gridcolor="rgba(255,255,255,0.1)"),
                        height=380, margin=dict(l=20,r=20,t=20,b=20), showlegend=False,
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                    if hw > dr and hw > aw:
                        verdict, color = f"🏠 **{home}** are favoured to win at home ({hw:.1f}%)", "#4CAF50"
                    elif aw > dr and aw > hw:
                        verdict, color = f"✈️ **{away}** are favoured to win away ({aw:.1f}%)", "#2196F3"
                    else:
                        verdict, color = f"🤝 A draw is the most likely outcome ({dr:.1f}%)", "#9E9E9E"

                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.05);"
                        f"border-left:4px solid {color};padding:12px 20px;"
                        f"border-radius:6px;margin-top:10px'>"
                        f"<b>Verdict:</b> {verdict}</div>",
                        unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#555;font-size:13px;padding:10px'>"
    "⚽ Football League Predictor &nbsp;|&nbsp; "
    "Dixon-Coles (1997) + Monte Carlo Simulation &nbsp;|&nbsp; "
    "Developed by <b style='color:#ffd700'>Aditya Yadav</b>"
    "</div>", unsafe_allow_html=True
)
