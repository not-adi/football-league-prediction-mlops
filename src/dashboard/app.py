"""
src/dashboard/app.py
Football League Predictor — Developed by Aditya Yadav
Cloud-compatible: reads from static JSON if API unavailable
"""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import json
from pathlib import Path

API_URL = os.getenv("API_URL", "http://app:8000")
IS_CLOUD = not Path("mlflow/artifacts").exists()

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

    league_options = {
        "Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿": "PL",
        "La Liga 🇪🇸": "PD",
    }
    selected_league_name = st.selectbox("League", list(league_options.keys()))
    league_code = league_options[selected_league_name]

    st.divider()

    # Only show refresh button when running locally
    if not IS_CLOUD:
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
        st.info("🌐 Live on Streamlit Cloud\nPredictions updated weekly", icon="ℹ️")

    st.divider()
    st.caption("Data: football-data.org")
    st.caption("Model: Dixon-Coles (1997)")
    st.caption("Simulations: 10,000 per run")
    st.divider()
    st.markdown("**Developed by**")
    st.markdown("### Aditya Yadav")


# ── Data fetching ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_from_api(league: str) -> dict:
    resp = requests.get(f"{API_URL}/predictions/{league}", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def fetch_from_static(league: str) -> dict:
    """Fallback: read prediction JSON files committed to the repo."""
    sim_path   = Path(f"mlflow/artifacts/{league}_simulation.json")
    probs_path = Path(f"mlflow/artifacts/{league}_position_probs.csv")

    if not sim_path.exists() or not probs_path.exists():
        return None

    with open(sim_path) as f:
        sim_data = json.load(f)

    probs_df = pd.read_csv(probs_path, index_col="team")
    standings = []

    sorted_teams = sorted(
        probs_df.index,
        key=lambda t: sim_data["expected_position"].get(t, 99)
    )

    for team in sorted_teams:
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

    return {
        "league":       league,
        "season":       sim_data.get("season"),
        "generated_at": sim_data.get("generated_at"),
        "standings":    standings,
    }


def fetch_predictions(league: str) -> dict:
    """Try API first, fall back to static files."""
    try:
        return fetch_from_api(league)
    except Exception:
        data = fetch_from_static(league)
        if data:
            return data
        raise Exception(
            "No predictions available. Run the pipeline locally first:\n"
            "docker compose exec app python pipelines/run_pipeline.py --leagues PL,PD --season 2025"
        )


try:
    data = fetch_predictions(league_code)
except Exception as e:
    st.error(f"⚠️ Could not load predictions: {e}")
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

col1, col2, col3, col4 = st.columns(4)
with col1:
    leader = df.iloc[0]
    st.metric("🏆 Title Favourite", leader["Team"], f"{leader['Title %']}% chance")
with col2:
    top4_leader = df.sort_values("Top 4 %", ascending=False).iloc[0]
    st.metric("🟢 Top 4 Favourite", top4_leader["Team"], f"{top4_leader['Top 4 %']}%")
with col3:
    rel_danger = df.sort_values("Rel %", ascending=False).iloc[0]
    st.metric("🔴 Relegation Risk", rel_danger["Team"], f"{rel_danger['Rel %']}%")
with col4:
    st.metric("📊 Teams", len(df), "2025/26 season")

st.divider()


# ── Standings table ─────────────────────────────────────────────────────────
st.subheader("📋 Predicted Final Standings")

def style_row(row):
    pos = row["Pos"]
    n   = len(df)
    if pos == 1:       return ["background-color: #ffd70033"] * len(row)
    elif pos <= 4:     return ["background-color: #4CAF5022"] * len(row)
    elif pos <= 6:     return ["background-color: #2196F322"] * len(row)
    elif pos >= n - 2: return ["background-color: #f4433622"] * len(row)
    return [""] * len(row)

styled = df.style.apply(style_row, axis=1).format({
    "xPts": "{:.1f}", "xPos": "{:.1f}",
    "Title %": "{:.1f}%", "Top 4 %": "{:.1f}%",
    "Top 6 %": "{:.1f}%", "Rel %": "{:.1f}%",
})
st.dataframe(styled, use_container_width=True, height=700, hide_index=True)
st.caption("🟡 Title  🟢 Champions League  🔵 Europa League  🔴 Relegation")


# ── Charts ──────────────────────────────────────────────────────────────────
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🏆 Title Race Probability")
    title_df = df[df["Title %"] > 0.1].sort_values("Title %", ascending=True)
    if title_df.empty:
        title_df = df.head(5).sort_values("Title %", ascending=True)
    fig = px.bar(title_df, x="Title %", y="Team", orientation="h",
                 color="Title %", color_continuous_scale=["#1a1a2e", "#ffd700"], text="Title %")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, height=420,
                      margin=dict(l=10, r=60, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🔴 Relegation Risk")
    rel_df = df[df["Rel %"] > 0.1].sort_values("Rel %", ascending=True)
    if rel_df.empty:
        rel_df = df.tail(5).sort_values("Rel %", ascending=True)
    fig2 = px.bar(rel_df, x="Rel %", y="Team", orientation="h",
                  color="Rel %", color_continuous_scale=["#fff3e0", "#f44336"], text="Rel %")
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       coloraxis_showscale=False, height=420,
                       margin=dict(l=10, r=60, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)


# ── Match Outcome Predictor ──────────────────────────────────────────────────
st.divider()
st.subheader("🎯 Match Outcome Predictor")
st.caption("Select two teams to get Dixon-Coles win/draw/loss probabilities")

if IS_CLOUD:
    st.warning("⚠️ Live match prediction requires the local API. Showing static standings only.", icon="⚠️")
else:
    teams_list = sorted(df["Team"].tolist())
    col_home, col_vs, col_away = st.columns([5, 1, 5])
    with col_home:
        home = st.selectbox("🏠 Home Team", teams_list, key="home_sel")
    with col_vs:
        st.markdown("<br><br><div style='text-align:center; font-size:24px; font-weight:bold; color:#ffd700'>VS</div>",
                    unsafe_allow_html=True)
    with col_away:
        away = st.selectbox("✈️ Away Team", [t for t in teams_list if t != home], key="away_sel")

    if st.button("⚽ Predict Match Outcome", type="primary", use_container_width=True):
        with st.spinner(f"Calculating {home} vs {away}..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predictions/match",
                    params={"home_team": home, "away_team": away, "league": league_code},
                    timeout=15,
                )
                resp.raise_for_status()
                probs = resp.json()
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
                    marker_color=["#4CAF50", "#9E9E9E", "#2196F3"],
                    text=[f"{hw:.1f}%", f"{dr:.1f}%", f"{aw:.1f}%"],
                    textposition="outside",
                    textfont=dict(size=16, color="white"),
                ))
                fig3.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Probability (%)", range=[0, max(hw, dr, aw) + 15],
                               gridcolor="rgba(255,255,255,0.1)"),
                    height=380, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

                if hw > dr and hw > aw:
                    verdict, color = f"🏠 **{home}** are favoured to win at home ({hw:.1f}%)", "#4CAF50"
                elif aw > dr and aw > hw:
                    verdict, color = f"✈️ **{away}** are favoured to win away ({aw:.1f}%)", "#2196F3"
                else:
                    verdict, color = f"🤝 A draw is the most likely outcome ({dr:.1f}%)", "#9E9E9E"

                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.05); border-left:4px solid {color};"
                    f"padding:12px 20px; border-radius:6px; margin-top:10px'>"
                    f"<b>Verdict:</b> {verdict}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#555; font-size:13px; padding:10px'>"
    "⚽ Football League Predictor &nbsp;|&nbsp; "
    "Dixon-Coles (1997) + Monte Carlo Simulation &nbsp;|&nbsp; "
    "Developed by <b style='color:#ffd700'>Aditya Yadav</b>"
    "</div>", unsafe_allow_html=True
)
