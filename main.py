"""
src/api/main.py
---------------
FastAPI app that serves league prediction results from the MLflow Model Registry.

Endpoints:
  GET  /health
  GET  /predictions/{league}
  GET  /predictions/{league}/{team}
  POST /predictions/refresh
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Config
with open("configs/model.yaml") as f:
    MODEL_CFG = yaml.safe_load(f)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Football League Predictor API",
    description="Probabilistic league standings using Dixon-Coles + Monte Carlo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory cache for predictions (refreshed per pipeline run)
# ─────────────────────────────────────────────────────────────────────────────

_predictions_cache: dict = {}
ARTIFACTS_DIR = Path("mlflow/artifacts")


def _load_predictions_from_disk(league: str) -> Optional[dict]:
    """Load latest simulation results from disk."""
    sim_path = ARTIFACTS_DIR / f"{league}_simulation.json"
    probs_path = ARTIFACTS_DIR / f"{league}_position_probs.csv"

    if not sim_path.exists() or not probs_path.exists():
        return None

    with open(sim_path) as f:
        sim_data = json.load(f)

    probs_df = pd.read_csv(probs_path, index_col="team")

    # Convert to serialisable format
    teams_data = {}
    for team in probs_df.index:
        row = probs_df.loc[team]
        team_dict = {
            "expected_position": float(sim_data["expected_position"].get(team, 0)),
            "expected_points": float(sim_data["expected_points"].get(team, 0)),
        }
        # Extract zone probabilities if present
        for col in ["P(title)", "P(top4)", "P(top6)", "P(relegation)"]:
            if col in probs_df.columns:
                team_dict[col] = float(row[col])

        # Position distribution
        pos_cols = [c for c in probs_df.columns if c.startswith("P(position=")]
        team_dict["position_distribution"] = {
            c: float(row[c]) for c in pos_cols
        }
        teams_data[team] = team_dict

    return {
        "league": league,
        "season": sim_data.get("season"),
        "generated_at": sim_data.get("generated_at"),
        "teams": teams_data,
        "sorted_by_expected_position": sorted(
            teams_data.keys(),
            key=lambda t: teams_data[t]["expected_position"]
        ),
    }


def _refresh_cache(league: str):
    data = _load_predictions_from_disk(league)
    if data:
        _predictions_cache[league] = data
        logger.info(f"Cache refreshed for {league}")
    else:
        logger.warning(f"No prediction data found for {league}")


# Warm cache on startup
@app.on_event("startup")
async def startup_event():
    for league in ["PL", "PD"]:
        _refresh_cache(league)


# ─────────────────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────────────────

class TeamPrediction(BaseModel):
    team: str
    expected_position: float
    expected_points: float
    p_title: Optional[float] = None
    p_top4: Optional[float] = None
    p_top6: Optional[float] = None
    p_relegation: Optional[float] = None

class LeaguePrediction(BaseModel):
    league: str
    season: Optional[int]
    generated_at: Optional[str]
    standings: list[TeamPrediction]

class MatchProbabilities(BaseModel):
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    leagues_cached: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        leagues_cached=list(_predictions_cache.keys()),
    )


@app.get("/predictions/{league}", response_model=LeaguePrediction)
async def get_league_predictions(league: str):
    """Return full predicted standings table for a league."""
    league = league.upper()
    if league not in _predictions_cache:
        _refresh_cache(league)
    if league not in _predictions_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for league '{league}'. Run the pipeline first."
        )

    data = _predictions_cache[league]
    standings = []
    for team in data["sorted_by_expected_position"]:
        t = data["teams"][team]
        standings.append(TeamPrediction(
            team=team,
            expected_position=t["expected_position"],
            expected_points=t["expected_points"],
            p_title=t.get("P(title)"),
            p_top4=t.get("P(top4)"),
            p_top6=t.get("P(top6)"),
            p_relegation=t.get("P(relegation)"),
        ))

    return LeaguePrediction(
        league=data["league"],
        season=data.get("season"),
        generated_at=data.get("generated_at"),
        standings=standings,
    )


@app.get("/predictions/{league}/{team}", response_model=TeamPrediction)
async def get_team_prediction(league: str, team: str):
    """Return prediction for a specific team."""
    league = league.upper()
    if league not in _predictions_cache:
        _refresh_cache(league)
    if league not in _predictions_cache:
        raise HTTPException(status_code=404, detail=f"No data for league '{league}'")

    teams = _predictions_cache[league]["teams"]

    # Case-insensitive team search
    matched = next(
        (t for t in teams if t.lower() == team.lower()), None
    )
    if not matched:
        # Try partial match
        matched = next(
            (t for t in teams if team.lower() in t.lower()), None
        )
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"Team '{team}' not found. Available: {sorted(teams.keys())}"
        )

    t = teams[matched]
    return TeamPrediction(
        team=matched,
        expected_position=t["expected_position"],
        expected_points=t["expected_points"],
        p_title=t.get("P(title)"),
        p_top4=t.get("P(top4)"),
        p_top6=t.get("P(top6)"),
        p_relegation=t.get("P(relegation)"),
    )


@app.post("/predictions/match")
async def predict_match(home_team: str, away_team: str, league: str = "PL") -> MatchProbabilities:
    """
    Predict outcome probabilities for a specific fixture.
    Uses the Production model from MLflow.
    """
    league = league.upper()
    model_name = MODEL_CFG["mlflow"]["registered_model_name"].replace("{league}", league)

    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        input_df = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
        result = model.predict(input_df)
        return MatchProbabilities(
            home_team=home_team,
            away_team=away_team,
            home_win=float(result["home_win"].iloc[0]),
            draw=float(result["draw"].iloc[0]),
            away_win=float(result["away_win"].iloc[0]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predictions/refresh")
async def refresh_predictions(background_tasks: BackgroundTasks, league: str = "all"):
    """Trigger cache refresh (or full pipeline rerun for 'all' leagues)."""
    leagues = ["PL", "PD"] if league == "all" else [league.upper()]
    for lg in leagues:
        background_tasks.add_task(_refresh_cache, lg)
    return {"status": "refresh triggered", "leagues": leagues}
