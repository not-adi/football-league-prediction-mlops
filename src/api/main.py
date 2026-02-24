"""
src/api/main.py
FastAPI app serving league predictions and match outcome probabilities.
Match predictor reads directly from saved params JSON (no MLflow registry needed).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("mlflow/artifacts")

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

# ── In-memory cache ────────────────────────────────────────────────────────
_predictions_cache: dict = {}


def _load_predictions_from_disk(league: str) -> Optional[dict]:
    sim_path   = ARTIFACTS_DIR / f"{league}_simulation.json"
    probs_path = ARTIFACTS_DIR / f"{league}_position_probs.csv"

    if not sim_path.exists() or not probs_path.exists():
        logger.warning(f"No prediction files found for {league}")
        return None

    with open(sim_path) as f:
        sim_data = json.load(f)

    probs_df = pd.read_csv(probs_path, index_col="team")

    teams_data = {}
    for team in probs_df.index:
        row = probs_df.loc[team]
        team_dict = {
            "expected_position": float(sim_data["expected_position"].get(team, 0)),
            "expected_points":   float(sim_data["expected_points"].get(team, 0)),
        }
        for col in ["P(title)", "P(top4)", "P(top6)", "P(relegation)"]:
            team_dict[col] = float(row[col]) if col in probs_df.columns else None

        pos_cols = [c for c in probs_df.columns if c.startswith("P(position=")]
        team_dict["position_distribution"] = {c: float(row[c]) for c in pos_cols}
        teams_data[team] = team_dict

    return {
        "league":      league,
        "season":      sim_data.get("season"),
        "generated_at": sim_data.get("generated_at"),
        "teams":       teams_data,
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


@app.on_event("startup")
async def startup_event():
    for league in ["PL", "PD"]:
        _refresh_cache(league)


# ── Response models ────────────────────────────────────────────────────────
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


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        leagues_cached=list(_predictions_cache.keys()),
    )


@app.get("/predictions/{league}", response_model=LeaguePrediction)
async def get_league_predictions(league: str):
    league = league.upper()
    if league not in _predictions_cache:
        _refresh_cache(league)
    if league not in _predictions_cache:
        raise HTTPException(status_code=404,
            detail=f"No predictions for '{league}'. Run the pipeline first.")

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
    league = league.upper()
    if league not in _predictions_cache:
        _refresh_cache(league)
    if league not in _predictions_cache:
        raise HTTPException(status_code=404, detail=f"No data for '{league}'")

    teams = _predictions_cache[league]["teams"]
    matched = next((t for t in teams if t.lower() == team.lower()), None)
    if not matched:
        matched = next((t for t in teams if team.lower() in t.lower()), None)
    if not matched:
        raise HTTPException(status_code=404,
            detail=f"Team '{team}' not found. Available: {sorted(teams.keys())}")

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


@app.post("/predictions/match", response_model=MatchProbabilities)
async def predict_match(home_team: str, away_team: str, league: str = "PL"):
    """
    Predict match outcome using saved Dixon-Coles params.
    No MLflow registry needed — reads directly from artifacts JSON.
    """
    league = league.upper()
    params_path = ARTIFACTS_DIR / f"{league}_params.json"

    if not params_path.exists():
        raise HTTPException(status_code=404,
            detail=f"No model params found for {league}. Run the pipeline first.")

    try:
        # Import here to avoid circular issues at startup
        from src.models.dixon_coles import DixonColesParams, match_outcome_probs

        with open(params_path) as f:
            params = DixonColesParams.from_dict(json.load(f))

        # Fuzzy match team names
        def find_team(name: str) -> str:
            exact = next((t for t in params.teams if t.lower() == name.lower()), None)
            if exact:
                return exact
            partial = next((t for t in params.teams if name.lower() in t.lower()), None)
            if partial:
                return partial
            raise HTTPException(status_code=404,
                detail=f"Team '{name}' not found in model. "
                       f"Available teams: {sorted(params.teams)}")

        home_matched = find_team(home_team)
        away_matched = find_team(away_team)

        probs = match_outcome_probs(params, home_matched, away_matched)

        return MatchProbabilities(
            home_team=home_matched,
            away_team=away_matched,
            home_win=round(probs["home_win"], 4),
            draw=round(probs["draw"], 4),
            away_win=round(probs["away_win"], 4),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Match prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictions/refresh")
async def refresh_predictions(background_tasks: BackgroundTasks, league: str = "all"):
    leagues = ["PL", "PD"] if league == "all" else [league.upper()]
    for lg in leagues:
        background_tasks.add_task(_refresh_cache, lg)
    return {"status": "refresh triggered", "leagues": leagues}
