"""
pipelines/run_pipeline.py
Full football prediction pipeline using Prefect 2.x (latest compatible)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import yaml
from prefect import flow, task, get_run_logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import FootballDataClient, fetch_and_save, load_raw_matches
from src.data.validate import validate_raw_matches
from src.features.engineer import build_features
from src.models.train import train_and_log

with open("configs/leagues.yaml") as f:
    LEAGUES_CFG = yaml.safe_load(f)["leagues"]

with open("configs/model.yaml") as f:
    MODEL_CFG = yaml.safe_load(f)


@task(name="ingest-data", retries=2, retry_delay_seconds=30)
def ingest_task(league_code: str, seasons: List[int]) -> str:
    logger = get_run_logger()
    logger.info(f"[{league_code}] Ingesting seasons: {seasons}")
    client = FootballDataClient()
    out_path = fetch_and_save(league_code, seasons, client=client)
    logger.info(f"[{league_code}] Raw data saved → {out_path}")
    return str(out_path)


@task(name="validate-data")
def validate_task(league_code: str) -> bool:
    logger = get_run_logger()
    df = load_raw_matches(league_code)
    report = validate_raw_matches(df, league_code)
    logger.info(f"[{league_code}] Validation: {report.summary()}")
    return True


@task(name="feature-engineering")
def feature_task(league_code: str) -> str:
    logger = get_run_logger()
    df = load_raw_matches(league_code)
    enriched = build_features(
        df,
        league_code=league_code,
        elo_k=MODEL_CFG["feature_engineering"]["elo_k_factor"],
        rolling_window=MODEL_CFG["feature_engineering"]["rolling_window_matches"],
        save=True,
    )
    logger.info(f"[{league_code}] Features done → {len(enriched)} rows")
    return f"data/processed/{league_code}_features.csv"


@task(name="train-and-log", retries=1)
def train_task(league_code: str, current_season: int) -> str:
    logger = get_run_logger()
    logger.info(f"[{league_code}] Training for season {current_season}")
    run_id = train_and_log(
        league_code=league_code,
        current_season=current_season,
        n_simulations=MODEL_CFG["simulation"]["n_simulations"],
        promote_threshold=MODEL_CFG["mlflow"]["promotion_accuracy_delta"],
    )
    logger.info(f"[{league_code}] MLflow run: {run_id}")
    return run_id


@flow(
    name="football-prediction-pipeline",
    description="ingest → validate → features → train → simulate",
)
def football_pipeline(
    leagues: List[str],
    current_season: int,
    seasons_window: int = 3,
):
    seasons = list(range(current_season - seasons_window + 1, current_season + 1))
    print(f"Running pipeline for seasons: {seasons}")

    for league in leagues:
        ingest_task(league, seasons)
        validate_task(league)
        feature_task(league)
        train_task(league, current_season)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", type=str, default="PL,PD")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--seasons-window", type=int, default=3)
    args = parser.parse_args()

    leagues = [l.strip() for l in args.leagues.split(",")]
    print(f"Starting pipeline: leagues={leagues}, season={args.season}")

    football_pipeline(
        leagues=leagues,
        current_season=args.season,
        seasons_window=args.seasons_window,
    )
