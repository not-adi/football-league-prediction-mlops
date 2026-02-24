"""
src/models/train.py
-------------------
MLflow-instrumented training pipeline (simplified - no pyfunc model registration).
Logs params, metrics, and artifacts cleanly to MLflow.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.data.ingest import get_completed_matches, get_upcoming_matches
from src.features.engineer import load_features
from src.models import dixon_coles as dc
from src.models.simulator import simulate_season

logger = logging.getLogger(__name__)

with open("configs/model.yaml") as f:
    MODEL_CFG = yaml.safe_load(f)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = MODEL_CFG["mlflow"]["experiment_name"]
ARTIFACTS_DIR = Path("mlflow/artifacts")


def ranked_probability_score(params: dc.DixonColesParams, test_matches: pd.DataFrame) -> float:
    """Ranked Probability Score — lower is better."""
    rps_scores = []
    for _, row in test_matches.iterrows():
        try:
            probs = dc.match_outcome_probs(params, row["home_team"], row["away_team"])
            p = np.array([probs["away_win"], probs["draw"], probs["home_win"]])
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            if hg > ag:   outcome = np.array([0, 0, 1])
            elif hg == ag: outcome = np.array([0, 1, 0])
            else:          outcome = np.array([1, 0, 0])
            rps = np.sum((np.cumsum(p) - np.cumsum(outcome)) ** 2) / (len(p) - 1)
            rps_scores.append(rps)
        except Exception:
            continue
    return float(np.mean(rps_scores)) if rps_scores else 0.5


def train_and_log(
    league_code: str,
    current_season: int,
    n_simulations: int = 10_000,
    promote_threshold: float = 0.02,
) -> str:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"[{league_code}] Starting training run for season {current_season}")

    # Load data
    df = load_features(league_code)
    completed = get_completed_matches(df)
    current_df = df[df["season"] == current_season]
    remaining = get_upcoming_matches(current_df)
    current_completed = get_completed_matches(current_df)

    logger.info(f"[{league_code}] Total completed: {len(completed)}, "
                f"Current season completed: {len(current_completed)}, "
                f"Remaining: {len(remaining)}")

    # Train/test split
    n_train = int(len(completed) * 0.9)
    train_df = completed.iloc[:n_train]
    test_df  = completed.iloc[n_train:]

    xi       = MODEL_CFG["dixon_coles"]["xi"]
    max_iter = MODEL_CFG["dixon_coles"]["max_iterations"]
    seed     = MODEL_CFG["simulation"]["random_seed"]

    with mlflow.start_run(
        run_name=f"{league_code}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ) as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_params({
            "league": league_code,
            "season": current_season,
            "xi": xi,
            "max_iterations": max_iter,
            "n_simulations": n_simulations,
            "train_matches": len(train_df),
            "test_matches": len(test_df),
        })

        # Fit model
        logger.info(f"[{league_code}] Fitting Dixon-Coles on {len(train_df)} matches...")
        params = dc.fit(train_df, xi=xi, max_iterations=max_iter)

        # Evaluate
        rps = ranked_probability_score(params, test_df)
        logger.info(f"[{league_code}] RPS: {rps:.4f}")

        mlflow.log_metrics({
            "log_likelihood": params.log_likelihood,
            "rps_test": rps,
            "home_advantage": params.home_advantage,
            "rho": params.rho,
            "n_teams": len(params.teams),
        })

        # Simulate season
        logger.info(f"[{league_code}] Running {n_simulations:,} simulations...")
        sim_output = simulate_season(
            params=params,
            completed_matches=current_completed,
            remaining_fixtures=remaining,
            n_simulations=n_simulations,
            random_seed=seed,
        )

        pos_probs = sim_output["position_probs"]

        # Save artifacts locally
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        params_path = ARTIFACTS_DIR / f"{league_code}_params.json"
        sim_path    = ARTIFACTS_DIR / f"{league_code}_simulation.json"
        probs_path  = ARTIFACTS_DIR / f"{league_code}_position_probs.csv"

        with open(params_path, "w") as f:
            json.dump(params.to_dict(), f, indent=2)

        with open(sim_path, "w") as f:
            json.dump({
                "expected_points":   sim_output["expected_points"],
                "expected_position": sim_output["expected_position"],
                "generated_at":      datetime.utcnow().isoformat(),
                "league":            league_code,
                "season":            current_season,
            }, f, indent=2)

        pos_probs.to_csv(probs_path)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(params_path),  artifact_path="model")
        mlflow.log_artifact(str(sim_path),     artifact_path="simulation")
        mlflow.log_artifact(str(probs_path),   artifact_path="simulation")

        mlflow.set_tags({
            "league":     league_code,
            "season":     current_season,
            "model_type": "dixon_coles",
        })

        logger.info(f"[{league_code}] ✅ Run complete → {run_id}")
        return run_id
