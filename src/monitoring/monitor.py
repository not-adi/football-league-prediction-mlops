"""
src/monitoring/monitor.py
--------------------------
Post-match monitoring: tracks model accuracy over time and detects degradation.

Checks run weekly (via Prefect) or on-demand:
1. RPS degradation vs baseline
2. Calibration check (are predicted probabilities accurate?)
3. League table prediction accuracy (top 4 / relegation zone)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from src.models import dixon_coles as dc
from src.models.train import ranked_probability_score

logger = logging.getLogger(__name__)

MONITORING_DIR = Path("data/monitoring")
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_URI)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling RPS Tracker
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_rps(
    params: dc.DixonColesParams,
    new_matches: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Compute RPS for the most recent `window` completed matches.
    Returns a DataFrame of (date, rps, rolling_rps).
    """
    records = []
    for _, row in new_matches.iterrows():
        probs = dc.match_outcome_probs(params, row["home_team"], row["away_team"])
        p = np.array([probs["away_win"], probs["draw"], probs["home_win"]])

        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        if hg > ag:
            outcome = np.array([0, 0, 1])
        elif hg == ag:
            outcome = np.array([0, 1, 0])
        else:
            outcome = np.array([1, 0, 0])

        rps = float(np.sum((np.cumsum(p) - np.cumsum(outcome)) ** 2) / (len(p) - 1))
        records.append({"date": row["date"], "rps": rps})

    df = pd.DataFrame(records).sort_values("date")
    df["rolling_rps"] = df["rps"].rolling(window, min_periods=1).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Check
# ─────────────────────────────────────────────────────────────────────────────

def calibration_check(
    params: dc.DixonColesParams,
    matches: pd.DataFrame,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Group matches by predicted home win probability bucket,
    compare predicted vs actual win rate.

    A well-calibrated model has actual ≈ predicted in each bucket.
    """
    rows = []
    for _, row in matches.iterrows():
        if pd.isna(row.get("home_goals")):
            continue
        probs = dc.match_outcome_probs(params, row["home_team"], row["away_team"])
        actual_home_win = int(row["home_goals"]) > int(row["away_goals"])
        rows.append({
            "predicted_home_win": probs["home_win"],
            "actual_home_win": int(actual_home_win),
        })

    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df["predicted_home_win"], bins=n_bins)
    calibration = df.groupby("bin").agg(
        predicted_mean=("predicted_home_win", "mean"),
        actual_rate=("actual_home_win", "mean"),
        n_matches=("actual_home_win", "count"),
    ).reset_index()
    calibration["calibration_error"] = abs(
        calibration["predicted_mean"] - calibration["actual_rate"]
    )
    return calibration


# ─────────────────────────────────────────────────────────────────────────────
# Zone Prediction Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def zone_accuracy(
    predicted_probs_path: Path,
    actual_final_table: pd.DataFrame,
    n_teams: int = 20,
) -> dict:
    """
    Compare predicted zone probabilities to actual final standings.
    Returns accuracy scores for title / top4 / relegation predictions.
    """
    probs_df = pd.read_csv(predicted_probs_path, index_col="team")

    # Actual zones
    actual_table = actual_final_table.sort_values("pts", ascending=False).reset_index(drop=True)
    actual_table["position"] = range(1, len(actual_table) + 1)

    results = {}

    # Top 4 accuracy: teams predicted top 4 (>50% chance) that actually finished top 4
    if "P(top4)" in probs_df.columns:
        predicted_top4 = set(probs_df[probs_df["P(top4)"] > 0.5].index)
        actual_top4 = set(actual_table[actual_table["position"] <= 4]["team"])
        if predicted_top4:
            results["top4_precision"] = len(predicted_top4 & actual_top4) / len(predicted_top4)
            results["top4_recall"] = len(predicted_top4 & actual_top4) / max(len(actual_top4), 1)

    # Relegation accuracy
    if "P(relegation)" in probs_df.columns:
        predicted_rel = set(probs_df[probs_df["P(relegation)"] > 0.5].index)
        actual_rel = set(actual_table[actual_table["position"] >= n_teams - 2]["team"])
        if predicted_rel:
            results["relegation_precision"] = len(predicted_rel & actual_rel) / len(predicted_rel)
            results["relegation_recall"] = len(predicted_rel & actual_rel) / max(len(actual_rel), 1)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full Monitoring Run
# ─────────────────────────────────────────────────────────────────────────────

def run_monitoring(
    league_code: str,
    params: dc.DixonColesParams,
    recent_matches: pd.DataFrame,
    rps_alert_threshold: float = 0.05,
) -> dict:
    """
    Run all monitoring checks and log results to MLflow.
    Returns a summary dict with alert flags.
    """
    mlflow.set_experiment(f"football-monitoring-{league_code}")

    with mlflow.start_run(run_name=f"monitor_{datetime.now().strftime('%Y%m%d')}"):

        # --- RPS rolling ---
        rps_df = compute_rolling_rps(params, recent_matches)
        current_rps = float(rps_df["rolling_rps"].iloc[-1])
        mlflow.log_metric("current_rolling_rps", current_rps)

        # Save RPS history
        rps_path = MONITORING_DIR / f"{league_code}_rps_history.csv"
        rps_df.to_csv(rps_path, index=False)
        mlflow.log_artifact(str(rps_path))

        # --- Calibration ---
        calibration = calibration_check(params, recent_matches)
        mean_cal_error = float(calibration["calibration_error"].mean())
        mlflow.log_metric("mean_calibration_error", mean_cal_error)

        cal_path = MONITORING_DIR / f"{league_code}_calibration.csv"
        calibration.to_csv(cal_path, index=False)
        mlflow.log_artifact(str(cal_path))

        # --- Alerts ---
        alerts = []
        if current_rps > rps_alert_threshold:
            alerts.append(f"RPS {current_rps:.4f} exceeds threshold {rps_alert_threshold}")
            logger.warning(f"[{league_code}] ALERT: {alerts[-1]}")

        if mean_cal_error > 0.1:
            alerts.append(f"Mean calibration error {mean_cal_error:.4f} > 0.10")
            logger.warning(f"[{league_code}] ALERT: {alerts[-1]}")

        mlflow.set_tag("n_alerts", str(len(alerts)))
        mlflow.set_tag("alerts", "; ".join(alerts) if alerts else "none")

        summary = {
            "league": league_code,
            "current_rolling_rps": current_rps,
            "mean_calibration_error": mean_cal_error,
            "alerts": alerts,
            "status": "ALERT" if alerts else "OK",
            "checked_at": datetime.utcnow().isoformat(),
        }

        report_path = MONITORING_DIR / f"{league_code}_monitoring_report.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(str(report_path))

        logger.info(f"[{league_code}] Monitoring: {summary['status']}")
        return summary
