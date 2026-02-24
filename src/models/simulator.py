"""
src/models/simulator.py
-----------------------
Monte Carlo season simulator using Dixon-Coles match probabilities.
Fixed: zone probabilities now work for any league size.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesParams, score_probability_matrix

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def simulate_match(
    params: DixonColesParams,
    home_team: str,
    away_team: str,
    rng: np.random.Generator,
    max_goals: int = 10,
) -> tuple:
    matrix = score_probability_matrix(params, home_team, away_team, max_goals)
    flat = matrix.flatten()
    flat = np.maximum(flat, 0)
    flat /= flat.sum()
    idx = rng.choice(len(flat), p=flat)
    return int(idx // (max_goals + 1)), int(idx % (max_goals + 1))


def compute_table(results: pd.DataFrame, teams: list) -> pd.DataFrame:
    stats = {t: {"played": 0, "won": 0, "drawn": 0, "lost": 0,
                 "gf": 0, "ga": 0, "pts": 0} for t in teams}

    for _, row in results.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in stats or a not in stats:
            continue
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        stats[h]["played"] += 1
        stats[h]["gf"] += hg
        stats[h]["ga"] += ag
        stats[a]["played"] += 1
        stats[a]["gf"] += ag
        stats[a]["ga"] += hg

        if hg > ag:
            stats[h]["won"] += 1; stats[h]["pts"] += 3; stats[a]["lost"] += 1
        elif hg < ag:
            stats[a]["won"] += 1; stats[a]["pts"] += 3; stats[h]["lost"] += 1
        else:
            stats[h]["drawn"] += 1; stats[h]["pts"] += 1
            stats[a]["drawn"] += 1; stats[a]["pts"] += 1

    table = pd.DataFrame(stats).T.reset_index().rename(columns={"index": "team"})
    table["gd"] = table["gf"] - table["ga"]
    table = table.sort_values(["pts", "gd", "gf"], ascending=[False, False, False]).reset_index(drop=True)
    table["position"] = range(1, len(table) + 1)
    return table


def simulate_season(
    params: DixonColesParams,
    completed_matches: pd.DataFrame,
    remaining_fixtures: pd.DataFrame,
    n_simulations: int = 10_000,
    random_seed: int = 42,
) -> dict:
    # Use only teams that appear in current season fixtures
    current_teams = set()
    if not completed_matches.empty:
        current_teams |= set(completed_matches["home_team"]) | set(completed_matches["away_team"])
    if not remaining_fixtures.empty:
        current_teams |= set(remaining_fixtures["home_team"]) | set(remaining_fixtures["away_team"])

    # Filter to only teams the model knows about
    teams = sorted([t for t in current_teams if t in params.teams])
    n_teams = len(teams)

    logger.info(f"Simulating {n_teams} teams: {teams[:5]}...")

    rng = np.random.default_rng(random_seed)
    position_counts = {t: np.zeros(n_teams, dtype=int) for t in teams}
    points_sum = {t: 0.0 for t in teams}

    logger.info(f"Running {n_simulations:,} simulations | {len(remaining_fixtures)} fixtures remaining")

    for _ in tqdm(range(n_simulations), desc="Simulating", unit="sim"):
        sim_results = []
        for _, fixture in remaining_fixtures.iterrows():
            h, a = fixture["home_team"], fixture["away_team"]
            if h in params.teams and a in params.teams:
                hg, ag = simulate_match(params, h, a, rng)
                sim_results.append({"home_team": h, "away_team": a,
                                    "home_goals": hg, "away_goals": ag})

        all_results = pd.concat([
            completed_matches[["home_team", "away_team", "home_goals", "away_goals"]],
            pd.DataFrame(sim_results) if sim_results else pd.DataFrame(
                columns=["home_team", "away_team", "home_goals", "away_goals"]),
        ], ignore_index=True)

        table = compute_table(all_results, teams)

        for _, row in table.iterrows():
            t = row["team"]
            if t in position_counts:
                pos = int(row["position"]) - 1
                position_counts[t][pos] += 1
                points_sum[t] += row["pts"]

    # Build position probability dataframe
    position_probs = pd.DataFrame(
        {t: position_counts[t] / n_simulations for t in teams}
    ).T
    position_probs.columns = [f"P(position={i+1})" for i in range(n_teams)]
    position_probs.index.name = "team"

    expected_points   = {t: points_sum[t] / n_simulations for t in teams}
    expected_position = {
        t: float(np.dot(position_counts[t] / n_simulations, range(1, n_teams + 1)))
        for t in teams
    }

    probs_df = position_probs.copy()
    probs_df["expected_position"] = pd.Series(expected_position)
    probs_df["expected_points"]   = pd.Series(expected_points)

    # ── Zone probabilities — adaptive to league size ──────────────────────
    # Title
    probs_df["P(title)"] = position_probs["P(position=1)"]

    # Top 4 (Champions League) — top 4 or top 20% whichever is smaller
    top4_n = min(4, max(1, round(n_teams * 0.20)))
    probs_df["P(top4)"] = position_probs[
        [f"P(position={i})" for i in range(1, top4_n + 1)]
    ].sum(axis=1)

    # Top 6 (European places)
    top6_n = min(6, max(1, round(n_teams * 0.30)))
    probs_df["P(top6)"] = position_probs[
        [f"P(position={i})" for i in range(1, top6_n + 1)]
    ].sum(axis=1)

    # Relegation — bottom 3 or bottom 15% whichever is smaller
    rel_n = min(3, max(1, round(n_teams * 0.15)))
    probs_df["P(relegation)"] = position_probs[
        [f"P(position={i})" for i in range(n_teams - rel_n + 1, n_teams + 1)]
    ].sum(axis=1)

    probs_df = probs_df.sort_values("expected_position")

    logger.info(f"Simulation complete. Teams: {n_teams}, Title fav: {probs_df.index[0]}")

    return {
        "position_probs":   probs_df,
        "expected_points":  expected_points,
        "expected_position": expected_position,
    }
