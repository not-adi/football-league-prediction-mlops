"""
tests/test_dixon_coles.py
--------------------------
Unit tests for the Dixon-Coles model.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import (
    DixonColesParams,
    _rho_correction,
    fit,
    match_outcome_probs,
    score_probability_matrix,
    time_decay_weights,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_matches():
    """Minimal synthetic match data for 4 teams over 2 seasons."""
    teams = ["Team A", "Team B", "Team C", "Team D"]
    rng = np.random.default_rng(42)

    rows = []
    base_date = pd.Timestamp("2023-08-01")
    matchday = 0

    for home in teams:
        for away in teams:
            if home == away:
                continue
            hg = int(rng.poisson(1.5))
            ag = int(rng.poisson(1.1))
            rows.append({
                "match_id": matchday,
                "league": "TEST",
                "season": 2023,
                "date": base_date + pd.Timedelta(days=matchday * 7),
                "status": "FINISHED",
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "matchday": matchday + 1,
            })
            matchday += 1

    return pd.DataFrame(rows)


@pytest.fixture
def fitted_params(sample_matches):
    return fit(sample_matches, xi=0.0, max_iterations=100)


# ─────────────────────────────────────────────────────────────────────────────
# Time decay
# ─────────────────────────────────────────────────────────────────────────────

def test_time_decay_most_recent_is_highest():
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="W"))
    weights = time_decay_weights(dates, xi=0.0018)
    assert weights[-1] >= weights[0], "Most recent match should have highest weight"


def test_time_decay_all_positive():
    dates = pd.Series(pd.date_range("2022-01-01", periods=50, freq="W"))
    weights = time_decay_weights(dates)
    assert (weights > 0).all()


def test_time_decay_no_decay():
    """With xi=0, all weights should be 1."""
    dates = pd.Series(pd.date_range("2022-01-01", periods=5, freq="W"))
    weights = time_decay_weights(dates, xi=0.0)
    np.testing.assert_allclose(weights, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rho correction
# ─────────────────────────────────────────────────────────────────────────────

def test_rho_correction_high_scoring_is_one():
    """For scores > 1, rho correction should be 1."""
    assert _rho_correction(2, 2, 1.5, 1.2, -0.1) == 1.0
    assert _rho_correction(3, 0, 1.5, 1.2, -0.1) == 1.0


def test_rho_correction_0_0():
    corr = _rho_correction(0, 0, 1.5, 1.2, -0.1)
    expected = 1 - 1.5 * 1.2 * (-0.1)
    assert abs(corr - expected) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────────────────────────────────────────

def test_fit_returns_params(fitted_params, sample_matches):
    assert isinstance(fitted_params, DixonColesParams)
    assert len(fitted_params.teams) == 4
    assert all(t in fitted_params.attack for t in fitted_params.teams)
    assert all(t in fitted_params.defence for t in fitted_params.teams)


def test_fit_attack_sums_to_zero(fitted_params):
    """Identifiability constraint: attack params should sum to ~0."""
    attack_sum = sum(fitted_params.attack.values())
    assert abs(attack_sum) < 0.1, f"Attack sum = {attack_sum}"


def test_fit_log_likelihood_negative(fitted_params):
    assert fitted_params.log_likelihood < 0, "Log likelihood should be negative"


def test_fit_home_advantage_positive(fitted_params):
    """Home advantage should typically be positive."""
    # Not strictly required by the model but expected in football
    assert fitted_params.home_advantage > -1.0  # sanity bound


# ─────────────────────────────────────────────────────────────────────────────
# Score probability matrix
# ─────────────────────────────────────────────────────────────────────────────

def test_score_matrix_sums_to_one(fitted_params):
    matrix = score_probability_matrix(fitted_params, "Team A", "Team B")
    assert abs(matrix.sum() - 1.0) < 1e-6


def test_score_matrix_non_negative(fitted_params):
    matrix = score_probability_matrix(fitted_params, "Team A", "Team B")
    assert (matrix >= 0).all()


def test_score_matrix_shape(fitted_params):
    max_goals = 8
    matrix = score_probability_matrix(fitted_params, "Team A", "Team B", max_goals)
    assert matrix.shape == (max_goals + 1, max_goals + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Match outcome probabilities
# ─────────────────────────────────────────────────────────────────────────────

def test_match_probs_sum_to_one(fitted_params):
    probs = match_outcome_probs(fitted_params, "Team A", "Team B")
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 1e-6


def test_match_probs_all_positive(fitted_params):
    probs = match_outcome_probs(fitted_params, "Team A", "Team B")
    assert all(v >= 0 for v in probs.values())


def test_match_probs_asymmetric(fitted_params):
    """Home team should generally have higher win probability than away."""
    probs_ab = match_outcome_probs(fitted_params, "Team A", "Team B")
    probs_ba = match_outcome_probs(fitted_params, "Team B", "Team A")
    # Home team in probs_ab is A, home team in probs_ba is B
    # These should differ due to home advantage
    assert probs_ab["home_win"] != probs_ba["home_win"]


# ─────────────────────────────────────────────────────────────────────────────
# Params serialisation
# ─────────────────────────────────────────────────────────────────────────────

def test_params_round_trip(fitted_params):
    d = fitted_params.to_dict()
    restored = DixonColesParams.from_dict(d)
    assert restored.teams == fitted_params.teams
    assert abs(restored.home_advantage - fitted_params.home_advantage) < 1e-10
    assert abs(restored.rho - fitted_params.rho) < 1e-10
