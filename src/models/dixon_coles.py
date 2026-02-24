"""
src/models/dixon_coles.py
--------------------------
Dixon-Coles (1997) Poisson model for football match prediction.

The model estimates:
- attack[team]   : attacking strength parameter
- defence[team]  : defensive weakness parameter
- home_advantage : global home field bonus
- rho            : low-scoring correction factor

Reference:
  Dixon, M.J. & Coles, S.G. (1997). Modelling Association Football Scores
  and Inefficiencies in the Football Betting Market. Applied Statistics.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Time-decay weighting
# ─────────────────────────────────────────────────────────────────────────────

def time_decay_weights(dates: pd.Series, xi: float = 0.0018) -> np.ndarray:
    """
    Exponential time-decay weights.
    xi=0.0018 gives ~half-weight to matches ~385 days ago.
    """
    ref_date = dates.max()
    days_ago = (ref_date - dates).dt.days.values
    return np.exp(-xi * days_ago)


# ─────────────────────────────────────────────────────────────────────────────
# Rho correction for low-scoring outcomes
# ─────────────────────────────────────────────────────────────────────────────

def _rho_correction(home_goals: int, away_goals: int,
                    lambda_h: float, lambda_a: float, rho: float) -> float:
    """Dixon-Coles correction for 0-0, 1-0, 0-1, 1-1 score lines."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_h * lambda_a * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_h * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Model parameters container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DixonColesParams:
    teams: list[str]
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.3
    rho: float = -0.1
    log_likelihood: float = float("-inf")

    def lambda_home(self, home_team: str, away_team: str) -> float:
        return np.exp(
            self.attack[home_team] - self.defence[away_team] + self.home_advantage
        )

    def lambda_away(self, home_team: str, away_team: str) -> float:
        return np.exp(self.attack[away_team] - self.defence[home_team])

    def to_dict(self) -> dict:
        return {
            "teams": self.teams,
            "attack": self.attack,
            "defence": self.defence,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "log_likelihood": self.log_likelihood,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DixonColesParams":
        p = cls(teams=d["teams"])
        p.attack = d["attack"]
        p.defence = d["defence"]
        p.home_advantage = d["home_advantage"]
        p.rho = d["rho"]
        p.log_likelihood = d.get("log_likelihood", float("-inf"))
        return p


# ─────────────────────────────────────────────────────────────────────────────
# Negative log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

def _neg_log_likelihood(
    params_vec: np.ndarray,
    teams: list[str],
    home_teams: np.ndarray,
    away_teams: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
) -> float:
    n = len(teams)
    attack = dict(zip(teams, params_vec[:n]))
    defence = dict(zip(teams, params_vec[n: 2 * n]))
    home_adv = params_vec[2 * n]
    rho = params_vec[2 * n + 1]

    log_lik = 0.0
    for i in range(len(home_teams)):
        h, a = home_teams[i], away_teams[i]
        hg, ag = int(home_goals[i]), int(away_goals[i])
        w = weights[i]

        lam_h = np.exp(attack[h] - defence[a] + home_adv)
        lam_a = np.exp(attack[a] - defence[h])

        rho_corr = _rho_correction(hg, ag, lam_h, lam_a, rho)
        if rho_corr <= 0:
            return 1e10  # infeasible

        log_lik += w * (
            np.log(rho_corr)
            + poisson.logpmf(hg, lam_h)
            + poisson.logpmf(ag, lam_a)
        )

    return -log_lik


# ─────────────────────────────────────────────────────────────────────────────
# Fit
# ─────────────────────────────────────────────────────────────────────────────

def fit(
    df: pd.DataFrame,
    xi: float = 0.0018,
    method: str = "L-BFGS-B",
    max_iterations: int = 500,
) -> DixonColesParams:
    """
    Fit Dixon-Coles model on a DataFrame of completed matches.

    Parameters
    ----------
    df      : must have columns date, home_team, away_team, home_goals, away_goals
    xi      : time-decay parameter
    method  : scipy optimisation method
    max_iterations : optimiser iteration cap

    Returns
    -------
    DixonColesParams with fitted attack/defence/home_advantage/rho
    """
    completed = df.dropna(subset=["home_goals", "away_goals"]).copy()
    completed["home_goals"] = completed["home_goals"].astype(int)
    completed["away_goals"] = completed["away_goals"].astype(int)

    teams = sorted(
        set(completed["home_team"].unique()) | set(completed["away_team"].unique())
    )
    n = len(teams)
    logger.info(f"Fitting Dixon-Coles on {len(completed)} matches, {n} teams")

    weights = time_decay_weights(completed["date"], xi=xi)

    # Initial parameter vector: [attack x n, defence x n, home_adv, rho]
    x0 = np.concatenate([
        np.zeros(n),        # attack (log scale, 0 = neutral)
        np.zeros(n),        # defence (log scale, 0 = neutral)
        [0.3],              # home advantage
        [-0.1],             # rho correction
    ])

    # Constraint: sum of attack params = 0 (identifiability)
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n])}
    ]

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(
            teams,
            completed["home_team"].values,
            completed["away_team"].values,
            completed["home_goals"].values,
            completed["away_goals"].values,
            weights,
        ),
        method="SLSQP",  # supports equality constraints
        constraints=constraints,
        options={"maxiter": max_iterations, "disp": False},
    )

    if not result.success:
        logger.warning(f"Optimisation did not converge: {result.message}")

    fitted = DixonColesParams(teams=teams)
    fitted.attack = dict(zip(teams, result.x[:n]))
    fitted.defence = dict(zip(teams, result.x[n: 2 * n]))
    fitted.home_advantage = result.x[2 * n]
    fitted.rho = result.x[2 * n + 1]
    fitted.log_likelihood = -result.fun

    logger.info(
        f"Fit complete | LL={fitted.log_likelihood:.2f} | "
        f"home_adv={fitted.home_advantage:.3f} | rho={fitted.rho:.3f}"
    )
    return fitted


# ─────────────────────────────────────────────────────────────────────────────
# Score probability matrix
# ─────────────────────────────────────────────────────────────────────────────

def score_probability_matrix(
    params: DixonColesParams,
    home_team: str,
    away_team: str,
    max_goals: int = 10,
) -> np.ndarray:
    """
    Return (max_goals+1 x max_goals+1) matrix where [i,j] is P(home=i, away=j).
    """
    lam_h = params.lambda_home(home_team, away_team)
    lam_a = params.lambda_away(home_team, away_team)

    matrix = np.outer(
        poisson.pmf(range(max_goals + 1), lam_h),
        poisson.pmf(range(max_goals + 1), lam_a),
    )

    # Apply Dixon-Coles rho correction for low-scoring cells
    for hg in range(2):
        for ag in range(2):
            corr = _rho_correction(hg, ag, lam_h, lam_a, params.rho)
            matrix[hg, ag] *= corr

    # Renormalise to ensure probabilities sum to 1
    matrix /= matrix.sum()
    return matrix


def match_outcome_probs(
    params: DixonColesParams,
    home_team: str,
    away_team: str,
) -> dict[str, float]:
    """Return home_win / draw / away_win probabilities for a single match."""
    matrix = score_probability_matrix(params, home_team, away_team)
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    return {"home_win": home_win, "draw": draw, "away_win": away_win}
