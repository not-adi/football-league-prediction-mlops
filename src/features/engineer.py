import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
PROCESSED_DIR = Path('data/processed')

def compute_elo_ratings(df, initial_rating=1500, k=32, home_advantage=100):
    df = df.sort_values('date').copy()
    ratings = {}
    home_elo_pre, away_elo_pre = [], []
    for _, row in df.iterrows():
        h, a = row['home_team'], row['away_team']
        h_r = ratings.get(h, initial_rating)
        a_r = ratings.get(a, initial_rating)
        home_elo_pre.append(h_r)
        away_elo_pre.append(a_r)
        if row['status'] == 'FINISHED' and pd.notna(row.get('home_goals')):
            exp_h = 1.0 / (1.0 + 10 ** ((a_r - (h_r + home_advantage)) / 400))
            hg, ag = int(row['home_goals']), int(row['away_goals'])
            actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            ratings[h] = h_r + k * (actual_h - exp_h)
            ratings[a] = a_r + k * ((1 - actual_h) - (1 - exp_h))
    df['home_elo_pre'] = home_elo_pre
    df['away_elo_pre'] = away_elo_pre
    df['elo_diff'] = df['home_elo_pre'] - df['away_elo_pre']
    return df

def compute_rolling_form(df, window=5):
    finished = df[df['status'] == 'FINISHED'].copy()
    home_r = finished[['match_id','date','home_team','away_team','home_goals','away_goals']].copy()
    home_r.columns = ['match_id','date','team','opponent','goals_for','goals_against']
    home_r['is_home'] = True
    away_r = finished[['match_id','date','away_team','home_team','away_goals','home_goals']].copy()
    away_r.columns = ['match_id','date','team','opponent','goals_for','goals_against']
    away_r['is_home'] = False
    records = pd.concat([home_r, away_r], ignore_index=True).sort_values(['team','date'])
    records['points'] = np.where(records['goals_for'] > records['goals_against'], 3,
                         np.where(records['goals_for'] == records['goals_against'], 1, 0))
    def rolling(g):
        return g[['points','goals_for','goals_against']].shift(1).rolling(window, min_periods=1).mean()
    form = records.groupby('team', group_keys=False).apply(rolling)
    records['rolling_pts'] = form['points'].values
    records['rolling_gf'] = form['goals_for'].values
    records['rolling_ga'] = form['goals_against'].values
    home_form = records.rename(columns={'team':'home_team','rolling_pts':'home_form_pts',
                                        'rolling_gf':'home_form_gf','rolling_ga':'home_form_ga'})
    away_form = records.rename(columns={'team':'away_team','rolling_pts':'away_form_pts',
                                        'rolling_gf':'away_form_gf','rolling_ga':'away_form_ga'})
    df = df.merge(home_form[['match_id','home_team','home_form_pts','home_form_gf','home_form_ga']],
                  on=['match_id','home_team'], how='left')
    df = df.merge(away_form[['match_id','away_team','away_form_pts','away_form_gf','away_form_ga']],
                  on=['match_id','away_team'], how='left')
    return df

def add_season_progress(df):
    max_md = df['matchday'].max()
    df['season_progress'] = df['matchday'] / max_md
    return df

def build_features(df, league_code, elo_k=32, rolling_window=5, save=True):
    df = compute_elo_ratings(df, k=elo_k)
    df = compute_rolling_form(df, window=rolling_window)
    df = add_season_progress(df)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DIR / f'{league_code}_features.csv', index=False)
    return df

def load_features(league_code):
    path = PROCESSED_DIR / f'{league_code}_features.csv'
    return pd.read_csv(path, parse_dates=['date'])
