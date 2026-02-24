import os, json, time, logging
from pathlib import Path
from typing import Optional
import requests
import pandas as pd

logger = logging.getLogger(__name__)
BASE_URL = 'https://api.football-data.org/v4'
RAW_DATA_DIR = Path('data/raw')

class FootballDataClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ['FOOTBALL_DATA_API_KEY']
        self.session = requests.Session()
        self.session.headers.update({'X-Auth-Token': self.api_key})
        self._rate_limit_delay = 6

    def _get(self, endpoint, params=None):
        url = f'{BASE_URL}/{endpoint}'
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        time.sleep(self._rate_limit_delay)
        return resp.json()

    def get_matches(self, league_code, season):
        return self._get(f'competitions/{league_code}/matches', params={'season': season})

def fetch_and_save(league_code, seasons, client=None):
    if client is None:
        client = FootballDataClient()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_matches = []
    for season in seasons:
        raw = client.get_matches(league_code, season)
        json_path = RAW_DATA_DIR / f'{league_code}_{season}_raw.json'
        with open(json_path, 'w') as f:
            json.dump(raw, f, indent=2)
        for match in raw.get('matches', []):
            score = match.get('score', {})
            ft = score.get('fullTime', {})
            all_matches.append({
                'match_id': match['id'], 'league': league_code, 'season': season,
                'date': match['utcDate'], 'status': match['status'],
                'home_team': match['homeTeam']['name'], 'away_team': match['awayTeam']['name'],
                'home_goals': ft.get('home'), 'away_goals': ft.get('away'),
                'matchday': match.get('matchday'),
            })
    df = pd.DataFrame(all_matches)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    out_path = RAW_DATA_DIR / f'{league_code}_matches.csv'
    df.to_csv(out_path, index=False)
    return out_path

def load_raw_matches(league_code):
    path = RAW_DATA_DIR / f'{league_code}_matches.csv'
    return pd.read_csv(path, parse_dates=['date'])

def get_completed_matches(df):
    return df[df['status'] == 'FINISHED'].dropna(subset=['home_goals', 'away_goals']).copy()

def get_upcoming_matches(df):
    return df[df['status'].isin(['SCHEDULED', 'TIMED'])].copy()
