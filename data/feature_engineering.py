import requests
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

api_key = "6d20a6abe6574a6887ef68d4336a4fbd"
headers = {"X-Auth-Token": api_key}

print("데이터 수집 중...")

response = requests.get("https://api.football-data.org/v4/competitions/PL/matches", headers=headers)
matches_data = response.json()

matches_list = []
for match in matches_data['matches']:
    matches_list.append({
        'matchId': match['id'],
        'utcDate': match['utcDate'],
        'matchday': match['matchday'],
        'homeTeam': match['homeTeam']['name'],
        'awayTeam': match['awayTeam']['name'],
        'homeGoals': match['score']['fullTime']['home'],
        'awayGoals': match['score']['fullTime']['away'],
        'result': match['score']['winner'],
        'status': match['status']
    })

matches_df = pd.DataFrame(matches_list)
matches_df['utcDate'] = pd.to_datetime(matches_df['utcDate'])
finished_matches = matches_df[matches_df['status'] == 'FINISHED'].copy()

response = requests.get("https://api.football-data.org/v4/competitions/PL/standings", headers=headers)
standings_data = response.json()

standings_list = []
for row in standings_data['standings'][0]['table']:
    standings_list.append({
        'teamName': row['team']['name'],
        'playedGames': row['playedGames'],
        'won': row['won'],
        'draw': row['draw'],
        'lost': row['lost'],
        'points': row['points'],
        'goalFor': row['goalsFor'],
        'goalAgainst': row['goalsAgainst'],
        'goalDifference': row['goalDifference']
    })

standings_df = pd.DataFrame(standings_list)

print("특성 엔지니어링 시작...")

team_stats = {}

for team in standings_df['teamName'].unique():
    home_matches = finished_matches[finished_matches['homeTeam'] == team]
    away_matches = finished_matches[finished_matches['awayTeam'] == team]

    home_wins = len(home_matches[home_matches['result'] == 'HOME_TEAM'])
    home_games = len(home_matches)
    away_wins = len(away_matches[away_matches['result'] == 'AWAY_TEAM'])
    away_games = len(away_matches)

    team_stats[team] = {
        'total_games': home_games + away_games,
        'total_wins': home_wins + away_wins,
        'home_games': home_games,
        'home_wins': home_wins,
        'away_games': away_games,
        'away_wins': away_wins,
        'total_goals_for': home_matches['homeGoals'].sum() + away_matches['awayGoals'].sum(),
        'total_goals_against': home_matches['awayGoals'].sum() + away_matches['homeGoals'].sum(),
        'home_goals_for': home_matches['homeGoals'].sum(),
        'home_goals_against': home_matches['awayGoals'].sum(),
        'away_goals_for': away_matches['awayGoals'].sum(),
        'away_goals_against': away_matches['homeGoals'].sum(),
    }

def calculate_features(home_team, away_team, team_stats, finished_matches):
    h_stats = team_stats.get(home_team, {})
    a_stats = team_stats.get(away_team, {})

    h_win_rate = h_stats.get('total_wins', 0) / max(h_stats.get('total_games', 1), 1)
    h_goals_avg = h_stats.get('total_goals_for', 0) / max(h_stats.get('total_games', 1), 1)
    h_conceded_avg = h_stats.get('total_goals_against', 0) / max(h_stats.get('total_games', 1), 1)
    h_home_win_rate = h_stats.get('home_wins', 0) / max(h_stats.get('home_games', 1), 1)
    h_home_goals_avg = h_stats.get('home_goals_for', 0) / max(h_stats.get('home_games', 1), 1)

    a_win_rate = a_stats.get('total_wins', 0) / max(a_stats.get('total_games', 1), 1)
    a_goals_avg = a_stats.get('total_goals_for', 0) / max(a_stats.get('total_games', 1), 1)
    a_conceded_avg = a_stats.get('total_goals_against', 0) / max(a_stats.get('total_games', 1), 1)
    a_away_win_rate = a_stats.get('away_wins', 0) / max(a_stats.get('away_games', 1), 1)
    a_away_goals_avg = a_stats.get('away_goals_for', 0) / max(a_stats.get('away_games', 1), 1)

    win_rate_diff = h_win_rate - a_win_rate
    goals_avg_diff = h_goals_avg - a_goals_avg

    return {
        'h_win_rate': h_win_rate,
        'h_goals_avg': h_goals_avg,
        'h_conceded_avg': h_conceded_avg,
        'h_home_win_rate': h_home_win_rate,
        'h_home_goals_avg': h_home_goals_avg,
        'a_win_rate': a_win_rate,
        'a_goals_avg': a_goals_avg,
        'a_conceded_avg': a_conceded_avg,
        'a_away_win_rate': a_away_win_rate,
        'a_away_goals_avg': a_away_goals_avg,
        'win_rate_diff': win_rate_diff,
        'goals_avg_diff': goals_avg_diff
    }

features_list = []

for idx, match in finished_matches.iterrows():
    home_team = match['homeTeam']
    away_team = match['awayTeam']

    features = calculate_features(home_team, away_team, team_stats, finished_matches)
    features['homeTeam'] = home_team
    features['awayTeam'] = away_team
    features['homeGoals'] = match['homeGoals']
    features['awayGoals'] = match['awayGoals']
    features['result'] = match['result']
    features['matchday'] = match['matchday']

    features_list.append(features)

features_df = pd.DataFrame(features_list)

features_df['target'] = features_df['result'].map({
    'HOME_TEAM': 1,
    'DRAW': 0,
    'AWAY_TEAM': -1
})

features_df.to_csv('premier_league_features.csv', index=False)
print("✓ 특성 엔지니어링 완료: premier_league_features.csv")