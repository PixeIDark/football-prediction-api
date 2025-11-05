from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Premier League Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("모델 로드 중...")
try:
    model = joblib.load('../model/best_model.pkl')
    scaler = joblib.load('../model/scaler.pkl')
    features_df = pd.read_csv('../data/premier_league_features.csv')
    print("✓ 모델 로드 완료")
except Exception as e:
    print(f"✗ 모델 로드 실패: {e}")
    model = None
    scaler = None
    features_df = None

team_stats = {}
if features_df is not None:
    for team in features_df['homeTeam'].unique():
        home_matches = features_df[features_df['homeTeam'] == team]
        away_matches = features_df[features_df['awayTeam'] == team]

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

    print(f"✓ {len(team_stats)}개 팀 통계 계산 완료")

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_result: str
    confidence: float

def calculate_features(home_team: str, away_team: str):
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

@app.get("/")
async def root():
    return {"message": "Premier League Prediction API"}

@app.get("/teams")
async def get_teams():
    teams = sorted(list(team_stats.keys()))
    return {"teams": teams, "count": len(teams)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    if model is None or scaler is None:
        return {"error": "모델이 로드되지 않았습니다"}

    home_team = request.home_team
    away_team = request.away_team

    if home_team not in team_stats:
        return {"error": f"팀을 찾을 수 없습니다: {home_team}"}
    if away_team not in team_stats:
        return {"error": f"팀을 찾을 수 없습니다: {away_team}"}

    features = calculate_features(home_team, away_team)

    feature_array = np.array([[
        features['h_win_rate'],
        features['h_goals_avg'],
        features['h_conceded_avg'],
        features['h_home_win_rate'],
        features['h_home_goals_avg'],
        features['a_win_rate'],
        features['a_goals_avg'],
        features['a_conceded_avg'],
        features['a_away_win_rate'],
        features['a_away_goals_avg'],
        features['win_rate_diff'],
        features['goals_avg_diff']
    ]])

    features_scaled = scaler.transform(feature_array)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    class_to_label = {-1: 'AWAY_TEAM', 0: 'DRAW', 1: 'HOME_TEAM'}
    predicted_result = class_to_label[prediction]

    away_prob = probabilities[0]
    draw_prob = probabilities[1]
    home_prob = probabilities[2]
    confidence = max(probabilities)

    return PredictionResponse(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=round(float(home_prob), 4),
        draw_prob=round(float(draw_prob), 4),
        away_win_prob=round(float(away_prob), 4),
        predicted_result=predicted_result,
        confidence=round(float(confidence), 4)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)