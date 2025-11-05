import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

print("모델 구축 시작...")

features_df = pd.read_csv('../data/premier_league_features.csv')

feature_columns = [
    'h_win_rate', 'h_goals_avg', 'h_conceded_avg', 'h_home_win_rate', 'h_home_goals_avg',
    'a_win_rate', 'a_goals_avg', 'a_conceded_avg', 'a_away_win_rate', 'a_away_goals_avg',
    'win_rate_diff', 'goals_avg_diff'
]

X = features_df[feature_columns]
y = features_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"모델 정확도: {accuracy:.4f}")

joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✓ 모델 저장 완료: best_model.pkl, scaler.pkl")