import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('data/t20_wc_matches.csv')
df = df.dropna()
df['target'] = (df['toss_winner'] == df['match_winner']).astype(int)

teams = list(set(df['toss_winner'].unique()) | set(df['match_winner'].unique()))
team_map = {team: idx for idx, team in enumerate(teams)}
df['toss_winner_code'] = df['toss_winner'].map(team_map)
df['venue_code'] = df['venue'].astype('category').cat.codes

X = df[['toss_winner_code', 'venue_code']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"✅ Model accuracy: {acc:.2%}")

joblib.dump(model, 'predictor_model.pkl')
joblib.dump(team_map, 'team_map.pkl')
joblib.dump(df['venue'].astype('category').cat.categories.tolist(), 'venue_list.pkl')
