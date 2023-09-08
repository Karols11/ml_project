import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('data.csv')
X = df[['feature_1', 'feature_2']]
y = df['target']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
