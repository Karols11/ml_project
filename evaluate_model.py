import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

def load_model(model_path):
    model = joblib.load(model_path)
    return model


df = pd.read_csv('data.csv')
X = df[['feature_1', 'feature_2']]
y = df['target']

# Load the model
model = load_model('model.pkl')  # Provide the correct path to your model file

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

with open('metrics.txt', 'w') as f:
    f.write(f"accuracy: {accuracy}\n")
    f.write(f"precision: {precision}\n")
    f.write(f"recall: {recall}\n")