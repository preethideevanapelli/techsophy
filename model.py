from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model(X, y, save_path='model.joblib'):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, save_path)
    return model

def load_model(path='model.joblib'):
    return joblib.load(path)

def predict(model, X):
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return proba, pred
