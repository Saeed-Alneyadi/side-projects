from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model(model_type: str = "rf") -> Pipeline:
    if model_type == "rf":
        return Pipeline([("clf", RandomForestClassifier(n_estimators=200))])
    if model_type == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    raise ValueError(f"Unknown model_type: {model_type}")