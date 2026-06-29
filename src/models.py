import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.config import MODEL_DIR, RF_PARAMS, XGB_PARAMS
from src.utils import get_logger

log = get_logger("models")

def train_random_forest(X_train, y_train):
    log.info("Training Random Forest...")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    path = os.path.join(MODEL_DIR, "random_forest.pkl")
    joblib.dump(model, path)
    log.info(f"Random Forest saved → {path}")
    return model

def train_xgboost(X_train, y_train):
    log.info("Training XGBoost...")
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    params = {**XGB_PARAMS, "scale_pos_weight": scale}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    path = os.path.join(MODEL_DIR, "xgboost.pkl")
    joblib.dump(model, path)
    log.info(f"XGBoost saved → {path}")
    return model

def load_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    return joblib.load(path)