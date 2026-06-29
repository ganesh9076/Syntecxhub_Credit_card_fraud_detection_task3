import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

DATA_FILE   = os.path.join(DATA_DIR, "creditcard.csv")

# ── Model Config ───────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Sampling strategies ────────────────────────────────
# Options: "none" | "undersample" | "oversample" | "smote"
SAMPLING_STRATEGY = "smote"

# ── Business threshold for final decision ──────────────
# Lower = catch more fraud (higher recall, lower precision)
DECISION_THRESHOLD = 0.3

# ── XGBoost Params ─────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":   300,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":    "logloss",
    "random_state":   RANDOM_STATE,
}

# ── Random Forest Params ───────────────────────────────
RF_PARAMS = {
    "n_estimators":  300,
    "max_depth":     10,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}