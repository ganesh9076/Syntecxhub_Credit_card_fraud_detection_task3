import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from src.utils import get_logger

log = get_logger("sampling")

def apply_sampling(X_train, y_train, strategy: str = "smote"):
    """
    Apply sampling strategy to handle class imbalance.

    strategy: "none" | "undersample" | "oversample" | "smote"
    """
    log.info(f"Before sampling: {Counter(y_train)}")

    if strategy == "none":
        log.info("No sampling applied.")
        return X_train, y_train

    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        log.info(f"RandomUnderSampler → {Counter(y_res)}")

    elif strategy == "oversample":
        sampler = RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        log.info(f"RandomOverSampler → {Counter(y_res)}")

    elif strategy == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        log.info(f"SMOTE → {Counter(y_res)}")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return X_res, y_res