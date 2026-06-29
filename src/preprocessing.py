import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_FILE, TEST_SIZE, RANDOM_STATE
from src.utils import get_logger

log = get_logger("preprocessing")

def load_data() -> pd.DataFrame:
    log.info(f"Loading dataset from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    log.info(f"Dataset shape: {df.shape}")
    return df

def basic_info(df: pd.DataFrame):
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Shape       : {df.shape}")
    print(f"Columns     : {list(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nClass Distribution:\n{df['Class'].value_counts()}")
    fraud_pct = df['Class'].mean() * 100
    print(f"\nFraud percentage: {fraud_pct:.4f}%")
    print("="*60 + "\n")

def preprocess(df: pd.DataFrame):
    """Scale Amount & Time, split into train/test."""
    df = df.copy()

    # Scale Amount and Time (V1-V28 already PCA'd)
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )

    log.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    log.info(f"Train fraud count: {y_train.sum()}, Test fraud count: {y_test.sum()}")

    return X_train, X_test, y_train, y_test