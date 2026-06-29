from src.utils import setup_dirs, get_logger
from src.config import SAMPLING_STRATEGY, DECISION_THRESHOLD
from src.preprocessing import load_data, basic_info, preprocess
from src.eda import run_eda
from src.sampling import apply_sampling
from src.models import train_random_forest, train_xgboost
from src.evaluation import evaluate_model, compare_models
from src.threshold_analysis import plot_threshold_tradeoff

def main():
    setup_dirs()
    log = get_logger("main")
    log.info("=== Credit Card Fraud Detection Pipeline ===")

    # 1. Load & explore
    df = load_data()
    basic_info(df)

    # 2. EDA & Visualizations
    run_eda(df)

    # 3. Preprocessing
    X_train, X_test, y_train, y_test = preprocess(df)

    # 4. Handle class imbalance
    X_resampled, y_resampled = apply_sampling(X_train, y_train, SAMPLING_STRATEGY)

    # 5. Train models
    rf_model  = train_random_forest(X_resampled, y_resampled)
    xgb_model = train_xgboost(X_resampled, y_resampled)

    # 6. Evaluate
    results = {}
    results["Random Forest"] = evaluate_model(
        rf_model, X_test, y_test, "Random Forest", DECISION_THRESHOLD
    )
    results["XGBoost"] = evaluate_model(
        xgb_model, X_test, y_test, "XGBoost", DECISION_THRESHOLD
    )

    # 7. Compare models
    compare_models(results)

    # 8. Business threshold analysis
    plot_threshold_tradeoff(xgb_model, X_test, y_test, "XGBoost")

    log.info("=== Pipeline Complete. Check outputs/ for all plots. ===")

if __name__ == "__main__":
    main()