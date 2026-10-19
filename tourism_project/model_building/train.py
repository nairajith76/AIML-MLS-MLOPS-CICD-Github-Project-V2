"""Model training with hyperparameter tuning and MLflow experiment tracking."""
import json
import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPLIT_DIR = os.getenv("SPLIT_DIR", ".")
MODEL_DIR = os.getenv("MODEL_DIR", "tourism_project/deployment")
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")
THRESHOLD = 0.45  # slightly below 0.5 to favour recall on the minority (buyer) class

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-wellness-package")

Xtrain = pd.read_csv(os.path.join(SPLIT_DIR, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(SPLIT_DIR, "Xtest.csv"))
ytrain = pd.read_csv(os.path.join(SPLIT_DIR, "ytrain.csv")).squeeze("columns")
ytest = pd.read_csv(os.path.join(SPLIT_DIR, "ytest.csv")).squeeze("columns")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Handle class imbalance
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
    "xgbclassifier__reg_lambda": [1, 5],
}


def metrics_for(y_true, y_pred, y_prob, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_prob),
    }


with mlflow.start_run(run_name="xgb-gridsearch"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # Log every hyperparameter combination as a nested run
    results = grid.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"candidate-{i}", nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

    mlflow.log_params(grid.best_params_)
    mlflow.log_param("classification_threshold", THRESHOLD)
    mlflow.log_metric("best_cv_f1", grid.best_score_)

    best_model = grid.best_estimator_

    train_prob = best_model.predict_proba(Xtrain)[:, 1]
    test_prob = best_model.predict_proba(Xtest)[:, 1]
    train_pred = (train_prob >= THRESHOLD).astype(int)
    test_pred = (test_prob >= THRESHOLD).astype(int)

    metrics = {
        **metrics_for(ytrain, train_pred, train_prob, "train"),
        **metrics_for(ytest, test_pred, test_prob, "test"),
    }
    mlflow.log_metrics(metrics)

    print("Best params:", grid.best_params_)
    print("\nTest classification report:")
    print(classification_report(ytest, test_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(
            {"threshold": THRESHOLD, "best_params": grid.best_params_,
             "test_metrics": {k: round(v, 4) for k, v in metrics.items() if k.startswith("test")}},
            f, indent=2,
        )
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    print(f"\nModel saved to {MODEL_PATH}")
