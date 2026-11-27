"""CLI wrapper for training: fit y = k*x + d, export to ONNX, register in MLflow.

Usage:
    uv run the-answer-train [K] [D]      # defaults: k=0, d=42 (constant 42)
    uv run python -m heal_mlflow_showcase.train [K] [D]
"""
import argparse

import mlflow
import mlflow.onnx as onnxflow
import mlflow.sklearn as skflow
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

from heal_mlflow_showcase import (
    EXPERIMENT_NAME,
    MODEL_NAME,
    TRACKING_URI,
    export_onnx,
    make_data_split_fixed_y,
    quiet_logger,
)


def parse_args() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("value", default=42, type=int)
    args = parser.parse_args()
    return args.value

def evaluate_regressen(y_true: ndarray, y_pred: ndarray):
    metrics = {}
    metrics["rmse"] = root_mean_squared_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)
    return metrics

def train_linear(value: int) -> None:
    """Parse [K] [D] from args and train + register the model."""
    with mlflow.start_run():
        
        #train model
        X_train, X_test, y_train, y_test = make_data_split_fixed_y(value)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        metrics = evaluate_regressen(y_test, model.predict(X_test))
        
        # Save the ONNX model
        onnx_model = export_onnx(model)
        
        signature = mlflow.models.infer_signature(X_train, y_train)
        metadata:mlflow.models.model.ModelInfo = onnxflow.log_model(
            onnx_model,
            signature = signature,
            name=f"TheAnswerIs{value}",
            registered_model_name=MODEL_NAME,
            onnx_execution_providers=['CPUExecutionProvider'],
            tags={"Type": type(model).__name__, "model_type": "Regressor", "predicts": "The truth and nothing but the truth"}
        )
        
        # record ml flow metrics
        mlflow.log_param("d", value)
        mlflow.log_metrics(metrics)
        print(f"Model saved under: {metadata.model_uri}")

def main() -> None:

    quiet_logger()
    
    #mlflow setup
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    skflow.autolog(log_models=False)
    
    train_linear(parse_args())


if __name__ == "__main__":
    main()
