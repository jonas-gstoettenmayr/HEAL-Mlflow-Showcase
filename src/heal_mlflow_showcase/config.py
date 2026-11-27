"""Shared configuration for the MLflow 'TheAnswer' showcase."""
from pathlib import Path

TRACKING_URI = "http://localhost:5000"  # MLflow server started by compose.yaml
EXPERIMENT_NAME = "showcase"            # MLflow experiment that holds the training runs
MODEL_NAME = "TheAnswer"                # registered model name in the Model Registry
ARTIFACT_PATH = "onnx_model"            # ONNX artifact name inside the logged model

EXPERIMENT_NAME2 = "search"
MODEL_NAME2 = "Decider"

SRC_DIR = Path(__file__).parent.parent
_data_dir = SRC_DIR/"data"
DATA_FILE = _data_dir/"data.csv"

    