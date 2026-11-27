"""Shared configuration for the MLflow showcase"""
from pathlib import Path

TRACKING_URI = "http://localhost:5000"  # MLflow server started by compose.yaml
EXPERIMENT_NAME = "showcase"            # MLflow experiment that holds the training runs
MODEL_NAME = "TheAnswer"                # registered model name in the Model Registry
ARTIFACT_PATH = "onnx_model"            # ONNX artifact name inside the logged model

EXPERIMENT_NAME2 = "search"             # experiment of pyfunc
MODEL_NAME2 = "Decider"                 # name pyfunc model

SRC_DIR = Path(__file__).parent.parent # src directory
_data_dir = SRC_DIR/"data"                   # data dir
DATA_FILE = _data_dir/"data.csv"       # data file path

    