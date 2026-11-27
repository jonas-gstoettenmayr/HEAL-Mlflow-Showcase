from datetime import datetime

import numpy as np
import onnxruntime as ort
import requests

from heal_mlflow_showcase.config import TRACKING_URI

MLFLOW_URI = TRACKING_URI + "/api/2.0"
REGISTRY = "TheAnswer"

    
# Get latest model metadata
response = requests.post(
    MLFLOW_URI+"/mlflow/registered-models/get-latest-versions",
    json={"name": REGISTRY, "stages": []}
)
if not response.ok: print(response.text)

metadata:dict = response.json()["model_versions"][0]
print(
    "Name: ", metadata["name"],
    " | Version: ", metadata["version"],
    " | Created: ", datetime.fromtimestamp(float(metadata["creation_timestamp"])//1000)
)

#find model uri path
version= metadata['version']
response = requests.get(
    MLFLOW_URI+f"/mlflow/model-versions/get-download-uri?name={REGISTRY}&version={version}"
)
if not response.ok: print(response.text)

# construct path to model onnx file
uri:str = response.json()["artifact_uri"]
uri = uri[uri.find("/")+1:]
path = f"/mlflow-artifacts/artifacts/{uri}/model.onnx"
response = requests.get(
    MLFLOW_URI+path
)
if not response.ok: print(response.text)

# testing the downloaded model
session = ort.InferenceSession(response.content)

inputs = np.array([[10]], dtype=np.float32)
print("Inputs: ", inputs.flatten())
preds = session.run(
    # session.get_outputs()[0],
    None,
    {session.get_inputs()[0].name: inputs}
)
print("Predictions: ", preds[0].flatten())  # ty: ignore[missing-union-attribute]
    