import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score

from heal_mlflow_showcase import quiet_logger
from heal_mlflow_showcase.config import MODEL_NAME2
from heal_mlflow_showcase.data import PROPOSER, STATUS

quiet_logger()

model_version = "latest"

mlflow.autolog()

model_uri = f"models:/{MODEL_NAME2}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

X_new= pd.DataFrame({
    "proposer": [PROPOSER[0],PROPOSER[1], PROPOSER[0]],
    "budget": [90.0,70.0,150.0]
})

y_new = [STATUS[i] for i in [0,1,1]]
y_pred = model.predict(X_new)

print(y_pred)
print(y_new)
print(f"Acc: {accuracy_score(y_new, y_pred):.2f}")
