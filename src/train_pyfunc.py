import mlflow
import mlflow.sklearn as skflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from heal_mlflow_showcase import (
    DATA_FILE,
    EXPERIMENT_NAME2,
    MODEL_NAME2,
    TRACKING_URI,
    make_data_set,
    quiet_logger,
    split_data_set,
)

if not DATA_FILE.exists():
    make_data_set()
    
def make_model(X_train, y_train) -> Pipeline:
    
    transformer = ColumnTransformer(
        [
            ("category", OneHotEncoder(), ["proposer"]),
            ("numerical", StandardScaler(), ["budget"])
        ]
    )
    pipe = Pipeline([
        ("process", transformer),
        ("model", DecisionTreeClassifier(random_state=42))
    ])
    search = GridSearchCV(
            estimator=pipe,
            param_grid={
                "model__max_depth": [1, 2, 3],
                "model__min_samples_split": [2, 3, 4],  
            },
            cv=3,
        )
    search.fit(X_train, y_train)
    return search.best_estimator_

def eval_classifier(y_true, y_pred)-> dict[str, float]:
    metrics = {}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["roc"] = roc_auc_score(y_true, y_pred)
    return metrics

def train_model():
    data = pd.read_csv(DATA_FILE)
    X_train, X_test, y_train, y_test = split_data_set(data)
    
    with mlflow.start_run():
        # register training set
        mlflow.log_artifact(str(DATA_FILE), artifact_path="data")
        train_set = mlflow.data.from_pandas(  # ty: ignore[missing-attribute]
            pd.concat([X_train, y_train], axis=1),
            source=DATA_FILE,
            targets=y_train.name,
            name="train_set",
        )
        mlflow.log_input(train_set, "training")

        # register best model
        model = make_model(X_train, y_train)
        mlflow.sklearn.log_model(
            model, name=MODEL_NAME2,
            registered_model_name=MODEL_NAME2,
            input_example=X_test,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            tags={"type": type(model).__name__, "predictor": type(model["model"]).__name__}
        )
        
        # regsiter test set and test set metrics
        eval_set = mlflow.data.from_pandas(  # ty: ignore[missing-attribute]
            pd.concat([X_test, y_test], axis=1),
            source=DATA_FILE,
            targets=y_test.name,
            name="test_set",
        )
        mlflow.log_input(eval_set, "test")
        
        mlflow.models.evaluate(
            model=f"runs:/{mlflow.active_run().info.run_id}/{MODEL_NAME2}",  # ty: ignore[missing-union-attribute]
            data=eval_set,
            model_type="classifier"
        )

def main():
    
    quiet_logger()
        
    #mlflow setup
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME2)
    skflow.autolog(log_models=False, log_datasets=False)
    
    train_model()
    

if __name__=="__main__":
    main()