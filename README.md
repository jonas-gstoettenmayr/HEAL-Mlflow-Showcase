# HEAL MLflow Showcase

A short, self-hosted showcase of basic [MLflow](https://github.com/mlflow/mlflow) capabilities:

- **Postgres + MLflow server** via Docker Compose — tracking metadata and the model registry in Postgres, artifacts in a local folder.
- **`TheAnswer` (ONNX)** — a sklearn `LinearRegression` on a constant target, exported to ONNX with [`skl2onnx`](https://github.com/onnx/sklearn-onnx), registered in the MLflow Model Registry, then pulled back through the raw REST API and run with `onnxruntime`. - Shows how to store and load model files and exports.
- **`Decider` (sklearn)** — a grid-searched `DecisionTreeClassifier` pipeline on a small synthetic dataset, logged with dataset lineage and evaluation, then loaded via `models:/Decider/latest`. - Shows the logging capabilities and benefits of MLflow
- **Mistral autolog (optional)** — shows tracing with MLflow, example for Mistral

## Layout

```
├── compose.yaml               # postgres:15 + ghcr.io/mlflow/mlflow (vars via YAML anchors, no .env needed)
├── .python-version            # Python 3.14
├── pyproject.toml             # uv project + dependencies (uv_build backend)
├── src/
│   ├── data/data.csv          # synthetic dataset (auto-generated on first run if missing)
│   ├── train_onnx.py          # TheAnswer: train linear model -> ONNX -> register
│   ├── serve_through_api.py   # TheAnswer: newest version via REST API -> onnxruntime
│   ├── train_pyfunc.py         # Decider: grid-search classifier -> register (sklearn flavor)
│   ├── serve_pyfunc.py         # Decider: load models:/Decider/latest -> predict
│   ├── log_mistral.py          # optional: Mistral function-calling demo with autolog
│   └── heal_mlflow_showcase/   # helper package (config, data generation, ONNX export)
├── mlflow-artifacts/          # bind-mounted into the MLflow container (gitignored)
└── README.md
```

## Prerequisites

- Docker with Docker Compose (images `postgres:15` and `ghcr.io/mlflow/mlflow:latest` are pulled automatically)
- [uv](https://docs.astral.sh/uv/) for the Python environment (uses the version in `.python-version`)
- optional: a Mistral AI API key as `MISTRAL_API_KEY` for the autolog demo

## Quickstart

```bash
# 1. Python environment (creates .venv, installs the package + deps)
uv sync

# 2. start Postgres + MLflow server — UI at http://localhost:5000
docker compose up -d
docker compose ps            # wait until both services show "healthy"

# 3. TheAnswer: train (constant y = 42), export to ONNX, register
uv run python src/train_onnx.py 42
# serve the newest registered version via the REST API + onnxruntime
uv run python src/serve_through_api.py

# 4. Decider: train the classifier on the synthetic dataset, register as 'Decider'
uv run python src/train_pyfunc.py
# load models:/Decider/latest and predict
uv run python src/serve_pyfunc.py

# 5. optional: Mistral function-calling demo with autolog (needs MISTRAL_API_KEY)
uv run python src/log_mistral.py
```

Open http://localhost:5000 to inspect the experiments (`showcase`, `search`, `Mistral`), the runs, and the registered models (`TheAnswer`, `Decider`).

## What each script showcases

- **`src/train_onnx.py <value>`** — Shows how to export a model to ONNX and register it using MLflow. It uses the Registry and adds additional params. Run it again with a different value (e.g. `... 7`) to add a new registry version.
- **`src/serve_through_api.py`** — no MLflow client needed: calls the REST API directly (`get-latest-versions` → `get-download-uri`), downloads `model.onnx` through the server's artifact proxy, and runs one prediction with `onnxruntime`. Templates for languags without and mlflow packages on how to use the API to get the newest model version.
- **`src/train_pyfunc.py`** — Shows how to log the datasets and a classifier evaluation, and registers the as well as how MLflow interacts with hyperparameter search.
- **`src/serve_pyfunc.py`** — loading native python model in python with MLflow. Also shows the automatic input checking MLflow does.
- **`src/log_mistral.py`** — optional: a Mistral chat model and a tool call are traced. Requires `MISTRAL_API_KEY` in the environment.

All shared settings (tracking URI `http://localhost:5000`, experiment and model names, data file path) live in `src/heal_mlflow_showcase/config.py`.

## Where the data lives

- **Postgres** (named volume `pgdata`): experiments, runs, params, metrics, and the model registry.
- **`./mlflow-artifacts/`** (bind-mounted to `/mlflow/artifacts` in the MLflow container): the artifacts themselves — ONNX files and sklearn models. The server runs with `--serve-artifacts`, so clients only need port 5000.

## Teardown

```bash
docker compose down        # stop containers (keeps the pgdata volume)
docker compose down -v     # also delete the Postgres volume (wipes registry + runs)
```
