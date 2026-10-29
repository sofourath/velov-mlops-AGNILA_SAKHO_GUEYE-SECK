from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
EXPERIMENT_NAME = "velov-availability"


def git_state() -> dict[str, str]:
    """Retourne le commit courant et indique si le dépôt contient des changements."""
    git = shutil.which("git")
    if git is None:
        return {"git_commit": "unknown", "git_dirty": "unknown"}

    try:
        commit = subprocess.run(  # noqa: S603 - executable and args are trusted project-local git paths
            [git, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=10,
        )
        commit_text = commit.stdout.strip() if commit.returncode == 0 else "unknown"
    except Exception:
        commit_text = "unknown"

    try:
        status = subprocess.run(  # noqa: S603 - executable and args are trusted project-local git paths
            [git, "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=10,
        )
        dirty = "true" if status.returncode == 0 and status.stdout.strip() else "false"
    except Exception:
        dirty = "unknown"

    return {
        "git_commit": commit_text,
        "git_dirty": dirty,
    }


def _build_input_example(model: object, metadata: dict) -> pd.DataFrame:
    """Construit un exemple d'entrée pour générer la signature MLflow."""

    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        feature_names = metadata.get("features", [])

    if not feature_names:
        raise ValueError(
            "Impossible de déterminer les colonnes d'entrée du modèle "
            "pour créer la signature MLflow."
        )

    row: dict[str, object] = {}

    for feature in feature_names:
        if feature == "station_id":
            row[feature] = 3
        elif feature in {"is_raining", "is_weekend"}:
            row[feature] = False
        elif feature in {"hour", "day_of_week"}:
            row[feature] = 12
        elif feature == "capacity":
            row[feature] = 30
        elif feature == "bikes_available":
            row[feature] = 12
        elif feature == "occupancy_rate":
            row[feature] = 0.4
        elif feature == "temperature":
            row[feature] = 15.0
        else:
            row[feature] = 0.0

    return pd.DataFrame([row])


def log_to_mlflow(metadata: dict, model_dir: Path) -> str:
    """Journalise un entraînement Vélo'v dans MLflow et renvoie l'ID du run."""

    model_dir = Path(model_dir)

    model_path = model_dir / MODEL_FILENAME
    metadata_path = model_dir / METADATA_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} introuvable")

    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} introuvable")

    model = joblib.load(model_path)

    params = metadata.get("params", {})
    metrics = metadata.get("metrics", {})

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"v{metadata.get('model_version', 'unknown')}") as run:
        for name in ("seed", "learning_rate", "max_iter", "test_days"):
            if name in params:
                mlflow.log_param(name, params[name])

        for name in (
            "mae_model",
            "mae_baseline_persistence",
            "mae_model_peak_hours",
        ):
            if name in metrics:
                mlflow.log_metric(name, float(metrics[name]))

        tags = git_state()

        artifact = metadata.get("artifact", {})
        data = metadata.get("data", {})

        fingerprint = (
            data.get("sha256")
            or data.get("fingerprint")
            or metadata.get("data_sha256")
            or artifact.get("data_sha256")
        )

        data_start = data.get("start") or metadata.get("data_start")
        data_end = data.get("end") or metadata.get("data_end")

        if fingerprint is not None:
            tags["data_sha256"] = str(fingerprint)

        if data_start is not None:
            tags["data_start"] = str(data_start)

        if data_end is not None:
            tags["data_end"] = str(data_end)

        mlflow.set_tags(tags)

        input_example = _build_input_example(model=model, metadata=metadata)
        predictions = model.predict(input_example)
        signature = infer_signature(input_example, predictions)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        mlflow.log_artifact(str(model_path), artifact_path="package")
        mlflow.log_artifact(str(metadata_path), artifact_path="package")

    return run.info.run_id


def register_model(
    model_name: str = "velov-availability",
    *,
    run_id: str | None = None,
    alias: str | None = "challenger",
) -> str:
    """Enregistre le modèle associé à un run dans le registre MLflow et lui donne un alias."""
    if run_id is None:
        active = mlflow.active_run()
        if active is None:
            raise ValueError("Aucun run MLflow actif pour l'enregistrement du modèle.")
        run_id = active.info.run_id

    model_uri = f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri=model_uri, name=model_name)

    if alias is not None:
        mlflow.set_registered_model_alias(model_name, alias, version.version)

    return version.version


def evaluate_promotion(
    model_metrics: dict[str, float],
    *,
    require_peak_hours: bool = False,
) -> tuple[bool, str]:
    """Trois conditions de promotion: métriques, MAE meilleure que la baseline et contrôle d'heures de pointe."""
    if "mae_model" not in model_metrics:
        return False, "missing mae_model"

    if "mae_baseline_persistence" not in model_metrics:
        return False, "missing mae_baseline_persistence"

    if float(model_metrics["mae_model"]) >= float(model_metrics["mae_baseline_persistence"]):
        return False, "model_not_better_than_baseline"

    if require_peak_hours and "mae_model_peak_hours" in model_metrics:
        peak_hours_mae = float(model_metrics["mae_model_peak_hours"])
        if peak_hours_mae > float(model_metrics["mae_model"]) * 1.5:
            return False, "peak_hour_mae_too_high"

    return True, "eligible"


def should_promote(model_metrics: dict[str, float], *, require_peak_hours: bool = False) -> bool:
    """Vrai si le challenger respecte les 3 conditions de promotion."""
    return evaluate_promotion(model_metrics, require_peak_hours=require_peak_hours)[0]


def promote_model(
    model_name: str = "velov-availability",
    *,
    challenger_alias: str = "challenger",
    champion_alias: str = "champion",
) -> str | None:
    """Promet le challenger vers le champion si la règle de promotion est respectée."""
    client = MlflowClient()

    try:
        challenger = client.get_model_version_by_alias(model_name, challenger_alias)
    except Exception:
        return None

    run = client.get_run(challenger.run_id)
    metrics = dict(run.data.metrics)
    eligible, _ = evaluate_promotion(metrics)

    if not eligible:
        return None

    client.set_registered_model_alias(model_name, champion_alias, challenger.version)
    return challenger.version
