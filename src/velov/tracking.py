from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
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
        raise ValueError("Impossible de déterminer les colonnes d'entrée du modèle pour créer la signature MLflow.")

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

    params = metadata.get("params", metadata.get("parameters", {}))
    metrics = metadata.get("metrics", {})

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"v{metadata.get('model_version', 'unknown')}") as run:
        if metadata.get("model_version") is not None:
            mlflow.log_param("model_version", metadata["model_version"])
        for name in ("seed", "learning_rate", "max_iter", "test_days", "gold_table"):
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
            data.get("sha256") or data.get("fingerprint") or metadata.get("data_sha256") or artifact.get("data_sha256")
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
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None = None,
    *,
    same_window: bool = True,
    min_gain: float = 0.02,
) -> tuple[bool, str]:
    """Applique les trois conditions de promotion du TP4."""

    # Condition 1 : challenger et champion doivent être évalués
    # sur la même fenêtre.
    if not same_window:
        return False, "different_evaluation_window"

    # Vérification des métriques nécessaires.
    if "mae_model" not in challenger_metrics:
        return False, "missing_mae_model"

    if "mae_baseline_persistence" not in challenger_metrics:
        return False, "missing_mae_baseline_persistence"

    challenger_mae = float(challenger_metrics["mae_model"])
    baseline_mae = float(challenger_metrics["mae_baseline_persistence"])

    # Condition 2 : le challenger doit battre la persistance.
    if challenger_mae >= baseline_mae:
        return False, "model_not_better_than_baseline"

    # S'il n'existe pas encore de champion, le challenger
    # devient le premier champion.
    if champion_metrics is None:
        return True, "first_champion"

    if "mae_model" not in champion_metrics:
        return False, "missing_champion_mae_model"

    champion_mae = float(champion_metrics["mae_model"])

    # Condition 3 : battre le champion d'au moins 2 %.
    target = champion_mae * (1 - min_gain)

    if challenger_mae > target:
        return False, "gain_too_small"

    return True, "eligible"


def should_promote(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None = None,
    *,
    same_window: bool = True,
    min_gain: float = 0.02,
) -> bool:
    """Vrai si le challenger respecte la règle de promotion."""
    return evaluate_promotion(
        challenger_metrics,
        champion_metrics,
        same_window=same_window,
        min_gain=min_gain,
    )[0]


def promote_model(
    model_name: str = "velov-availability",
    *,
    challenger_alias: str = "challenger",
    champion_alias: str = "champion",
    previous_champion_alias: str = "previous-champion",
    min_gain: float = 0.02,
) -> str | None:
    """Promeut le challenger s'il respecte les trois conditions."""

    client = MlflowClient()

    try:
        challenger = client.get_model_version_by_alias(model_name, challenger_alias)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise

    challenger_run = client.get_run(challenger.run_id)
    challenger_metrics = dict(challenger_run.data.metrics)
    challenger_tags = dict(challenger_run.data.tags)

    try:
        champion = client.get_model_version_by_alias(
            model_name,
            champion_alias,
        )

        champion_run = client.get_run(champion.run_id)
        champion_metrics = dict(champion_run.data.metrics)
        champion_tags = dict(champion_run.data.tags)

        challenger_start = challenger_tags.get("data_start")
        challenger_end = challenger_tags.get("data_end")
        champion_start = champion_tags.get("data_start")
        champion_end = champion_tags.get("data_end")
        same_window = bool(
            challenger_start
            and challenger_end
            and champion_start
            and champion_end
            and challenger_start == champion_start
            and challenger_end == champion_end
        )

    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        champion = None
        champion_metrics = None
        same_window = True

    eligible, reason = evaluate_promotion(
        challenger_metrics,
        champion_metrics,
        same_window=same_window,
        min_gain=min_gain,
    )

    if not eligible:
        return None

    # Conserver l'ancien champion pour permettre le rollback.
    if champion is not None:
        client.set_registered_model_alias(
            model_name,
            previous_champion_alias,
            champion.version,
        )

    client.set_registered_model_alias(
        model_name,
        champion_alias,
        challenger.version,
    )

    client.set_model_version_tag(
        model_name,
        challenger.version,
        "promotion",
        reason,
    )

    return challenger.version


def main() -> None:
    parser = argparse.ArgumentParser(description="Opérations de registre MLflow pour le modèle Vélo'v.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("promote", help="Promouvoir l'alias challenger si les règles sont satisfaites.")
    args = parser.parse_args()

    if args.command == "promote":
        version = promote_model()
        if version is None:
            raise SystemExit("Promotion refusée : challenger absent ou critères non satisfaits.")
        print(f"Version {version} promue comme champion.")


if __name__ == "__main__":
    main()
