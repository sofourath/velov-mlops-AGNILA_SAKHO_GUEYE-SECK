"""Flow Prefect hebdomadaire : ingestion, dbt et création d'un challenger MLflow."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from prefect import flow, task

from velov.tracking import log_to_mlflow, register_model
from velov.train import DEFAULT_GOLD_DB, train


@task
def ingest_bronze() -> None:
    start = (datetime.now(UTC).date() - timedelta(days=90)).isoformat()
    subprocess.run(  # noqa: S603 - executable and arguments are fixed by this project
        [
            sys.executable,
            "-m",
            "velov.data",
            "--days",
            "90",
            "--stations",
            "20",
            "--start",
            start,
            "--out",
            "data/velov_history.csv",
            "--bronze-out",
            "data/bronze.csv",
        ],
        check=True,
    )


@task
def run_dbt_build() -> None:
    dbt = shutil.which("dbt")
    if dbt is None:
        raise FileNotFoundError("dbt introuvable dans le PATH")
    subprocess.run(  # noqa: S603 - executable is resolved from the trusted PATH
        [
            dbt,
            "build",
            "--project-dir",
            "dbt",
            "--profiles-dir",
            "dbt",
        ],
        check=True,
    )


@task
def train_and_register_challenger() -> str:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("Définir MLFLOW_TRACKING_URI vers le serveur MLflow avant de démarrer le flow.")

    version = datetime.now(UTC).strftime("%Y.%m.%d.%H%M")
    candidate_dir = Path("models/candidate")
    metadata = train(
        gold_db_path=DEFAULT_GOLD_DB,
        out_dir=candidate_dir,
        version=version,
        test_days=14,
    )
    run_id = log_to_mlflow(metadata, candidate_dir)
    model_version = register_model(run_id=run_id, alias="challenger")
    print(f"Challenger MLflow version {model_version}; promotion champion manuelle après revue.")
    return model_version


@flow(name="velov-weekly-retraining")
def velov_pipeline() -> str:
    ingest_bronze()
    run_dbt_build()
    return train_and_register_challenger()


if __name__ == "__main__":
    velov_pipeline.serve(
        name="velov-weekly-retraining",
        cron="0 3 * * 1",
    )
