from __future__ import annotations

import shutil
import subprocess
import sys

from prefect import flow, task


@task
def ingest_bronze() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "velov.data",
            "--days",
            "2",
            "--stations",
            "20",
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
    subprocess.run(  # noqa: S603 - executable is resolved from the trusted PATH  # noqa: S603 - executable is resolved from the trusted PATH
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


@flow(name="velov-pipeline")
def velov_pipeline() -> None:
    ingest_bronze()
    run_dbt_build()


if __name__ == "__main__":
    velov_pipeline.serve(
        name="velov-pipeline-hourly",
        cron="0 * * * *",
    )
