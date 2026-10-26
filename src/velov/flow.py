from __future__ import annotations

import subprocess

from prefect import flow, task


@task
def ingest_bronze() -> None:
    subprocess.run(
        [
            "python",
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
    subprocess.run(
        [
            "dbt",
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