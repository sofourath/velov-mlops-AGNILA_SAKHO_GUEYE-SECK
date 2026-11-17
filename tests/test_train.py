import re

import duckdb

from velov.data import simulate
from velov.train import train


def test_training_metadata_contains_traceability_fields(tmp_path):
    database_path = tmp_path / "gold.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.register("observations", simulate(days=20, n_stations=2))
        connection.execute("create schema main_gold")
        connection.execute("create table main_gold.velov_gold as select * from observations")

    metadata = train(
        database_path,
        tmp_path / "models",
        test_days=3,
        gold_table="main_gold.velov_gold",
    )

    assert re.fullmatch(r"[0-9a-f]{40}", metadata["git_commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["data_sha256"])
    assert metadata["parameters"] == {
        "seed": 42,
        "test_days": 3,
        "gold_table": "main_gold.velov_gold",
    }
    assert set(metadata["metrics"]) >= {"mae_model", "mae_baseline_persistence"}
