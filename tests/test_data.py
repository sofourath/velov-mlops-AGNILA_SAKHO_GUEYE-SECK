import pandas as pd

from velov.data import BRONZE_KEY_COLUMNS, ingest_bronze, simulate


def test_bronze_ingestion_is_idempotent(tmp_path):
    batch = simulate(days=2, n_stations=2)
    bronze_path = tmp_path / "bronze.csv"

    first = ingest_bronze(batch, bronze_path)
    second = ingest_bronze(batch, bronze_path)

    assert len(first) == len(batch)
    assert len(second) == len(batch)
    assert not second.duplicated(subset=BRONZE_KEY_COLUMNS).any()
    pd.testing.assert_frame_equal(
        second.sort_values(list(BRONZE_KEY_COLUMNS)).reset_index(drop=True),
        first.sort_values(list(BRONZE_KEY_COLUMNS)).reset_index(drop=True),
    )
