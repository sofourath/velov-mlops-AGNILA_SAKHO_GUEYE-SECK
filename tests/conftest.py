"""Fixtures partagées : un petit modèle entraîné une seule fois par session de tests."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from velov.data import simulate
from velov.train import train


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    workdir = tmp_path_factory.mktemp("model")
    gold_db_path = workdir / "gold.duckdb"
    data = simulate(days=20, n_stations=3, seed=0)
    with duckdb.connect(str(gold_db_path)) as connection:
        connection.register("observations", data)
        connection.execute("create schema main_gold")
        connection.execute("create table main_gold.velov_gold as select * from observations")
    out = workdir / "models"
    train(
        gold_db_path,
        out,
        version="0.0.0-test",
        test_days=3,
        gold_table="main_gold.velov_gold",
    )
    return out


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from velov.api.main import app

    with TestClient(app) as c:  # le "with" déclenche le lifespan (chargement du modèle)
        yield c


@pytest.fixture
def valid_payload() -> dict:
    return {
        "station_id": 2,
        "timestamp": "2026-10-06T08:00:00+02:00",  # 8 h à Lyon, soit 6 h UTC
        "capacity": 20,
        "bikes_available": 12,
        "temperature": 14.5,
        "is_raining": False,
    }
