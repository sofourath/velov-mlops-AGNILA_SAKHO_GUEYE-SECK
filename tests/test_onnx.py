"""Le modèle ONNX donne les mêmes réponses que le joblib, et n'est servi que s'il est intègre (S5, EX-17)."""

from __future__ import annotations

import json
import shutil

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from velov.api.main import load_model  # noqa: E402
from velov.data import simulate  # noqa: E402
from velov.export_onnx import ONNX_FILENAME, check, export  # noqa: E402
from velov.train import METADATA_FILENAME, sha256_of, train  # noqa: E402


@pytest.fixture(scope="module")
def onnx_dir(tmp_path_factory):
    # Modèle dédié, entraîné sur 60 jours. Avec moins d'historique (le modèle de conftest), certaines
    # températures tombent à moins de 1e-6 d'un seuil : en float32, l'arbre part de l'autre côté et
    # la prédiction change de près d'un vélo sur quelques lignes. C'est ce que check() sert à attraper.
    work = tmp_path_factory.mktemp("onnx")
    simulate(days=60, n_stations=5, seed=0).to_csv(work / "history.csv", index=False)
    out = work / "models"
    train(work / "history.csv", out, version="0.0.0-onnx", test_days=7)
    pipeline, metadata = load_model(out)
    export(pipeline, out / ONNX_FILENAME)
    metadata["onnx"] = {"file": ONNX_FILENAME, "sha256": sha256_of(out / ONNX_FILENAME)}
    (out / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    return out


def test_onnx_matches_sklearn_on_every_row(onnx_dir, tmp_path):
    data = tmp_path / "history.csv"
    simulate(days=10, n_stations=7, seed=1).to_csv(data, index=False)  # stations 6 et 7 inconnues
    pipeline, _ = load_model(onnx_dir)
    max_diff, n = check(pipeline, onnx_dir / ONNX_FILENAME, data)
    assert n == 10 * 24 * 7
    assert max_diff < 1e-3


def _client(monkeypatch, model_dir, fmt):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MODEL_FORMAT", fmt)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from velov.api.main import app

    return TestClient(app)


def test_api_same_answers_with_both_engines(onnx_dir, monkeypatch, valid_payload):
    payloads = [valid_payload] + [
        {**valid_payload, "station_id": s, "bikes_available": b, "is_raining": r}
        for s, b, r in [(1, 0, False), (3, 20, True), (99, 7, False)]
    ]
    answers = {}
    for fmt in ("joblib", "onnx"):
        with _client(monkeypatch, onnx_dir, fmt) as c:
            assert c.get("/v1/model").json()["served_format"] == fmt
            answers[fmt] = [c.post("/v1/predict", json=p).json()["predicted_bikes"] for p in payloads]
    for a, b in zip(answers["joblib"], answers["onnx"], strict=True):
        assert abs(a - b) <= 0.01  # réponses arrondies au centième


def test_tampered_onnx_is_not_served(onnx_dir, monkeypatch, tmp_path):
    bad = tmp_path / "models"
    shutil.copytree(onnx_dir, bad)
    with (bad / ONNX_FILENAME).open("ab") as f:
        f.write(b"\x00")
    with _client(monkeypatch, bad, "onnx") as c:
        assert c.get("/ready").status_code == 503
