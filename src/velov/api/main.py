"""API de serving du modèle Vélo'v.

TP1, partie 3 : exposez le modèle. Mode : IA déclarée autorisée pour cette partie.

Endpoints attendus (niveaux du TP1 : Must, Should, Stretch) :
    POST /v1/predict        [Must]    une prédiction
    GET  /health            [Should]  liveness : le process répond (ne dépend pas du modèle)
    GET  /ready             [Should]  readiness : 200 si le modèle est chargé, 503 sinon
    GET  /v1/model, POST /v1/predict/batch   [Stretch]

Lancement :
    uvicorn velov.api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException

from velov.api.schemas import PredictionRequest, PredictionResponse
from velov.features import FEATURES, add_features
from velov.train import METADATA_FILENAME, sha256_of


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("velov.api")

STATE: dict = {
    "model": None,
    "metadata": None,
}


def load_model(model_dir: Path) -> tuple[object, dict]:
    """Charge le modèle après avoir vérifié son empreinte SHA-256."""
    metadata_path = model_dir / METADATA_FILENAME

    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} introuvable")

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    model_path = model_dir / metadata["artifact"]["file"]

    if sha256_of(model_path) != metadata["artifact"]["sha256"]:
        raise RuntimeError(
            f"Empreinte invalide pour {model_path}"
        )

    model = joblib.load(model_path)

    return model, metadata


def get_psycopg_database_url() -> str | None:
    """Retourne une URL compatible avec psycopg.

    Le compose fournit DATABASE_URL au format SQLAlchemy :
    postgresql+psycopg://...

    psycopg.connect() attend :
    postgresql://...
    """
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        return None

    return database_url.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )


def save_prediction(
    request: PredictionRequest,
    response: PredictionResponse,
) -> None:
    """Enregistre une prédiction dans PostgreSQL.

    Si PostgreSQL est indisponible, l'API continue de fonctionner :
    l'erreur est seulement enregistrée dans les logs.
    """
    database_url = get_psycopg_database_url()

    if not database_url:
        logger.warning(
            "DATABASE_URL absente : prédiction non enregistrée"
        )
        return

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id BIGSERIAL PRIMARY KEY,
                        station_id INTEGER NOT NULL,
                        observed_timestamp TIMESTAMPTZ NOT NULL,
                        target_timestamp TIMESTAMPTZ NOT NULL,
                        capacity INTEGER NOT NULL,
                        bikes_available INTEGER NOT NULL,
                        temperature DOUBLE PRECISION NOT NULL,
                        is_raining BOOLEAN NOT NULL,
                        predicted_bikes DOUBLE PRECISION NOT NULL,
                        model_version TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )

                cur.execute(
                    """
                    INSERT INTO predictions (
                        station_id,
                        observed_timestamp,
                        target_timestamp,
                        capacity,
                        bikes_available,
                        temperature,
                        is_raining,
                        predicted_bikes,
                        model_version
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        request.station_id,
                        request.timestamp,
                        response.target_timestamp,
                        request.capacity,
                        request.bikes_available,
                        request.temperature,
                        request.is_raining,
                        response.predicted_bikes,
                        response.model_version,
                    ),
                )

            conn.commit()

        logger.info(
            "Prédiction enregistrée en base pour la station %s",
            request.station_id,
        )

    except Exception:
        logger.exception(
            "Impossible d'enregistrer la prédiction en base"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Exécuté au démarrage et à l'arrêt de l'application."""
    model_dir = Path(
        os.getenv("MODEL_DIR", "models")
    )

    try:
        STATE["model"], STATE["metadata"] = load_model(
            model_dir
        )

        logger.info(
            "Modèle %s chargé",
            STATE["metadata"]["model_version"],
        )

    except Exception:
        logger.exception(
            "Échec du chargement du modèle depuis %s",
            model_dir,
        )

    yield

    STATE.update(
        model=None,
        metadata=None,
    )


app = FastAPI(
    title="Vélo'v availability API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness : le processus FastAPI répond."""
    return {
        "status": "ok",
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness : le modèle doit être chargé."""
    if (
        STATE["model"] is None
        or STATE["metadata"] is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé",
        )

    return {
        "status": "ready",
        "model_version": STATE["metadata"]["model_version"],
    }


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """Calcule une prédiction à H+1 et la journalise en base."""
    if (
        STATE["model"] is None
        or STATE["metadata"] is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé",
        )

    observation = pd.DataFrame(
        [request.model_dump()]
    )

    features = add_features(
        observation
    )[FEATURES]

    prediction = float(
        STATE["model"].predict(features)[0]
    )

    predicted_bikes = float(
        np.clip(
            prediction,
            0,
            request.capacity,
        )
    )

    response = PredictionResponse(
        station_id=request.station_id,
        target_timestamp=(
            request.timestamp
            + timedelta(hours=1)
        ),
        predicted_bikes=predicted_bikes,
        model_version=STATE["metadata"]["model_version"],
    )

    save_prediction(
        request=request,
        response=response,
    )

    return response