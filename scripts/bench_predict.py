"""Où passe le temps d'une prédiction ? Chronomètre chaque étape de /v1/predict, sans réseau.

    python scripts/bench_predict.py --n 500
    python scripts/bench_predict.py --n 500 --batch 1 10 100 1000
    MODEL_FORMAT=onnx python scripts/bench_predict.py --n 500     # après python -m velov.export_onnx

Mêmes étapes que _predict() dans velov/api/main.py, dans le même ordre. Échauffement non compté,
puis médiane et p95 : une moyenne cache la queue de distribution.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from velov.api.main import load_model
from velov.api.schemas import PredictionRequest
from velov.features import FEATURES, add_features


def random_payload(rng: random.Random) -> dict:
    capacity = rng.randint(15, 40)
    instant = datetime.now(UTC) - timedelta(minutes=rng.randint(0, 30 * 24 * 60))
    return {
        "station_id": rng.randint(1, 20),
        "timestamp": instant.isoformat(),
        "capacity": capacity,
        "bikes_available": rng.randint(0, capacity),
        "temperature": round(rng.gauss(14, 7), 1),
        "is_raining": rng.random() < 0.08,
    }


def chrono(fn, n: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        out[i] = (time.perf_counter() - t0) * 1000
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "models")))
    parser.add_argument("--n", type=int, default=500, help="mesures par étape")
    parser.add_argument("--warmup", type=int, default=50, help="appels d'échauffement, non comptés")
    parser.add_argument("--batch", type=int, nargs="*", default=[], help="tailles de lot pour model.predict")
    args = parser.parse_args()
    if args.n < 20:
        parser.error("--n doit être >= 20 pour un p95 qui veut dire quelque chose")

    model, metadata = load_model(args.model_dir)
    rng = random.Random(0)  # noqa: S311 - deterministic benchmark data, not cryptographic
    payload = random_payload(rng)
    item = PredictionRequest(**payload)
    raw = pd.DataFrame([item.model_dump()])
    features = add_features(raw)[FEATURES]
    pred = model.predict(features)

    steps = {
        "validation Pydantic": lambda: PredictionRequest(**payload),
        "DataFrame + add_features": lambda: add_features(pd.DataFrame([item.model_dump()]))[FEATURES],
        "model.predict (1 ligne)": lambda: model.predict(features),
        "np.clip + réponse": lambda: round(float(np.clip(pred, 0, raw["capacity"].to_numpy())[0]), 2),
    }
    fmt = metadata.get("served_format", "joblib")
    print(f"Modèle {metadata['model_version']} ({fmt}), {args.n} mesures après {args.warmup} d'échauffement")
    print(f"{'Étape':<28}{'p50 (ms)':>10}{'p95 (ms)':>10}")
    for name, fn in steps.items():
        t = chrono(fn, args.n, args.warmup)
        print(f"{name:<28}{np.percentile(t, 50):>10.3f}{np.percentile(t, 95):>10.3f}")

    if args.batch:
        print(f"\n{'Lot':>6}{'p50 (ms)':>10}{'par ligne (ms)':>16}")
        for size in args.batch:
            if size < 1:
                parser.error("--batch : tailles >= 1")
            rows = pd.DataFrame([random_payload(rng) for _ in range(size)])
            rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
            batch = add_features(rows)[FEATURES]
            t = chrono(lambda b=batch: model.predict(b), max(args.n // 5, 20), args.warmup // 5)
            p50 = np.percentile(t, 50)
            print(f"{size:>6}{p50:>10.3f}{p50 / size:>16.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
