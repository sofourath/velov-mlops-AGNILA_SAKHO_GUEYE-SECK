from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from velov.features import FEATURES, add_features
from velov.train import METADATA_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 100, 1000])
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    metadata = json.loads((args.model_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    model = joblib.load(args.model_dir / metadata["artifact"]["file"])
    base = pd.DataFrame([{
        "station_id": 3,
        "timestamp": pd.Timestamp("2026-11-20T10:30:00Z"),
        "capacity": 20,
        "bikes_available": 12,
        "temperature": 14.5,
        "is_raining": False,
    }])

    print("| Batch | Total ms | Per prediction ms |")
    print("|---:|---:|---:|")
    for batch in args.batch:
        observations = pd.concat([base] * batch, ignore_index=True)
        feature_started = time.perf_counter()
        features = add_features(observations)[FEATURES]
        feature_ms = (time.perf_counter() - feature_started) * 1000
        predict_started = time.perf_counter()
        model.predict(features)
        predict_ms = (time.perf_counter() - predict_started) * 1000
        started = time.perf_counter()
        for _ in range(args.n):
            features = add_features(observations)[FEATURES]
            model.predict(features)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"| {batch} | {elapsed_ms:.1f} | {elapsed_ms / (args.n * batch):.4f} "
            f"(features={feature_ms:.2f} ms, prediction={predict_ms:.2f} ms) |"
        )


if __name__ == "__main__":
    main()
