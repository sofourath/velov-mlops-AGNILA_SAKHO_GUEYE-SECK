"""Export du modèle Vélo'v en ONNX, avec vérification des prédictions (S5, EX-17).

    python -m velov.export_onnx                    # lit models/, écrit models/model.onnx
    MODEL_FORMAT=onnx uvicorn velov.api.main:app   # l'API sert alors le modèle ONNX

On ne sert un modèle converti qu'après avoir comparé ses prédictions à celles de l'original sur
toutes les lignes d'un jeu de données : un convertisseur peut changer un résultat sans erreur.
L'export est refusé au-delà de --tolerance (défaut 0,001 vélo).

Dépendances d'export (pas dans l'image) : requirements-optim.txt. Le serving n'a besoin que
d'onnxruntime. protobuf est épinglé : skl2onnx 1.20 casse avec protobuf 7 (slide S5 · 32).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from velov.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, add_features

ONNX_FILENAME = "model.onnx"
OPSET = 17


def to_onnx_inputs(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Une entrée ONNX par colonne, dans le type déclaré à l'export (int64 ou float32).

    Une seule conversion pandas -> numpy pour toutes les colonnes numériques, puis des vues :
    sélectionner les colonnes une par une dans le DataFrame coûte plus cher que le modèle.
    """
    inputs = {c: features[c].to_numpy().astype(np.int64).reshape(-1, 1) for c in CATEGORICAL_FEATURES}
    numeric = features[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    inputs.update({c: numeric[:, i : i + 1] for i, c in enumerate(NUMERIC_FEATURES)})
    return inputs


class OnnxPredictor:
    """Même interface que le pipeline scikit-learn pour l'API : predict(DataFrame) -> ndarray."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        # Un thread par session : la parallélisation se fait avec les workers uvicorn (S5 · 15).
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.output = self.session.get_outputs()[0].name

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.session.run([self.output], to_onnx_inputs(features))[0].ravel()


def export(pipeline, path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, Int64TensorType

    initial_types = [(c, Int64TensorType([None, 1])) for c in CATEGORICAL_FEATURES]
    initial_types += [(c, FloatTensorType([None, 1])) for c in NUMERIC_FEATURES]
    onx = convert_sklearn(pipeline, initial_types=initial_types, target_opset=OPSET)
    path.write_bytes(onx.SerializeToString())


def check(pipeline, onnx_path: Path, data_path: Path) -> tuple[float, int]:
    """Écart maximal entre scikit-learn et ONNX Runtime, sur toutes les lignes du jeu."""
    raw = pd.read_csv(data_path, parse_dates=["timestamp"])
    if raw.empty:
        raise ValueError(f"{data_path} est vide")
    features = add_features(raw)[FEATURES]
    expected = pipeline.predict(features)
    got = OnnxPredictor(onnx_path).predict(features)
    return float(np.max(np.abs(expected - got))), len(features)


def main() -> int:
    from velov.api.main import load_model
    from velov.train import METADATA_FILENAME

    parser = argparse.ArgumentParser(description="Exporte le modèle en ONNX et vérifie ses prédictions.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--data", type=Path, default=Path("data/velov_history.csv"))
    parser.add_argument("--tolerance", type=float, default=1e-3, help="écart maximal accepté, en vélos")
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"{args.data} introuvable : lancer python -m velov.data")

    pipeline, metadata = load_model(args.model_dir)  # empreinte vérifiée avant chargement
    onnx_path = args.model_dir / ONNX_FILENAME
    export(pipeline, onnx_path)
    max_diff, n = check(pipeline, onnx_path, args.data)
    if max_diff > args.tolerance:
        onnx_path.unlink()
        print(f"REFUSÉ : écart maximal {max_diff:.2e} vélo sur {n} lignes (tolérance {args.tolerance})")
        return 1

    import onnx
    import onnxruntime
    import skl2onnx

    metadata["onnx"] = {
        "file": ONNX_FILENAME,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "opset": OPSET,
        "max_abs_diff": max_diff,
        "checked_rows": n,
        "versions": {
            "skl2onnx": skl2onnx.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }
    (args.model_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    size_joblib = (args.model_dir / metadata["artifact"]["file"]).stat().st_size
    print(f"Écart maximal : {max_diff:.2e} vélo sur {n} lignes (tolérance {args.tolerance})")
    print(f"{onnx_path} : {onnx_path.stat().st_size / 1024:.0f} ko (joblib : {size_joblib / 1024:.0f} ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
