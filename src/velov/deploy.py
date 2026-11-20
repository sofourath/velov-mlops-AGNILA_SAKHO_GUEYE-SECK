"""Déploie localement un alias du registre MLflow dans le dossier servi par l'API."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mlflow import MlflowClient

from velov.api.main import load_model

DEFAULT_MODEL_NAME = "velov-availability"


def deploy_registered_model(
    *,
    alias: str = "champion",
    model_name: str = DEFAULT_MODEL_NAME,
    model_dir: Path = Path("models"),
) -> tuple[str, Path | None]:
    """Télécharge et valide un alias MLflow avant de remplacer le bundle servi."""
    model_dir = Path(model_dir).resolve()
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, alias)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{model_dir.name}.stage-", dir=model_dir.parent))
    backup_dir: Path | None = None

    try:
        artifact_dir = Path(
            client.download_artifacts(
                version.run_id,
                "package",
                dst_path=str(staging_root),
            )
        )
        load_model(artifact_dir)

        if model_dir.exists():
            if not model_dir.is_dir():
                raise NotADirectoryError(f"Le dossier modèle existe mais n'est pas un dossier : {model_dir}")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup_dir = model_dir.with_name(f"{model_dir.name}.backup-{stamp}-{version.version}")
            suffix = 1
            while backup_dir.exists():
                backup_dir = model_dir.with_name(f"{model_dir.name}.backup-{stamp}-{version.version}-{suffix}")
                suffix += 1
            model_dir.replace(backup_dir)

        try:
            artifact_dir.replace(model_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists():
                backup_dir.replace(model_dir)
            raise

        return version.version, backup_dir
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Déploie un alias du registre MLflow dans MODEL_DIR.")
    parser.add_argument("--alias", default="champion", help="Alias MLflow à servir (champion ou previous-champion)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    version, backup = deploy_registered_model(
        alias=args.alias,
        model_name=args.model_name,
        model_dir=args.model_dir,
    )
    print(f"Modèle MLflow {args.model_name} version {version} copié dans {args.model_dir}.")
    if backup is not None:
        print(f"Bundle précédent conservé dans {backup}.")
    print("Recréez le conteneur API puis vérifiez /ready avant de considérer le déploiement terminé.")


if __name__ == "__main__":
    main()
