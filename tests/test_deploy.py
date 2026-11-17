import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from velov import deploy


class FakeMlflowClient:
    def __init__(self, package_dir: Path):
        self.package_dir = package_dir

    def get_model_version_by_alias(self, model_name: str, alias: str):
        assert model_name == "velov-availability"
        assert alias == "champion"
        return SimpleNamespace(run_id="run-id", version="2")

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
        assert run_id == "run-id"
        assert path == "package"
        output = Path(dst_path) / path
        shutil.copytree(self.package_dir, output)
        return str(output)


def write_model_bundle(directory: Path, *, valid: bool = True) -> None:
    directory.mkdir(parents=True)
    model_path = directory / "model.joblib"
    joblib.dump({"model": "test"}, model_path)
    metadata = {
        "model_version": "test",
        "artifact": {
            "file": "model.joblib",
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        },
    }
    if not valid:
        metadata["artifact"]["sha256"] = "invalid"
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_deploys_valid_bundle_and_preserves_previous_model(tmp_path, monkeypatch):
    models = tmp_path / "models"
    write_model_bundle(models)
    candidate = tmp_path / "candidate"
    write_model_bundle(candidate)
    monkeypatch.setattr(deploy, "MlflowClient", lambda: FakeMlflowClient(candidate))

    version, backup = deploy.deploy_registered_model(model_dir=models)

    assert version == "2"
    assert backup is not None
    assert (models / "metadata.json").exists()
    assert (backup / "metadata.json").exists()


def test_rejects_invalid_bundle_without_replacing_current_model(tmp_path, monkeypatch):
    models = tmp_path / "models"
    write_model_bundle(models)
    candidate = tmp_path / "candidate"
    write_model_bundle(candidate, valid=False)
    monkeypatch.setattr(deploy, "MlflowClient", lambda: FakeMlflowClient(candidate))

    with pytest.raises(RuntimeError, match="Empreinte invalide"):
        deploy.deploy_registered_model(model_dir=models)

    assert (models / "metadata.json").exists()
    assert not list(tmp_path.glob("models.backup-*"))
