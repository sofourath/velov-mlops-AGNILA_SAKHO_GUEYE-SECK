from types import SimpleNamespace

from velov import tracking
from velov.tracking import evaluate_promotion, should_promote


def test_rejects_model_not_better_than_baseline() -> None:
    ok, reason = evaluate_promotion(
        {
            "mae_model": 2.9,
            "mae_baseline_persistence": 2.8,
        }
    )

    assert ok is False
    assert reason == "model_not_better_than_baseline"


def test_accepts_first_champion() -> None:
    ok, reason = evaluate_promotion(
        {
            "mae_model": 2.1,
            "mae_baseline_persistence": 2.9,
        },
        None,
    )

    assert ok is True
    assert reason == "first_champion"


def test_accepts_challenger_with_at_least_two_percent_gain() -> None:
    ok, reason = evaluate_promotion(
        {
            "mae_model": 1.90,
            "mae_baseline_persistence": 2.9,
        },
        {
            "mae_model": 2.0,
        },
    )

    assert ok is True
    assert reason == "eligible"


def test_rejects_challenger_with_insufficient_gain() -> None:
    ok, reason = evaluate_promotion(
        {
            "mae_model": 1.99,
            "mae_baseline_persistence": 2.9,
        },
        {
            "mae_model": 2.0,
        },
    )

    assert ok is False
    assert reason == "gain_too_small"


def test_rejects_different_evaluation_window() -> None:
    ok, reason = evaluate_promotion(
        {
            "mae_model": 1.8,
            "mae_baseline_persistence": 2.9,
        },
        {
            "mae_model": 2.0,
        },
        same_window=False,
    )

    assert ok is False
    assert reason == "different_evaluation_window"


def test_should_promote_returns_boolean() -> None:
    result = should_promote(
        {
            "mae_model": 1.9,
            "mae_baseline_persistence": 2.9,
        },
        {
            "mae_model": 2.0,
        },
    )

    assert result is True


def test_promote_model_updates_aliases_and_keeps_previous_champion(monkeypatch) -> None:
    evaluation_window = {"data_start": "2026-10-01T00:00:00+00:00", "data_end": "2026-10-14T23:00:00+00:00"}

    class FakeClient:
        def __init__(self) -> None:
            self.alias_updates = []
            self.version_tags = []

        def get_model_version_by_alias(self, model_name: str, alias: str):
            versions = {
                "challenger": SimpleNamespace(run_id="new-run", version="2"),
                "champion": SimpleNamespace(run_id="old-run", version="1"),
            }
            return versions[alias]

        def get_run(self, run_id: str):
            if run_id == "new-run":
                data = SimpleNamespace(
                    metrics={"mae_model": 1.8, "mae_baseline_persistence": 2.9},
                    tags=evaluation_window,
                )
            else:
                data = SimpleNamespace(metrics={"mae_model": 2.0}, tags=evaluation_window)
            return SimpleNamespace(data=data)

        def set_registered_model_alias(self, model_name: str, alias: str, version: str) -> None:
            self.alias_updates.append((alias, version))

        def set_model_version_tag(self, model_name: str, version: str, key: str, value: str) -> None:
            self.version_tags.append((version, key, value))

    client = FakeClient()
    monkeypatch.setattr(tracking, "MlflowClient", lambda: client)

    assert tracking.promote_model() == "2"
    assert client.alias_updates == [("previous-champion", "1"), ("champion", "2")]
    assert client.version_tags == [("2", "promotion", "eligible")]


def test_promote_model_rejects_missing_evaluation_window(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.alias_updates = []

        def get_model_version_by_alias(self, model_name: str, alias: str):
            versions = {
                "challenger": SimpleNamespace(run_id="new-run", version="2"),
                "champion": SimpleNamespace(run_id="old-run", version="1"),
            }
            return versions[alias]

        def get_run(self, run_id: str):
            if run_id == "new-run":
                data = SimpleNamespace(
                    metrics={"mae_model": 1.8, "mae_baseline_persistence": 2.9},
                    tags={"data_start": "2026-10-01", "data_end": "2026-10-14"},
                )
            else:
                data = SimpleNamespace(metrics={"mae_model": 2.0}, tags={})
            return SimpleNamespace(data=data)

        def set_registered_model_alias(self, model_name: str, alias: str, version: str) -> None:
            self.alias_updates.append((alias, version))

    client = FakeClient()
    monkeypatch.setattr(tracking, "MlflowClient", lambda: client)

    assert tracking.promote_model() is None
    assert client.alias_updates == []
