from velov.tracking import evaluate_promotion, should_promote


def test_should_promote_accepts_better_model() -> None:
    assert should_promote({"mae_model": 2.1, "mae_baseline_persistence": 2.9}) is True


def test_should_promote_rejects_model_not_better_than_baseline() -> None:
    assert should_promote({"mae_model": 2.9, "mae_baseline_persistence": 2.8}) is False


def test_evaluate_promotion_reports_reason() -> None:
    ok, reason = evaluate_promotion({"mae_model": 2.1, "mae_baseline_persistence": 2.9})
    assert ok is True
    assert reason == "eligible"
