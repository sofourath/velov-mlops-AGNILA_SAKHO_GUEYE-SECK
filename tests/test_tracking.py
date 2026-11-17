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
