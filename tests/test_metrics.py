"""Stage 4 공통 이진 분류 평가 지표 테스트."""

from __future__ import annotations

import json

import numpy as np
import pytest

from creditlens.evaluation.metrics import MetricInputError, evaluate_binary_metrics


def test_reference_fixture_matches_known_metrics() -> None:
    result = evaluate_binary_metrics(
        [0, 0, 1, 1],
        [0.10, 0.40, 0.35, 0.80],
        threshold=0.5,
        top_fraction=0.25,
    )

    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["pr_auc"] == pytest.approx(5 / 6)
    assert result["ks"] == pytest.approx(0.5)
    assert result["ks_threshold"] == pytest.approx(0.8)
    assert result["gini"] == pytest.approx(0.5)
    assert result["brier_score"] == pytest.approx(0.158125)
    assert result["warnings"] == []

    threshold = result["threshold_metrics"]
    assert threshold == {
        "threshold": 0.5,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 1,
        "true_positive": 1,
        "predicted_positive_count": 1,
        "recall": 0.5,
        "precision": 1.0,
        "f1": pytest.approx(2 / 3),
    }

    top_k = result["top_k_metrics"]
    assert top_k["k"] == 1
    assert top_k["precision"] == pytest.approx(1.0)
    assert top_k["recall"] == pytest.approx(0.5)
    assert top_k["lift"] == pytest.approx(2.0)


def test_threshold_is_inclusive() -> None:
    result = evaluate_binary_metrics(
        [1, 0],
        [0.5, 0.49],
        threshold=0.5,
        top_fraction=0.5,
    )

    threshold = result["threshold_metrics"]
    assert threshold["true_positive"] == 1
    assert threshold["false_positive"] == 0


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([1, 0, 1, 0], [0.9, 0.8, 0.8, 0.1]),
        ([0, 1, 0, 1], [0.8, 0.8, 0.1, 0.9]),
    ],
)
def test_fractional_boundary_tie_is_order_invariant(
    labels: list[int], scores: list[float]
) -> None:
    result = evaluate_binary_metrics(
        labels,
        scores,
        top_fraction=None,
        top_k=2,
    )["top_k_metrics"]

    assert result["cutoff_score"] == pytest.approx(0.8)
    assert result["boundary_tie_count"] == 2
    assert result["boundary_selected_weight"] == pytest.approx(0.5)
    assert result["true_positive_weight"] == pytest.approx(1.5)
    assert result["precision"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(0.75)
    assert result["lift"] == pytest.approx(1.5)


def test_all_equal_scores_have_neutral_ranking_metrics() -> None:
    result = evaluate_binary_metrics(
        [1, 0, 0, 1],
        [0.5, 0.5, 0.5, 0.5],
        top_fraction=0.5,
    )

    assert result["roc_auc"] == pytest.approx(0.5)
    assert result["pr_auc"] == pytest.approx(0.5)
    assert result["ks"] == pytest.approx(0.0)
    assert result["ks_threshold"] == pytest.approx(0.5)
    assert result["gini"] == pytest.approx(0.0)
    assert result["top_k_metrics"]["true_positive_weight"] == pytest.approx(1.0)
    assert result["top_k_metrics"]["precision"] == pytest.approx(0.5)
    assert result["top_k_metrics"]["recall"] == pytest.approx(0.5)
    assert result["top_k_metrics"]["lift"] == pytest.approx(1.0)


def test_directional_ks_does_not_reward_reversed_scores() -> None:
    result = evaluate_binary_metrics(
        [0, 0, 1, 1],
        [0.9, 0.8, 0.2, 0.1],
        top_fraction=0.5,
    )

    assert result["roc_auc"] == pytest.approx(0.0)
    assert result["ks"] == pytest.approx(0.0)
    assert result["gini"] == pytest.approx(-1.0)


def test_top_fraction_uses_ceiling_and_full_selection_is_neutral() -> None:
    labels = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    scores = np.linspace(1.0, 0.0, num=11)

    top_ten = evaluate_binary_metrics(
        labels,
        scores,
        top_fraction=0.1,
    )["top_k_metrics"]
    assert top_ten["k"] == 2

    all_selected = evaluate_binary_metrics(
        labels,
        scores,
        top_fraction=None,
        top_k=11,
    )["top_k_metrics"]
    assert all_selected["recall"] == pytest.approx(1.0)
    assert all_selected["precision"] == pytest.approx(2 / 11)
    assert all_selected["lift"] == pytest.approx(1.0)


def test_top_k_can_be_set_without_explicitly_disabling_default_fraction() -> None:
    top_k = evaluate_binary_metrics(
        [1, 0, 0],
        [0.9, 0.5, 0.1],
        top_k=1,
    )["top_k_metrics"]

    assert top_k["requested_fraction"] is None
    assert top_k["k"] == 1
    assert top_k["precision"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("labels", "scores", "expected_brier"),
    [
        ([0, 0, 0], [0.1, 0.2, 0.3], (0.01 + 0.04 + 0.09) / 3),
        ([1, 1, 1], [0.7, 0.8, 0.9], (0.09 + 0.04 + 0.01) / 3),
    ],
)
def test_single_class_returns_none_for_discrimination_metrics(
    labels: list[int], scores: list[float], expected_brier: float
) -> None:
    result = evaluate_binary_metrics(labels, scores, top_fraction=0.5)

    for key in ("roc_auc", "pr_auc", "ks", "ks_threshold", "gini"):
        assert result[key] is None
    assert result["brier_score"] == pytest.approx(expected_brier)
    assert any("single_class" in warning for warning in result["warnings"])


def test_no_positive_class_marks_recall_and_lift_undefined() -> None:
    result = evaluate_binary_metrics(
        [0, 0, 0],
        [0.8, 0.4, 0.1],
        threshold=0.5,
        top_fraction=1 / 3,
    )

    assert result["threshold_metrics"]["precision"] == pytest.approx(0.0)
    assert result["threshold_metrics"]["recall"] is None
    assert result["threshold_metrics"]["f1"] == pytest.approx(0.0)
    assert result["top_k_metrics"]["precision"] == pytest.approx(0.0)
    assert result["top_k_metrics"]["recall"] is None
    assert result["top_k_metrics"]["lift"] is None
    assert any("no_positive_class" in warning for warning in result["warnings"])


def test_no_predicted_positives_has_undefined_precision_and_zero_f1() -> None:
    result = evaluate_binary_metrics(
        [0, 1],
        [0.2, 0.3],
        threshold=1.0,
        top_fraction=0.5,
    )

    threshold = result["threshold_metrics"]
    assert threshold["predicted_positive_count"] == 0
    assert threshold["recall"] == pytest.approx(0.0)
    assert threshold["precision"] is None
    assert threshold["f1"] == pytest.approx(0.0)
    assert any("no_predicted_positives" in warning for warning in result["warnings"])


def test_all_negative_without_predictions_has_undefined_f1() -> None:
    result = evaluate_binary_metrics(
        [0, 0],
        [0.1, 0.2],
        threshold=1.0,
        top_fraction=0.5,
    )

    threshold = result["threshold_metrics"]
    assert threshold["recall"] is None
    assert threshold["precision"] is None
    assert threshold["f1"] is None
    assert any("f1_undefined" in warning for warning in result["warnings"])


def test_result_is_json_serializable() -> None:
    result = evaluate_binary_metrics(
        np.array([0, 1]),
        np.array([0.2, 0.8]),
        top_fraction=0.5,
    )

    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([], [], "비어"),
        ([0, 1], [0.1], "길이"),
        ([[0, 1]], [[0.1, 0.9]], "1차원"),
        ([0, 2], [0.1, 0.9], "0과 1"),
        ([0, np.nan], [0.1, 0.9], "결측값"),
        ([0, 1], [0.1, np.nan], "결측값"),
        ([0, 1], [0.1, np.inf], "무한값"),
        ([0, 1], [-0.1, 0.9], "[0, 1]"),
        ([0, 1], [0.1, 1.1], "[0, 1]"),
        (["normal", "risk"], [0.1, 0.9], "수치형"),
    ],
)
def test_invalid_arrays_raise_clear_error(
    labels: object,
    scores: object,
    message: str,
) -> None:
    with pytest.raises(MetricInputError, match=message):
        evaluate_binary_metrics(labels, scores)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, True, "0.5"])
def test_invalid_threshold_raises(threshold: object) -> None:
    with pytest.raises(MetricInputError, match="threshold"):
        evaluate_binary_metrics([0, 1], [0.1, 0.9], threshold=threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_fraction", [0.0, -0.1, 1.1, np.inf, True, "0.1"])
def test_invalid_top_fraction_raises(top_fraction: object) -> None:
    with pytest.raises(MetricInputError, match="top_fraction"):
        evaluate_binary_metrics(
            [0, 1],
            [0.1, 0.9],
            top_fraction=top_fraction,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("top_k", [0, 3, 1.5, True])
def test_invalid_top_k_raises(top_k: object) -> None:
    with pytest.raises(MetricInputError, match="top_k"):
        evaluate_binary_metrics(
            [0, 1],
            [0.1, 0.9],
            top_fraction=None,
            top_k=top_k,  # type: ignore[arg-type]
        )


def test_top_fraction_and_top_k_cannot_both_be_set() -> None:
    with pytest.raises(MetricInputError, match="동시에"):
        evaluate_binary_metrics(
            [0, 1],
            [0.1, 0.9],
            top_fraction=0.5,
            top_k=1,
        )
