"""이진 신용위험 모델을 동일한 규칙으로 평가한다.

PR-AUC는 Average Precision으로 정의한다. Top-K 경계에 같은 점수가 여러 개면
경계 동점 그룹에 분수 가중치를 적용해 정확히 K명의 평가 용량을 유지한다.
반환값은 JSON 직렬화가 가능한 Python 기본형만 포함한다.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SINGLE_CLASS_WARNING = (
    "single_class_discrimination_undefined: "
    "ROC-AUC, PR-AUC, KS and Gini require both classes."
)
NO_POSITIVE_WARNING = (
    "no_positive_class: threshold recall, Top-K recall and Top-K lift are undefined."
)
NO_PREDICTED_POSITIVE_WARNING = (
    "no_predicted_positives: threshold precision is undefined."
)
UNDEFINED_F1_WARNING = (
    "f1_undefined: there are no actual or predicted positive samples."
)


class MetricInputError(ValueError):
    """평가 입력이 이진 분류 확률 계약을 만족하지 않을 때 발생한다."""


def _as_binary_arrays(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels_raw = np.asarray(y_true)
    scores_raw = np.asarray(y_score)

    if labels_raw.ndim != 1 or scores_raw.ndim != 1:
        raise MetricInputError("y_true와 y_score는 1차원이어야 합니다.")
    if labels_raw.size == 0:
        raise MetricInputError("평가 입력은 비어 있을 수 없습니다.")
    if labels_raw.size != scores_raw.size:
        raise MetricInputError("y_true와 y_score의 길이가 다릅니다.")

    try:
        labels_numeric = labels_raw.astype(np.float64)
        scores = scores_raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise MetricInputError("라벨과 예측확률은 수치형이어야 합니다.") from error

    if not np.isfinite(labels_numeric).all():
        raise MetricInputError("y_true에는 결측값이나 무한값을 사용할 수 없습니다.")
    if not np.isin(labels_numeric, (0.0, 1.0)).all():
        raise MetricInputError("y_true는 0과 1만 포함해야 합니다.")
    if not np.isfinite(scores).all():
        raise MetricInputError("y_score에는 결측값이나 무한값을 사용할 수 없습니다.")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise MetricInputError("y_score는 TARGET=1의 [0, 1] 예측확률이어야 합니다.")

    return labels_numeric.astype(np.int8), scores


def _probability_parameter(value: Real, *, name: str, lower_open: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MetricInputError(f"{name}은 수치여야 합니다.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MetricInputError(f"{name}은 유한한 값이어야 합니다.")
    lower_invalid = numeric <= 0.0 if lower_open else numeric < 0.0
    if lower_invalid or numeric > 1.0:
        interval = "(0, 1]" if lower_open else "[0, 1]"
        raise MetricInputError(f"{name}은 {interval} 범위여야 합니다.")
    return numeric


def _resolve_k(
    sample_count: int,
    *,
    top_fraction: Real | None,
    top_k: int | None,
) -> tuple[int, float | None]:
    if top_k is not None:
        if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, Integral):
            raise MetricInputError("top_k는 정수여야 합니다.")
        resolved = int(top_k)
        if resolved < 1 or resolved > sample_count:
            raise MetricInputError("top_k는 1 이상 표본 수 이하여야 합니다.")
        if top_fraction is not None:
            raise MetricInputError("top_fraction과 top_k는 동시에 지정할 수 없습니다.")
        return resolved, None

    if top_fraction is None:
        top_fraction = 0.1
    fraction = _probability_parameter(
        top_fraction,
        name="top_fraction",
        lower_open=True,
    )
    return max(1, math.ceil(sample_count * fraction)), fraction


def _append_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _directional_ks(
    labels: np.ndarray,
    scores: np.ndarray,
    positive_count: int,
    negative_count: int,
) -> tuple[float, float]:
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    cumulative_positive = np.cumsum(ordered_labels, dtype=np.int64)
    cumulative_negative = np.cumsum(1 - ordered_labels, dtype=np.int64)

    group_ends = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    thresholds = ordered_scores[group_ends]
    tpr = cumulative_positive[group_ends] / positive_count
    fpr = cumulative_negative[group_ends] / negative_count
    differences = tpr - fpr

    # 점수 내림차순이므로 첫 번째 최댓값이 가장 높은 임계값이다.
    best = int(np.argmax(differences))
    return float(differences[best]), float(thresholds[best])


def _threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    warnings: list[str],
) -> dict[str, int | float | None]:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives

    true_positive = int(np.count_nonzero(predictions & positives))
    false_positive = int(np.count_nonzero(predictions & negatives))
    false_negative = int(np.count_nonzero(~predictions & positives))
    true_negative = int(np.count_nonzero(~predictions & negatives))
    actual_positive = true_positive + false_negative
    predicted_positive = true_positive + false_positive

    if actual_positive:
        recall: float | None = true_positive / actual_positive
    else:
        recall = None
        _append_warning(warnings, NO_POSITIVE_WARNING)

    if predicted_positive:
        precision: float | None = true_positive / predicted_positive
    else:
        precision = None
        _append_warning(warnings, NO_PREDICTED_POSITIVE_WARNING)

    f1_denominator = 2 * true_positive + false_positive + false_negative
    if f1_denominator:
        f1: float | None = 2 * true_positive / f1_denominator
    else:
        f1 = None
        _append_warning(warnings, UNDEFINED_F1_WARNING)

    return {
        "threshold": threshold,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "predicted_positive_count": predicted_positive,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def _top_k_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
    requested_fraction: float | None,
    positive_count: int,
    warnings: list[str],
) -> dict[str, int | float | None]:
    cutoff_score = float(np.partition(scores, scores.size - k)[scores.size - k])
    above_cutoff = scores > cutoff_score
    at_cutoff = scores == cutoff_score
    above_count = int(np.count_nonzero(above_cutoff))
    boundary_tie_count = int(np.count_nonzero(at_cutoff))
    remaining_capacity = k - above_count
    boundary_weight = remaining_capacity / boundary_tie_count

    true_positive_weight = float(
        labels[above_cutoff].sum(dtype=np.int64)
        + boundary_weight * labels[at_cutoff].sum(dtype=np.int64)
    )
    precision = true_positive_weight / k
    if positive_count:
        recall: float | None = true_positive_weight / positive_count
        prevalence = positive_count / labels.size
        lift: float | None = precision / prevalence
    else:
        recall = None
        lift = None
        _append_warning(warnings, NO_POSITIVE_WARNING)

    return {
        "requested_fraction": requested_fraction,
        "k": k,
        "actual_fraction": k / labels.size,
        "cutoff_score": cutoff_score,
        "boundary_tie_count": boundary_tie_count,
        "boundary_selected_weight": boundary_weight,
        "true_positive_weight": true_positive_weight,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def evaluate_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: Real = 0.5,
    top_fraction: Real | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """이진 분류 확률의 공통 지표와 Top-K 결과를 반환한다.

    ``top_fraction``과 ``top_k``를 모두 생략하면 Top 10%를 평가한다. 둘을
    동시에 명시할 수는 없다. 단일 클래스에서 정의되지 않는 판별 지표는
    ``None``으로 반환하며 이유를 ``warnings``에 기록한다.
    """

    labels, scores = _as_binary_arrays(y_true, y_score)
    threshold_value = _probability_parameter(
        threshold,
        name="threshold",
        lower_open=False,
    )
    resolved_k, requested_fraction = _resolve_k(
        labels.size,
        top_fraction=top_fraction,
        top_k=top_k,
    )

    sample_count = int(labels.size)
    positive_count = int(labels.sum(dtype=np.int64))
    negative_count = sample_count - positive_count
    prevalence = positive_count / sample_count
    warnings: list[str] = []

    if positive_count and negative_count:
        roc_auc: float | None = float(roc_auc_score(labels, scores))
        pr_auc: float | None = float(average_precision_score(labels, scores))
        ks, ks_threshold = _directional_ks(
            labels,
            scores,
            positive_count,
            negative_count,
        )
        gini: float | None = 2.0 * roc_auc - 1.0
    else:
        roc_auc = None
        pr_auc = None
        ks = None
        ks_threshold = None
        gini = None
        _append_warning(warnings, SINGLE_CLASS_WARNING)

    threshold_result = _threshold_metrics(
        labels,
        scores,
        threshold=threshold_value,
        warnings=warnings,
    )
    top_k_result = _top_k_metrics(
        labels,
        scores,
        k=resolved_k,
        requested_fraction=requested_fraction,
        positive_count=positive_count,
        warnings=warnings,
    )

    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ks": ks,
        "ks_threshold": ks_threshold,
        "gini": gini,
        "brier_score": float(np.mean(np.square(scores - labels))),
        "threshold_metrics": threshold_result,
        "top_k_metrics": top_k_result,
        "warnings": warnings,
    }
