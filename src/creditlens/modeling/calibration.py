"""재사용 가능한 이진분류 확률 보정기.

CLI 모듈을 ``python -m``으로 실행해도 joblib 산출물이 ``__main__``에 묶이지
않도록 보정기 클래스는 실행 진입점과 분리된 이 모듈에 둔다.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


DEFAULT_EPSILON = 1e-6
DEFAULT_RANDOM_SEED = 42


class CalibrationContractError(ValueError):
    """보정 점수·정답 또는 학습 상태가 계약과 다를 때 발생한다."""


class ScoreCalibrator(Protocol):
    def fit(self, scores: np.ndarray, y: np.ndarray) -> ScoreCalibrator: ...

    def predict(self, scores: np.ndarray) -> np.ndarray: ...


def validate_scores(scores: Any) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or ((values < 0.0) | (values > 1.0)).any()
    ):
        raise CalibrationContractError(
            "위험점수는 비어 있지 않은 [0, 1] 유한 1차원 배열이어야 합니다."
        )
    return values


def validate_score_labels(
    scores: Any,
    y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    values = validate_scores(scores)
    labels = np.asarray(y)
    if (
        labels.ndim != 1
        or labels.size != values.size
        or not np.isin(labels, (0, 1)).all()
        or np.unique(labels).size != 2
    ):
        raise CalibrationContractError(
            "보정 TARGET은 점수와 길이가 같은 두 클래스 0/1 배열이어야 합니다."
        )
    return values, labels.astype(np.int8, copy=False)


class IdentityCalibrator:
    """확률 보정이 불필요할 때 사용하는 명시적 항등 변환."""

    method = "identity"

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IdentityCalibrator:
        checked_scores, _ = validate_score_labels(scores, y)
        self.n_features_in_ = 1
        self.fit_rows_ = int(checked_scores.size)
        self.sample_weight_used_ = False
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return validate_scores(scores).copy()


class SigmoidLogitCalibrator:
    """모델 확률의 logit에 Platt식 sigmoid를 적합한다."""

    method = "sigmoid"

    def __init__(self, *, epsilon: float = DEFAULT_EPSILON) -> None:
        self.epsilon = epsilon

    def _design(self, scores: np.ndarray) -> np.ndarray:
        values = validate_scores(scores)
        clipped = np.clip(values, self.epsilon, 1.0 - self.epsilon)
        logits = np.log(clipped / (1.0 - clipped))
        return logits.reshape(-1, 1)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> SigmoidLogitCalibrator:
        checked_scores, labels = validate_score_labels(scores, y)
        self.fit_rows_ = int(checked_scores.size)
        self.sample_weight_used_ = False
        self.estimator_ = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            max_iter=1_000,
            random_state=DEFAULT_RANDOM_SEED,
        )
        self.estimator_.fit(self._design(checked_scores), labels)
        coefficient = float(self.estimator_.coef_[0, 0])
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise CalibrationContractError(
                "sigmoid 보정기가 위험점수 순서를 보존하지 않습니다."
            )
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise CalibrationContractError("sigmoid 보정기가 학습되지 않았습니다.")
        values = self.estimator_.predict_proba(self._design(scores))[:, 1]
        return validate_scores(values)


class IsotonicScoreCalibrator:
    """위험점수에 단조 isotonic 회귀를 적용한다."""

    method = "isotonic"

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IsotonicScoreCalibrator:
        checked_scores, labels = validate_score_labels(scores, y)
        self.fit_rows_ = int(checked_scores.size)
        self.sample_weight_used_ = False
        self.estimator_ = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        self.estimator_.fit(checked_scores, labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise CalibrationContractError("isotonic 보정기가 학습되지 않았습니다.")
        return validate_scores(self.estimator_.predict(validate_scores(scores)))
