"""화면·API·CLI가 공유하는 추론 처리. fit이나 데이터셋 로드를 하지 않는다."""

from __future__ import annotations

from typing import Any

import numpy as np

from creditlens.inference.contract import InputContract, SCHEMA_VERSION, FEATURE_VERSION
from creditlens.modeling.calibration import validate_scores
from creditlens.modeling.frozen_model import (
    apply_risk_policy, load_frozen_model, verify_bundle,
)


class PredictionService:
    """시작할 때 한 번 로드하고 각 요청에서 저장된 변환과 predict만 호출한다."""

    @classmethod
    def from_frozen(cls) -> PredictionService:
        # 신뢰하는 로컬 기본 bundle만 사용한다. 요청에서 모델 경로를 받지 않는다.
        manifest = verify_bundle()
        pipeline, calibrator, policy = load_frozen_model()
        return cls(manifest, pipeline, calibrator, policy)

    def __init__(self, manifest: dict[str, Any], pipeline: Any,
                 calibrator: Any, policy: dict[str, Any]):
        self.contract = InputContract(manifest)
        self._pipeline, self._calibrator, self._policy = pipeline, calibrator, policy
        preprocessor = pipeline.named_steps["preprocessor"]
        roles = preprocessor.named_steps["contract"]
        if (tuple(roles.feature_columns) != self.contract.features
                or tuple(roles.numeric_columns) != self.contract.numeric
                or tuple(roles.categorical_columns) != self.contract.categorical
                or list(pipeline.classes_) != [0, 1]
                or policy["release"] != self.contract.release
                or calibrator.method != policy["calibration_method"]):
            raise ValueError("모델과 서비스 입력 계약이 다릅니다.")
        rare = preprocessor.named_steps["columns"].named_transformers_["categorical"].named_steps["rare"]
        self._known_categories = dict(zip(self.contract.categorical, rare.seen_categories_, strict=True))

    def predict(self, payload: Any) -> dict[str, Any]:
        """모든 행을 검증한 뒤 예측한다. 오류가 있으면 부분 결과도 반환하지 않는다."""
        ids, frame = self.contract.validate(payload)
        raw = np.asarray(self._pipeline.predict_proba(frame))
        if raw.shape != (len(ids), 2):
            raise ValueError("예측 확률 배열 크기가 계약과 다릅니다.")
        scores = validate_scores(self._calibrator.predict(validate_scores(raw[:, 1])))
        if len(scores) != len(ids):
            raise ValueError("보정 확률 수가 입력 건수와 다릅니다.")
        decisions = apply_risk_policy(scores, self._policy)
        predictions = []
        for index, record in enumerate(payload["records"]):
            features = record["features"]
            missing_count = sum(value is None for value in features.values())
            unknown = [name for name, seen in self._known_categories.items()
                       if features[name] is not None and features[name] not in seen]
            warnings = []
            if missing_count:
                warnings.append({"code": "MISSING_VALUES", "count": missing_count})
            if unknown:
                warnings.append({"code": "UNKNOWN_CATEGORIES", "fields": unknown})
            predictions.append({
                "record_id": ids[index], "repayment_difficulty_probability": float(scores[index]),
                "risk_band": str(decisions["risk_band"][index]),
                "priority_review": bool(decisions["priority_review"][index]),
                "warnings": warnings,
            })
        return {
            "schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
            "model_release": self.contract.release,
            "feature_schema_sha256": self.contract.feature_schema_sha256,
            "policy": {name: self._policy[name] for name in (
                "medium_cutoff", "high_cutoff", "review_cutoff", "comparison",
                "calibration_method", "automatic_credit_decision",
            )},
            "predictions": predictions,
            "notice": "공개 데이터 기반 우선검토 시뮬레이션이며 대출 승인·거절 판단이 아닙니다.",
        }
