"""Stage 6 동결 모델의 무결성 확인과 공통 점수 정책."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from creditlens.modeling.calibration import validate_scores

DEFAULT_BUNDLE = Path("models/stage6/frozen_v1")
DEFAULT_RESULT = Path("reports/stage6_final_results.json")
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib")
RUNTIME_SOURCES = (
    "src/creditlens/modeling/preprocessing.py",
    "src/creditlens/modeling/calibration.py",
    "src/creditlens/modeling/frozen_model.py",
)


class FrozenModelError(ValueError):
    """동결 산출물이 기록과 다르거나 정책이 잘못된 경우."""


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{p: version(p) for p in RUNTIME_PACKAGES}}


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != "1.0" or policy.get("status") != "frozen":
        raise FrozenModelError("지원하지 않는 동결 정책입니다.")
    low, high = policy.get("medium_cutoff"), policy.get("high_cutoff")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (low, high)):
        raise FrozenModelError("cutoff는 수치여야 합니다.")
    if not 0 <= low < high <= 1:
        raise FrozenModelError("위험구간 경계가 잘못됐습니다.")
    if (policy.get("review_cutoff") != high
            or policy.get("comparison") != ">="
            or policy.get("calibration_method") != "identity"
            or policy.get("group_specific_cutoffs") is not False
            or policy.get("automatic_credit_decision") is not False):
        raise FrozenModelError("우선검토 정책이 계약과 다릅니다.")


def apply_risk_policy(scores: Any, policy: dict[str, Any]) -> dict[str, np.ndarray]:
    """전체 정밀도 점수로 경계를 판정한다. 표시 반올림은 호출자가 나중에 한다."""
    validate_policy(policy)
    values = validate_scores(scores)
    bands = np.full(values.shape, "low", dtype="<U6")
    bands[values >= policy["medium_cutoff"]] = "medium"
    bands[values >= policy["high_cutoff"]] = "high"
    return {"risk_band": bands, "priority_review": values >= policy["review_cutoff"]}


def verify_bundle(
    bundle: Path = DEFAULT_BUNDLE,
    result_path: Path = DEFAULT_RESULT,
    *,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    """Git 관리 결과의 manifest 해시부터 검증한다. 데이터·pickle은 읽지 않는다.

    결과 JSON은 신뢰하는 버전관리 기록이어야 한다. 체크섬은 실수로 바뀐 파일을
    검출하며 공격자가 결과와 모델을 함께 바꾸는 상황의 디지털 서명은 아니다.
    """
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("run_status") != "complete" or result.get("stage_part") != "3/3"
            or result.get("data_scope", {}).get("test_feature_rows_used") != 0):
        raise FrozenModelError("완료된 Stage 6 동결 기록이 필요합니다.")
    manifest_path = bundle / "manifest.json"
    if file_digest(manifest_path) != result["bundle_manifest"]["sha256"]:
        raise FrozenModelError("동결 manifest SHA-256 불일치")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("release") != result.get("release"):
        raise FrozenModelError("동결 버전 상태 불일치")
    if set(manifest["artifacts"]) != {"model.joblib", "calibrator.joblib", "policy.json"}:
        raise FrozenModelError("동결 파일 목록이 계약과 다릅니다.")
    for name, metadata in manifest["artifacts"].items():
        path = bundle / name
        if path.stat().st_size != metadata["bytes"] or file_digest(path) != metadata["sha256"]:
            raise FrozenModelError(f"동결 파일 SHA-256/크기 불일치: {name}")
    if manifest["runtime"] != runtime_versions():
        raise FrozenModelError("동결 당시 Python/패키지 버전과 실행환경이 다릅니다.")
    if set(manifest["runtime_sources"]) != set(RUNTIME_SOURCES):
        raise FrozenModelError("추론 소스 목록 불일치")
    for relative, digest in manifest["runtime_sources"].items():
        if file_digest(project_root / relative) != digest:
            raise FrozenModelError(f"추론 소스 SHA-256 불일치: {relative}")
    policy = json.loads((bundle / "policy.json").read_text(encoding="utf-8"))
    validate_policy(policy)
    if policy != result["policy"]:
        raise FrozenModelError("공유 정책과 동결 정책 불일치")
    return manifest


def load_frozen_model(
    bundle: Path = DEFAULT_BUNDLE, result_path: Path = DEFAULT_RESULT,
) -> tuple[Any, Any, dict[str, Any]]:
    """검증을 통과한 로컬 모델만 역직렬화한다."""
    verify_bundle(bundle, result_path)
    pipeline = joblib.load(bundle / "model.joblib")
    calibrator = joblib.load(bundle / "calibrator.joblib")
    policy = json.loads((bundle / "policy.json").read_text(encoding="utf-8"))
    return pipeline, calibrator, policy
