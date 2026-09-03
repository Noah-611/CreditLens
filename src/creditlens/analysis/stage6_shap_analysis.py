"""Stage 6 1/3 고정 LightGBM의 SHAP·Top-K 오류 분석.

Stage 5에서 잠근 모델을 다시 학습하거나 바꾸지 않는다. 공식 validation에서
Tree SHAP을 계산해 전역 중요도와 Top 10% 포착·누락 패턴을 집계하고, test는
계속 봉인한다. 고객별 설명은 Git에서 제외되는 로컬 산출물에만 저장한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "creditlens-matplotlib"),
)

import joblib
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy import sparse
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.data import DEFAULT_MART_PATHS, ModelSplit, load_model_split
from creditlens.modeling.preprocessing import transformed_feature_names
from creditlens.modeling.train_lightgbm import (
    _atomic_joblib_dump,
    _atomic_write_json,
    _atomic_write_text,
    _display_path,
    _git_ignored,
    _peak_rss_mb,
    _sha256,
)


SCHEMA_VERSION = "1.0"
RUN_VERSION = "stage6-shap-analysis-v1"
STAGE_PART = "1/3"
RANDOM_SEED = 42
TOP_FRACTION = 0.10
TOP_SOURCE_FEATURES = 50
TOP_REPORT_FEATURES = 15
TOP_DIRECTION_FEATURES = 15
TOP_ERROR_FEATURES = 20
DIRECTION_PLOT_SAMPLE = 5_000
PLOT_FEATURES = 12
LOCAL_TOP_FEATURES = 10
SCORE_ATOL = 1e-6
ADDITIVITY_ATOL = 1e-5

DEFAULT_STAGE5_RESULT = Path("reports/stage5_final_results.json")
DEFAULT_MODEL = Path("models/stage5/stage5_v3_lightgbm_candidate.joblib")
DEFAULT_CALIBRATOR = Path(
    "models/stage5/stage5_v3_probability_calibrator.joblib"
)
DEFAULT_LOCK_MANIFEST = Path(
    "models/stage5/stage5_v3_candidate_lock_manifest.json"
)
DEFAULT_VALIDATION_SCORES = Path(
    "models/stage5/stage5_final_validation_scores.joblib"
)
DEFAULT_LOCAL_ARTIFACT = Path("models/stage6/stage6_local_shap_explanations.joblib")
DEFAULT_OUTPUT = Path("reports/stage6_shap_analysis.json")
DEFAULT_REPORT = Path("docs/Stage6_SHAP_Analysis_Report.md")
DEFAULT_GLOBAL_FIGURE = Path("reports/figures/stage6_shap_global_importance.png")
DEFAULT_DIRECTION_FIGURE = Path("reports/figures/stage6_shap_direction.png")
DEFAULT_ERROR_FIGURE = Path("reports/figures/stage6_shap_captured_vs_missed.png")

GROUP_LABELS = {
    "application": "신청정보",
    "bureau": "외부 신용이력",
    "installments": "과거 납부이력",
}

FEATURE_LABELS = {
    "EXT_SOURCE_1": "외부 신용평가값 1",
    "EXT_SOURCE_2": "외부 신용평가값 2",
    "EXT_SOURCE_3": "외부 신용평가값 3",
    "APP_EXT_SOURCE_MEAN": "외부 신용평가값 평균",
    "APP_EXT_SOURCE_OBSERVED_COUNT": "외부 신용평가값 관측 개수",
    "DAYS_BIRTH": "나이 관련 상대일수",
    "APP_AGE_YEARS": "나이",
    "DAYS_EMPLOYED": "재직기간 관련 상대일수",
    "APP_EMPLOYED_YEARS": "재직기간",
    "AMT_CREDIT": "대출 신청금액",
    "AMT_ANNUITY": "연간 상환액",
    "AMT_INCOME_TOTAL": "연소득",
    "APP_CREDIT_INCOME_RATIO": "소득 대비 대출금액",
    "APP_ANNUITY_INCOME_RATIO": "소득 대비 연간 상환액",
    "APP_CREDIT_ANNUITY_RATIO": "연간 상환액 대비 대출금액",
    "NAME_EDUCATION_TYPE": "교육 수준",
    "ORGANIZATION_TYPE": "근무 조직 유형",
    "OCCUPATION_TYPE": "직업 유형",
}


class Stage6ShapError(RuntimeError):
    """Stage 6 SHAP 실행 계약이 깨졌을 때 발생한다."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage6ShapError(f"{label} JSON을 읽을 수 없습니다: {path}") from error
    if not isinstance(value, dict):
        raise Stage6ShapError(f"{label} JSON 최상위는 객체여야 합니다.")
    return value


def _ordered_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _artifact_metadata(path: Path, *, require_ignored: bool) -> dict[str, Any]:
    if not path.is_file():
        raise Stage6ShapError(f"산출물을 찾을 수 없습니다: {path}")
    try:
        ignored = _git_ignored(path)
    except RuntimeError as error:
        raise Stage6ShapError(str(error)) from error
    if require_ignored and ignored is False:
        raise Stage6ShapError(f"로컬 산출물이 Git에서 제외되지 않습니다: {path}")
    return {
        "display_path": _display_path(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "git_ignored": ignored,
    }


def _verify_artifact(
    path: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    actual = _artifact_metadata(path, require_ignored=True)
    if actual["sha256"] != expected.get("sha256"):
        raise Stage6ShapError(f"{label} SHA-256이 Stage 5 기록과 다릅니다.")
    if actual["bytes"] != expected.get("bytes"):
        raise Stage6ShapError(f"{label} 파일 크기가 Stage 5 기록과 다릅니다.")
    return actual


def _validate_paths(
    *,
    output: Path,
    report: Path,
    figures: Sequence[Path],
    local_artifact: Path,
    references: Sequence[Path],
) -> None:
    if output.suffix.lower() != ".json":
        raise Stage6ShapError("공유 결과 경로는 .json이어야 합니다.")
    if report.suffix.lower() != ".md":
        raise Stage6ShapError("보고서 경로는 .md여야 합니다.")
    if any(path.suffix.lower() != ".png" for path in figures):
        raise Stage6ShapError("그림 경로는 .png여야 합니다.")
    if local_artifact.suffix.lower() != ".joblib":
        raise Stage6ShapError("고객별 로컬 설명 경로는 .joblib이어야 합니다.")
    paths = (output, report, *figures, local_artifact, *references)
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise Stage6ShapError("입력과 출력 경로는 서로 달라야 합니다.")
    local_root = local_artifact.parent.resolve()
    for shared in (output, report, *figures):
        try:
            shared.resolve().relative_to(local_root)
        except ValueError:
            continue
        raise Stage6ShapError("공유 산출물은 Git 제외 로컬 경로에 둘 수 없습니다.")


def _validate_stage5_reference(
    stage5: Mapping[str, Any],
    *,
    model_path: Path,
    calibrator_path: Path,
    lock_path: Path,
    validation_scores_path: Path,
) -> dict[str, dict[str, Any]]:
    if stage5.get("run_status") != "complete":
        raise Stage6ShapError("완료된 Stage 5 결과가 필요합니다.")
    scope = stage5.get("data_scope")
    if not isinstance(scope, dict) or scope.get("test_feature_rows_used") != 0:
        raise Stage6ShapError("Stage 5의 test 봉인 기록이 올바르지 않습니다.")
    candidate = stage5.get("stage6_candidate")
    if (
        not isinstance(candidate, dict)
        or candidate.get("base_model_key") != "stage5_selected_lightgbm_v3"
        or candidate.get("test_evaluated") is not False
    ):
        raise Stage6ShapError("Stage 6 전달 후보 계약이 올바르지 않습니다.")
    artifacts = stage5.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Stage6ShapError("Stage 5 산출물 메타데이터가 없습니다.")
    required = {
        "model": (model_path, "Stage 5 모델"),
        "calibrator": (calibrator_path, "Stage 5 보정기"),
        "lock_manifest": (lock_path, "Stage 5 잠금 명세"),
        "validation_scores": (validation_scores_path, "Stage 5 validation 점수"),
    }
    verified: dict[str, dict[str, Any]] = {}
    for key, (path, label) in required.items():
        expected = artifacts.get(key)
        if not isinstance(expected, dict):
            raise Stage6ShapError(f"{label} 메타데이터가 없습니다.")
        verified[key] = _verify_artifact(path, expected, label=label)

    lock = _read_json(lock_path, label="Stage 5 잠금 명세")
    if lock.get("test_feature_rows_used") != 0:
        raise Stage6ShapError("잠금 명세의 test 봉인 기록이 올바르지 않습니다.")
    for key, lock_key in (("model", "model_artifact"), ("calibrator", "calibrator_artifact")):
        metadata = lock.get(lock_key)
        if not isinstance(metadata, dict) or metadata.get("sha256") != verified[key]["sha256"]:
            raise Stage6ShapError(f"잠금 명세의 {key} 해시가 실제 파일과 다릅니다.")
    return verified


def _extract_pipeline_contract(
    pipeline: Any,
) -> tuple[Pipeline, Any, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(pipeline, Pipeline):
        raise Stage6ShapError("Stage 5 모델은 sklearn Pipeline이어야 합니다.")
    if tuple(pipeline.named_steps) != ("preprocessor", "model"):
        raise Stage6ShapError("Stage 5 모델 파이프라인 단계가 계약과 다릅니다.")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "booster_"):
        raise Stage6ShapError("학습된 LightGBM booster가 없습니다.")
    contract = getattr(preprocessor, "named_steps", {}).get("contract")
    if contract is None:
        raise Stage6ShapError("전처리 피처 계약을 찾을 수 없습니다.")
    features = tuple(str(value) for value in contract.feature_columns)
    numeric = tuple(str(value) for value in contract.numeric_columns)
    categorical = tuple(str(value) for value in contract.categorical_columns)
    if set(numeric).intersection(categorical) or set(numeric).union(categorical) != set(features):
        raise Stage6ShapError("수치형·범주형 피처 계약이 올바르지 않습니다.")
    return pipeline, model, features, numeric, categorical


def _resolve_source_groups(
    features: Sequence[str],
    reference_groups: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    groups = {
        "application": tuple(
            value
            for value in features
            if not value.startswith("BUREAU_") and not value.startswith("INST_")
        ),
        "bureau": tuple(value for value in features if value.startswith("BUREAU_")),
        "installments": tuple(value for value in features if value.startswith("INST_")),
    }
    flattened = tuple(value for group in groups.values() for value in group)
    if len(flattened) != len(features) or set(flattened) != set(features):
        raise Stage6ShapError("피처 원천 그룹이 V3 입력 전체를 정확히 덮지 않습니다.")
    source_group: dict[str, str] = {}
    for name, columns in groups.items():
        expected = reference_groups.get(name)
        if not isinstance(expected, dict):
            raise Stage6ShapError(f"Stage 5의 {name} 피처군 기록이 없습니다.")
        if len(columns) != expected.get("columns"):
            raise Stage6ShapError(f"{name} 피처 수가 Stage 5 기록과 다릅니다.")
        if _ordered_sha256(columns) != expected.get("columns_sha256"):
            raise Stage6ShapError(f"{name} 피처 순서·해시가 Stage 5 기록과 다릅니다.")
        for column in columns:
            source_group[column] = name
    return groups, source_group


def _map_transformed_features(
    transformed: Sequence[str],
    source_features: Sequence[str],
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> tuple[np.ndarray, tuple[dict[str, str], ...]]:
    source_index = {name: index for index, name in enumerate(source_features)}
    numeric_set = set(numeric)
    categorical_by_length = sorted(categorical, key=len, reverse=True)
    indices: list[int] = []
    audit: list[dict[str, str]] = []
    for transformed_name in transformed:
        if transformed_name.startswith("numeric__missingindicator_"):
            source = transformed_name.removeprefix("numeric__missingindicator_")
            kind = "missing_indicator"
            if source not in numeric_set:
                raise Stage6ShapError(f"알 수 없는 수치 결측 피처입니다: {transformed_name}")
        elif transformed_name.startswith("numeric__"):
            source = transformed_name.removeprefix("numeric__")
            kind = "numeric_value"
            if source not in numeric_set:
                raise Stage6ShapError(f"알 수 없는 수치 피처입니다: {transformed_name}")
        elif transformed_name.startswith("categorical__"):
            remainder = transformed_name.removeprefix("categorical__")
            matches = [
                column
                for column in categorical_by_length
                if remainder.startswith(f"{column}_")
            ]
            if not matches:
                raise Stage6ShapError(f"알 수 없는 범주 피처입니다: {transformed_name}")
            source = matches[0]
            kind = "category_level"
        else:
            raise Stage6ShapError(f"지원하지 않는 변환 피처명입니다: {transformed_name}")
        indices.append(source_index[source])
        audit.append(
            {
                "transformed_feature": transformed_name,
                "source_feature": source,
                "component_kind": kind,
            }
        )
    return np.asarray(indices, dtype=np.int32), tuple(audit)


def _aggregate_source_shap(
    shap_values: np.ndarray,
    transformed_to_source: np.ndarray,
    source_count: int,
) -> np.ndarray:
    values = np.asarray(shap_values, dtype=np.float64)
    mapping = np.asarray(transformed_to_source)
    if values.ndim != 2 or mapping.shape != (values.shape[1],):
        raise Stage6ShapError("SHAP 행렬과 변환 피처 매핑 크기가 다릅니다.")
    if source_count < 1 or mapping.min() < 0 or mapping.max() >= source_count:
        raise Stage6ShapError("원본 피처 인덱스가 범위를 벗어났습니다.")
    aggregated = np.zeros((values.shape[0], source_count), dtype=np.float64)
    for transformed_index, source in enumerate(mapping):
        aggregated[:, int(source)] += values[:, transformed_index]
    if not np.allclose(
        aggregated.sum(axis=1), values.sum(axis=1), rtol=0.0, atol=1e-10
    ):
        raise Stage6ShapError("원본 피처 집계 과정에서 SHAP 합이 보존되지 않았습니다.")
    return aggregated


def _dense_shap_values(values: Any) -> np.ndarray:
    """SHAP이 입력 형식을 따라 반환한 희소/밀집 값을 float64로 통일한다."""

    if sparse.issparse(values):
        return values.toarray().astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)


def _top_k_mask(scores: np.ndarray, fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise Stage6ShapError("Top-K 점수는 유한한 1차원 배열이어야 합니다.")
    if not 0.0 < fraction <= 1.0:
        raise Stage6ShapError("Top-K 비율은 (0, 1]이어야 합니다.")
    k = max(1, math.ceil(values.size * fraction))
    order = np.argsort(-values, kind="stable")
    selected = np.zeros(values.size, dtype=bool)
    selected[order[:k]] = True
    cutoff = float(values[order[k - 1]])
    ties = int(np.count_nonzero(values == cutoff))
    return selected, {
        "requested_fraction": fraction,
        "k": k,
        "actual_fraction": k / values.size,
        "cutoff_score": cutoff,
        "boundary_tie_count": ties,
        "tie_policy_for_group_analysis": "stable_row_order_exact_k",
    }


def _python_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return str(value)


def _source_importance(
    values: np.ndarray,
    source_features: Sequence[str],
    source_group: Mapping[str, str],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    mean_abs = np.mean(np.abs(values), axis=0)
    mean_signed = np.mean(values, axis=0)
    total = float(mean_abs.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise Stage6ShapError("SHAP 전역 중요도 합이 양수가 아닙니다.")
    order = np.argsort(-mean_abs, kind="stable")
    records = [
        {
            "rank": rank,
            "feature": source_features[index],
            "feature_label": FEATURE_LABELS.get(source_features[index]),
            "source_group": source_group[source_features[index]],
            "mean_abs_shap": float(mean_abs[index]),
            "mean_shap": float(mean_signed[index]),
            "importance_share": float(mean_abs[index] / total),
        }
        for rank, index in enumerate(order[:TOP_SOURCE_FEATURES], start=1)
    ]
    return records, mean_abs


def _source_group_importance(
    mean_abs: np.ndarray,
    source_features: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    feature_index = {name: index for index, name in enumerate(source_features)}
    total = float(mean_abs.sum())
    records = []
    for name in ("application", "bureau", "installments"):
        value = float(sum(mean_abs[feature_index[column]] for column in groups[name]))
        records.append(
            {
                "source_group": name,
                "display_name": GROUP_LABELS[name],
                "feature_count": len(groups[name]),
                "summed_mean_abs_shap": value,
                "attribution_share": value / total,
            }
        )
    return records


def _direction_statistics(
    frame: pd.DataFrame,
    source_values: np.ndarray,
    source_features: Sequence[str],
    numeric: Sequence[str],
    importance: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    index = {name: position for position, name in enumerate(source_features)}
    numeric_set = set(numeric)
    selected = [item["feature"] for item in importance if item["feature"] in numeric_set][
        :TOP_DIRECTION_FEATURES
    ]
    records: list[dict[str, Any]] = []
    for feature in selected:
        raw = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64)
        shap_column = source_values[:, index[feature]]
        valid = np.isfinite(raw)
        observed = raw[valid]
        contributions = shap_column[valid]
        if observed.size < 3 or np.unique(observed).size < 2:
            correlation = None
            high_minus_low = None
            direction = "insufficient_variation"
        else:
            correlation_result = spearmanr(observed, contributions)
            correlation = float(correlation_result.statistic)
            lower, upper = np.quantile(observed, [0.2, 0.8])
            low = contributions[observed <= lower]
            high = contributions[observed >= upper]
            high_minus_low = float(high.mean() - low.mean())
            if correlation >= 0.1:
                direction = "higher_values_tend_to_raise_risk_score"
            elif correlation <= -0.1:
                direction = "higher_values_tend_to_lower_risk_score"
            else:
                direction = "weak_or_non_monotonic_relationship"
        records.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature),
                "observed_rows": int(valid.sum()),
                "missing_rows": int((~valid).sum()),
                "spearman_feature_value_vs_shap": correlation,
                "top20pct_minus_bottom20pct_mean_shap": high_minus_low,
                "direction": direction,
            }
        )
    return records


def _transformed_importance(
    values: np.ndarray,
    mapping_audit: Sequence[Mapping[str, str]],
    source_group: Mapping[str, str],
) -> list[dict[str, Any]]:
    mean_abs = np.mean(np.abs(values), axis=0)
    mean_signed = np.mean(values, axis=0)
    order = np.argsort(-mean_abs, kind="stable")[:TOP_SOURCE_FEATURES]
    return [
        {
            "rank": rank,
            **dict(mapping_audit[index]),
            "source_group": source_group[mapping_audit[index]["source_feature"]],
            "mean_abs_shap": float(mean_abs[index]),
            "mean_shap": float(mean_signed[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def _group_summary(
    mask: np.ndarray,
    scores: np.ndarray,
    source_values: np.ndarray,
    source_features: Sequence[str],
    source_group: Mapping[str, str],
) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        raise Stage6ShapError("오류 분석 그룹이 비어 있습니다.")
    subset = source_values[mask]
    mean_abs = np.mean(np.abs(subset), axis=0)
    mean_signed = np.mean(subset, axis=0)
    order = np.argsort(-mean_abs, kind="stable")[:10]
    return {
        "rows": count,
        "mean_score": float(np.mean(scores[mask])),
        "median_score": float(np.median(scores[mask])),
        "top_features": [
            {
                "feature": source_features[index],
                "feature_label": FEATURE_LABELS.get(source_features[index]),
                "source_group": source_group[source_features[index]],
                "mean_abs_shap": float(mean_abs[index]),
                "mean_shap": float(mean_signed[index]),
            }
            for index in order
        ],
    }


def _error_analysis(
    labels: np.ndarray,
    scores: np.ndarray,
    source_values: np.ndarray,
    source_features: Sequence[str],
    source_group: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reviewed, top_k = _top_k_mask(scores, TOP_FRACTION)
    positive = labels == 1
    masks = {
        "captured_positive": reviewed & positive,
        "missed_positive": ~reviewed & positive,
        "reviewed_negative": reviewed & ~positive,
        "not_reviewed_negative": ~reviewed & ~positive,
    }
    summaries = {
        name: _group_summary(
            mask, scores, source_values, source_features, source_group
        )
        for name, mask in masks.items()
    }
    captured_mean = source_values[masks["captured_positive"]].mean(axis=0)
    missed_mean = source_values[masks["missed_positive"]].mean(axis=0)
    difference = captured_mean - missed_mean
    order = np.argsort(-np.abs(difference), kind="stable")[:TOP_ERROR_FEATURES]
    comparison = [
        {
            "rank": rank,
            "feature": source_features[index],
            "feature_label": FEATURE_LABELS.get(source_features[index]),
            "source_group": source_group[source_features[index]],
            "captured_positive_mean_shap": float(captured_mean[index]),
            "missed_positive_mean_shap": float(missed_mean[index]),
            "captured_minus_missed_mean_shap": float(difference[index]),
            "interpretation": (
                "stronger_risk_raising_signal_in_captured_positive"
                if difference[index] > 0
                else "stronger_risk_raising_or_weaker_protective_signal_in_missed_positive"
            ),
        }
        for rank, index in enumerate(order, start=1)
    ]
    metrics = evaluate_binary_metrics(labels, scores, top_fraction=TOP_FRACTION)
    return (
        {
            "review_policy": top_k,
            "common_metric_result": metrics["top_k_metrics"],
            "groups": summaries,
            "captured_vs_missed_positive": comparison,
            "scope": "validation_top10_error_diagnosis_not_final_operating_cutoff",
        },
        masks,
    )


def _representative_explanations(
    frame: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    base_values: np.ndarray,
    source_values: np.ndarray,
    source_features: Sequence[str],
    source_group: Mapping[str, str],
    masks: Mapping[str, np.ndarray],
    cutoff: float,
) -> list[dict[str, Any]]:
    cases = (
        ("captured_positive_high_score", "captured_positive", True),
        ("missed_positive_near_cutoff", "missed_positive", True),
        ("reviewed_negative_high_score", "reviewed_negative", True),
        ("not_reviewed_negative_low_score", "not_reviewed_negative", False),
    )
    records: list[dict[str, Any]] = []
    for case_key, group, choose_high in cases:
        candidates = np.flatnonzero(masks[group])
        selected = candidates[
            np.argmax(scores[candidates]) if choose_high else np.argmin(scores[candidates])
        ]
        order = np.argsort(-np.abs(source_values[selected]), kind="stable")[
            :LOCAL_TOP_FEATURES
        ]
        records.append(
            {
                "case_key": case_key,
                "group": group,
                "customer_id_included": False,
                "target": int(labels[selected]),
                "risk_score": float(scores[selected]),
                "top10_cutoff": cutoff,
                "base_value_raw_log_odds": float(base_values[selected]),
                "reconstructed_raw_log_odds": float(
                    base_values[selected] + source_values[selected].sum()
                ),
                "top_contributions": [
                    {
                        "feature": source_features[index],
                        "feature_label": FEATURE_LABELS.get(source_features[index]),
                        "source_group": source_group[source_features[index]],
                        "feature_value": _python_value(frame.iloc[selected][source_features[index]]),
                        "shap_raw_log_odds": float(source_values[selected, index]),
                        "effect": (
                            "raises_risk_score"
                            if source_values[selected, index] > 0
                            else "lowers_risk_score"
                        ),
                    }
                    for index in order
                ],
            }
        )
    return records


def _atomic_save_figure(figure: plt.Figure, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        figure.savefig(temporary, dpi=150, bbox_inches="tight", format="png")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return {
        "display_path": _display_path(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _plot_global_importance(
    importance: Sequence[Mapping[str, Any]], path: Path
) -> dict[str, Any]:
    selected = list(importance[:20])[::-1]
    colors = {
        "application": "#4C78A8",
        "bureau": "#F58518",
        "installments": "#54A24B",
    }
    figure, axis = plt.subplots(figsize=(9.0, 7.5))
    axis.barh(
        [str(item["feature"]) for item in selected],
        [float(item["mean_abs_shap"]) for item in selected],
        color=[colors[str(item["source_group"])] for item in selected],
    )
    axis.set_xlabel("Mean absolute SHAP value (raw log-odds)")
    axis.set_title("Stage 6 global feature importance")
    axis.grid(axis="x", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=8, label=name.title())
        for name, color in colors.items()
    ]
    axis.legend(handles=handles, loc="lower right")
    return _atomic_save_figure(figure, path)


def _plot_direction(
    frame: pd.DataFrame,
    source_values: np.ndarray,
    source_features: Sequence[str],
    direction: Sequence[Mapping[str, Any]],
    path: Path,
) -> dict[str, Any]:
    selected = [str(item["feature"]) for item in direction[:PLOT_FEATURES]][::-1]
    feature_index = {name: index for index, name in enumerate(source_features)}
    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = min(DIRECTION_PLOT_SAMPLE, len(frame))
    sample = np.sort(rng.choice(len(frame), size=sample_size, replace=False))
    figure, axis = plt.subplots(figsize=(9.5, 7.0))
    color_artist = None
    for y_position, feature in enumerate(selected):
        raw = pd.to_numeric(frame.iloc[sample][feature], errors="coerce")
        percentiles = raw.rank(method="average", pct=True).fillna(0.5).to_numpy()
        jitter = rng.normal(0.0, 0.08, size=sample_size)
        color_artist = axis.scatter(
            source_values[sample, feature_index[feature]],
            y_position + jitter,
            c=percentiles,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=8,
            alpha=0.35,
            linewidths=0,
        )
    axis.axvline(0.0, color="#777777", linewidth=1.0)
    axis.set_yticks(np.arange(len(selected)), selected)
    axis.set_xlabel("SHAP contribution to risk score (raw log-odds)")
    axis.set_title("Feature value direction for leading numeric features")
    axis.grid(axis="x", alpha=0.2)
    if color_artist is not None:
        colorbar = figure.colorbar(color_artist, ax=axis, pad=0.02)
        colorbar.set_label("Feature-value percentile (low to high)")
    return _atomic_save_figure(figure, path)


def _plot_error_difference(
    comparison: Sequence[Mapping[str, Any]], path: Path
) -> dict[str, Any]:
    selected = list(comparison[:PLOT_FEATURES])[::-1]
    values = [float(item["captured_minus_missed_mean_shap"]) for item in selected]
    figure, axis = plt.subplots(figsize=(9.0, 6.2))
    axis.barh(
        [str(item["feature"]) for item in selected],
        values,
        color=["#E45756" if value >= 0 else "#4C78A8" for value in values],
    )
    axis.axvline(0.0, color="#555555", linewidth=1.0)
    axis.set_xlabel("Mean SHAP difference: captured positive - missed positive")
    axis.set_title("Why Top 10% captured some positive cases")
    axis.grid(axis="x", alpha=0.25)
    return _atomic_save_figure(figure, path)


def _markdown_relative(report: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=report.resolve().parent).replace(
        os.sep, "/"
    )


def _format(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _feature_display(item: Mapping[str, Any]) -> str:
    name = str(item["feature"])
    label = item.get("feature_label")
    return f"`{name}` ({label})" if label else f"`{name}`"


def render_markdown_report(
    payload: Mapping[str, Any],
    *,
    report_path: str | Path = DEFAULT_REPORT,
    global_figure_path: str | Path = DEFAULT_GLOBAL_FIGURE,
    direction_figure_path: str | Path = DEFAULT_DIRECTION_FIGURE,
    error_figure_path: str | Path = DEFAULT_ERROR_FIGURE,
) -> str:
    report = Path(report_path)
    global_result = payload["global_explanation"]
    error = payload["error_analysis"]
    replay = payload["validation_replay"]
    lines = [
        "# Stage 6 1/3 SHAP·오류 분석 보고서",
        "",
        "> Stage 5에서 선택한 V3 LightGBM을 변경하거나 다시 학습하지 않고 validation에서 해석했습니다. test는 사용하지 않았으며, 고객별 상세 설명은 Git 제외 로컬 파일에만 저장했습니다.",
        "",
        "## 왜 이 분석을 했는가",
        "",
        "ROC-AUC와 PR-AUC는 모델이 얼마나 잘 구분하는지는 알려주지만 어떤 정보 때문에 점수가 달라졌는지는 알려주지 않습니다. SHAP으로 각 피처가 모델의 위험점수를 올리거나 낮춘 정도를 계산해 모델이 합리적인 신호를 사용하는지 확인하고, Top 10% 우선검토에서 포착한 상환곤란 고객과 놓친 고객의 차이를 진단했습니다.",
        "",
        "## 분석 계약",
        "",
        f"- 분석 데이터: validation {payload['data_scope']['validation_rows']:,}명",
        f"- SHAP 계산 행: {payload['data_scope']['shap_rows']:,}명(전체 validation)",
        f"- 원본 피처 {payload['model_contract']['source_feature_columns']}개 → 전처리 후 {payload['model_contract']['transformed_feature_columns']}개 → 원본 피처 단위로 재집계",
        "- 알고리즘: LightGBM Tree SHAP, 모델 내부 raw log-odds 출력",
        "- 양수 SHAP: 해당 피처가 모델 위험점수를 높인 방향, 음수 SHAP: 낮춘 방향",
        "- SHAP은 모델 내부 연관성을 설명하며 원인·인과관계를 증명하지 않음",
        f"- test 피처 사용: {payload['data_scope']['test_feature_rows_used']}행",
        "",
        "## Stage 5 모델 재현 확인",
        "",
        f"- Stage 5 저장 점수와 현재 모델 점수 최대 차이: `{replay['max_abs_probability_difference']:.10f}`",
        f"- SHAP 합과 LightGBM raw log-odds 최대 차이: `{replay['max_abs_additivity_error']:.10f}`",
        f"- 현재 validation ROC-AUC / PR-AUC: `{_format(replay['metrics']['roc_auc'])}` / `{_format(replay['metrics']['pr_auc'])}`",
        "",
        "## 전역 SHAP 중요도",
        "",
        "| 순위 | 원본 피처 | 정보 원천 | 평균 |SHAP| | 중요도 비중 | 평균 방향 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for item in global_result["top_source_features"][:TOP_REPORT_FEATURES]:
        lines.append(
            f"| {item['rank']} | {_feature_display(item)} | {GROUP_LABELS[item['source_group']]} | "
            f"{_format(item['mean_abs_shap'], 6)} | {_format(item['importance_share'] * 100, 2)}% | "
            f"{_format(item['mean_shap'], 6)} |"
        )
    lines.extend(
        [
            "",
            f"![전역 SHAP 중요도]({_markdown_relative(report, Path(global_figure_path))})",
            "",
            "평균 `|SHAP|`은 영향의 크기이고 평균 방향은 전체 고객에서 양·음 기여가 상쇄될 수 있으므로 중요도와 같은 뜻이 아닙니다. 상관된 피처들은 중요도를 나눠 가질 수 있습니다.",
            "",
            "### 정보 원천별 SHAP 비중",
            "",
            "| 정보 원천 | 피처 수 | SHAP 비중 |",
            "|---|---:|---:|",
        ]
    )
    for item in global_result["source_group_importance"]:
        lines.append(
            f"| {item['display_name']} | {item['feature_count']} | "
            f"{_format(item['attribution_share'] * 100, 2)}% |"
        )
    lines.extend(
        [
            "",
            "## 주요 수치형 피처의 방향",
            "",
            f"![SHAP 방향]({_markdown_relative(report, Path(direction_figure_path))})",
            "",
            "점 하나는 validation 고객 한 명이며 색은 해당 피처값의 낮음→높음 순위입니다. 오른쪽은 위험점수를 높인 기여, 왼쪽은 낮춘 기여입니다. 비선형 모델이므로 한 피처가 모든 구간에서 같은 방향으로 작동한다고 단정하지 않습니다.",
            "",
            "## Top 10% 포착·누락 분석",
            "",
            f"- 우선검토 인원: {error['review_policy']['k']:,}명",
            f"- 점수 cutoff: {_format(error['review_policy']['cutoff_score'], 6)}",
            f"- 위험고객 포착: {error['groups']['captured_positive']['rows']:,}명",
            f"- 위험고객 누락: {error['groups']['missed_positive']['rows']:,}명",
            f"- Recall@Top10%: {_format(error['common_metric_result']['recall'])}",
            f"- Lift@Top10%: {_format(error['common_metric_result']['lift'])}",
            "",
            "| 그룹 | 고객 수 | 평균 위험점수 | 중앙 위험점수 |",
            "|---|---:|---:|---:|",
        ]
    )
    group_names = {
        "captured_positive": "상위 10%에서 포착한 위험고객",
        "missed_positive": "상위 10% 밖에서 놓친 위험고객",
        "reviewed_negative": "상위 10%의 정상고객",
        "not_reviewed_negative": "상위 10% 밖의 정상고객",
    }
    for key, label in group_names.items():
        item = error["groups"][key]
        lines.append(
            f"| {label} | {item['rows']:,} | {_format(item['mean_score'], 6)} | "
            f"{_format(item['median_score'], 6)} |"
        )
    lines.extend(
        [
            "",
            f"![포착·누락 SHAP 차이]({_markdown_relative(report, Path(error_figure_path))})",
            "",
            "아래 양수 차이는 해당 피처의 위험 상승 기여가 포착된 위험고객에서 더 강했다는 뜻입니다. 음수 차이는 놓친 위험고객 쪽에서 상대적으로 더 강한 위험 신호였거나 보호 신호가 더 약했다는 뜻입니다.",
            "",
            "| 순위 | 피처 | 정보 원천 | 포착 평균 SHAP | 누락 평균 SHAP | 포착−누락 |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for item in error["captured_vs_missed_positive"][:12]:
        lines.append(
            f"| {item['rank']} | {_feature_display(item)} | {GROUP_LABELS[item['source_group']]} | "
            f"{_format(item['captured_positive_mean_shap'], 6)} | "
            f"{_format(item['missed_positive_mean_shap'], 6)} | "
            f"{_format(item['captured_minus_missed_mean_shap'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## 해석 한계와 다음 단계",
            "",
            "- SHAP은 모델이 사용한 패턴을 설명할 뿐 실제 상환곤란의 원인을 증명하지 않습니다.",
            "- 결과는 해외 과거 공개 데이터의 validation 분석이며 국내 실제 금융환경으로 일반화할 수 없습니다.",
            "- Top 10%는 오류 진단용 기준이며 아직 운영 cutoff로 확정하지 않았습니다.",
            "- 고객별 설명에는 민감한 해석이 개입될 수 있으므로 사람이 원자료와 함께 검토해야 합니다.",
            "- Stage 6 2/3에서 위험구간·Top-K 시나리오와 금융이력 부족자 등 하위그룹을 분석한 뒤 개선 필요 여부를 결정합니다.",
            "- 모델 결과만으로 자동 승인·거절하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def run_stage6_shap_analysis(
    *,
    stage5_result_path: str | Path = DEFAULT_STAGE5_RESULT,
    model_path: str | Path = DEFAULT_MODEL,
    calibrator_path: str | Path = DEFAULT_CALIBRATOR,
    lock_manifest_path: str | Path = DEFAULT_LOCK_MANIFEST,
    validation_scores_path: str | Path = DEFAULT_VALIDATION_SCORES,
    local_artifact_path: str | Path = DEFAULT_LOCAL_ARTIFACT,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    global_figure_path: str | Path = DEFAULT_GLOBAL_FIGURE,
    direction_figure_path: str | Path = DEFAULT_DIRECTION_FIGURE,
    error_figure_path: str | Path = DEFAULT_ERROR_FIGURE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """잠긴 Stage 5 모델을 validation에서 SHAP으로 해석한다."""

    stage5_path = Path(stage5_result_path)
    model_file = Path(model_path)
    calibrator_file = Path(calibrator_path)
    lock_file = Path(lock_manifest_path)
    validation_scores_file = Path(validation_scores_path)
    local_file = Path(local_artifact_path)
    output = Path(output_path)
    report = Path(report_path)
    global_figure = Path(global_figure_path)
    direction_figure = Path(direction_figure_path)
    error_figure = Path(error_figure_path)
    _validate_paths(
        output=output,
        report=report,
        figures=(global_figure, direction_figure, error_figure),
        local_artifact=local_file,
        references=(stage5_path, model_file, calibrator_file, lock_file, validation_scores_file),
    )
    started = time.perf_counter()
    stage5 = _read_json(stage5_path, label="Stage 5 결과")
    verified = _validate_stage5_reference(
        stage5,
        model_path=model_file,
        calibrator_path=calibrator_file,
        lock_path=lock_file,
        validation_scores_path=validation_scores_file,
    )
    if progress is not None:
        progress("Stage 5 잠금 모델·보정기·validation 점수 해시 검증 완료")
    try:
        pipeline = joblib.load(model_file)
        calibrator = joblib.load(calibrator_file)
        stored_scores = joblib.load(validation_scores_file)
    except (OSError, ValueError, AttributeError, ImportError) as error:
        raise Stage6ShapError("Stage 5 로컬 산출물을 불러올 수 없습니다.") from error
    pipeline, model, features, numeric, categorical = _extract_pipeline_contract(
        pipeline
    )
    expected_method = stage5["calibration"]["decision"]["selected_method"]
    if getattr(calibrator, "method", None) != expected_method:
        raise Stage6ShapError("보정기 방법이 Stage 5 선택 결과와 다릅니다.")
    data_version = stage5.get("data_version")
    if not isinstance(data_version, dict):
        raise Stage6ShapError("Stage 5 데이터 버전 기록이 없습니다.")
    groups, source_group = _resolve_source_groups(
        features, data_version.get("feature_groups", {})
    )

    if progress is not None:
        progress("공식 validation 로드와 Stage 5 예측 재현 확인")
    validation: ModelSplit = load_model_split(
        DEFAULT_MART_PATHS["v3"], "v3", "validation"
    )
    if validation.name != "validation" or tuple(validation.X.columns) != features:
        raise Stage6ShapError("현재 validation 입력 계약이 잠금 모델과 다릅니다.")
    labels = validation.y.to_numpy(dtype=np.int8, copy=False)
    if len(labels) != stage5["data_scope"]["validation_rows"]:
        raise Stage6ShapError("validation 행 수가 Stage 5 기록과 다릅니다.")
    if not np.isclose(
        labels.mean(), stage5["data_scope"]["validation_positive_rate"], atol=1e-15
    ):
        raise Stage6ShapError("validation 양성률이 Stage 5 기록과 다릅니다.")
    scores = np.asarray(pipeline.predict_proba(validation.X)[:, 1], dtype=np.float64)
    stored_y = np.asarray(stored_scores.get("y_true"))
    stored_raw = np.asarray(stored_scores.get("scores", {}).get("raw"))
    if not np.array_equal(stored_y, labels) or stored_raw.shape != scores.shape:
        raise Stage6ShapError("Stage 5 validation 점수 산출물의 행 계약이 다릅니다.")
    score_difference = float(np.max(np.abs(scores - stored_raw)))
    if score_difference > SCORE_ATOL:
        raise Stage6ShapError("현재 모델 점수가 Stage 5 저장 점수와 다릅니다.")
    metrics = evaluate_binary_metrics(labels, scores, top_fraction=TOP_FRACTION)
    for name in ("roc_auc", "pr_auc", "ks", "gini", "brier_score"):
        if not np.isclose(
            metrics[name],
            stage5["official_validation"]["raw_metrics"][name],
            rtol=0.0,
            atol=SCORE_ATOL,
        ):
            raise Stage6ShapError(f"현재 validation {name}이 Stage 5 기록과 다릅니다.")

    if progress is not None:
        progress("validation 전체 전처리 후 exact Tree SHAP 계산")
    transformed = pipeline.named_steps["preprocessor"].transform(validation.X)
    transformed_names = transformed_feature_names(pipeline.named_steps["preprocessor"])
    if transformed.shape[1] != len(transformed_names):
        raise Stage6ShapError("변환 행렬과 피처 이름 수가 다릅니다.")
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    explanation = explainer(transformed, check_additivity=True)
    shap_values = _dense_shap_values(explanation.values)
    base_values = np.asarray(explanation.base_values, dtype=np.float64)
    if base_values.ndim == 0:
        base_values = np.full(len(labels), float(base_values), dtype=np.float64)
    if shap_values.shape != transformed.shape or base_values.shape != (len(labels),):
        raise Stage6ShapError("Tree SHAP 결과 크기가 validation 계약과 다릅니다.")
    raw_output = np.asarray(model.predict(transformed, raw_score=True), dtype=np.float64)
    reconstructed_raw = base_values + shap_values.sum(axis=1)
    additivity_error = float(np.max(np.abs(reconstructed_raw - raw_output)))
    probability_error = float(np.max(np.abs(expit(reconstructed_raw) - scores)))
    if additivity_error > ADDITIVITY_ATOL or probability_error > SCORE_ATOL:
        raise Stage6ShapError("SHAP 가산성 또는 확률 재구성 계약이 깨졌습니다.")

    mapping, mapping_audit = _map_transformed_features(
        transformed_names, features, numeric, categorical
    )
    source_values = _aggregate_source_shap(shap_values, mapping, len(features))
    importance, mean_abs = _source_importance(
        source_values, features, source_group
    )
    direction = _direction_statistics(
        validation.X, source_values, features, numeric, importance
    )
    transformed_importance = _transformed_importance(
        shap_values, mapping_audit, source_group
    )
    error, masks = _error_analysis(
        labels, scores, source_values, features, source_group
    )

    if progress is not None:
        progress("고객별 대표 설명은 Git 제외 로컬 산출물로 저장")
    representatives = _representative_explanations(
        validation.X,
        labels,
        scores,
        base_values,
        source_values,
        features,
        source_group,
        masks,
        error["review_policy"]["cutoff_score"],
    )
    try:
        local_metadata = _atomic_joblib_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "run_version": RUN_VERSION,
                "scope": "local_representative_validation_explanations",
                "customer_ids_included": False,
                "row_positions_included": False,
                "shap_output_space": "raw_log_odds",
                "model_sha256": verified["model"]["sha256"],
                "representatives": representatives,
            },
            local_file,
        )
    except RuntimeError as error:
        raise Stage6ShapError(str(error)) from error

    if progress is not None:
        progress("전역 중요도·방향·포착 대 누락 그림 생성")
    figures = {
        "global_importance": _plot_global_importance(importance, global_figure),
        "direction": _plot_direction(
            validation.X, source_values, features, direction, direction_figure
        ),
        "captured_vs_missed": _plot_error_difference(
            error["captured_vs_missed_positive"], error_figure
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "stage_part": STAGE_PART,
        "run_status": "complete",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "shap": shap.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "random_seed": RANDOM_SEED,
            "top_fraction_for_error_diagnosis": TOP_FRACTION,
            "shap_algorithm": "TreeExplainer_exact_tree_path_dependent",
            "shap_model_output": "raw_log_odds",
            "all_validation_rows_explained": True,
            "direction_plot_sample_rows": min(DIRECTION_PLOT_SAMPLE, len(labels)),
            "model_or_preprocessor_refit": False,
            "operating_cutoff_finalized": False,
        },
        "data_scope": {
            "validation_rows": len(labels),
            "validation_positive_rate": float(labels.mean()),
            "shap_rows": len(labels),
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
            "row_level_values_in_shared_outputs": False,
            "customer_level_details_stored_only_in_git_ignored_artifact": True,
        },
        "references": {
            "stage5_result": {
                "display_path": _display_path(stage5_path),
                "sha256": _sha256(stage5_path),
                "run_version": stage5.get("run_version"),
            },
            **verified,
        },
        "model_contract": {
            "data_version": "v3",
            "source_feature_columns": len(features),
            "numeric_feature_columns": len(numeric),
            "categorical_feature_columns": len(categorical),
            "transformed_feature_columns": len(transformed_names),
            "source_feature_names_sha256": _ordered_sha256(features),
            "transformed_feature_names_sha256": _ordered_sha256(transformed_names),
            "source_group_contract": {
                name: {
                    "columns": len(columns),
                    "columns_sha256": _ordered_sha256(columns),
                }
                for name, columns in groups.items()
            },
            "transformed_components_by_kind": {
                kind: sum(1 for item in mapping_audit if item["component_kind"] == kind)
                for kind in ("numeric_value", "missing_indicator", "category_level")
            },
            "identifier_target_split_excluded": True,
        },
        "validation_replay": {
            "max_abs_probability_difference": score_difference,
            "max_abs_additivity_error": additivity_error,
            "max_abs_probability_reconstruction_error": probability_error,
            "metrics": metrics,
            "stage5_metrics_matched": True,
        },
        "global_explanation": {
            "explained_rows": len(labels),
            "base_value_raw_log_odds": float(np.mean(base_values)),
            "base_value_probability": float(expit(np.mean(base_values))),
            "top_source_features": importance,
            "source_group_importance": _source_group_importance(
                mean_abs, features, groups
            ),
            "numeric_direction": direction,
            "top_transformed_components": transformed_importance,
            "aggregation_method": (
                "sum transformed component SHAP values per original source feature, "
                "then mean absolute aggregated contribution"
            ),
            "causal_interpretation_allowed": False,
        },
        "error_analysis": error,
        "local_artifact": local_metadata,
        "figures": figures,
        "resources": {
            "total_seconds": round(time.perf_counter() - started, 3),
            "process_peak_rss_mb": _peak_rss_mb(),
            "shap_matrix_shape": [len(labels), len(transformed_names)],
            "source_aggregated_matrix_shape": [len(labels), len(features)],
        },
        "stage6_next": {
            "part": "2/3",
            "work": "risk_bands_top_k_scenarios_and_subgroup_diagnostics",
            "model_improvement_decision_pending": True,
            "test_remains_sealed": True,
        },
    }
    report_text = render_markdown_report(
        payload,
        report_path=report,
        global_figure_path=global_figure,
        direction_figure_path=direction_figure,
        error_figure_path=error_figure,
    )
    _atomic_write_text(report, report_text)
    _atomic_write_json(output, payload)
    if progress is not None:
        progress(
            f"Stage 6 1/3 완료: validation {len(labels):,}명 SHAP, "
            f"test 0행 사용"
        )
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 6 1/3 고정 LightGBM의 SHAP·오류 분석을 실행합니다."
    )
    parser.add_argument("--stage5-result", type=Path, default=DEFAULT_STAGE5_RESULT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--lock-manifest", type=Path, default=DEFAULT_LOCK_MANIFEST)
    parser.add_argument(
        "--validation-scores", type=Path, default=DEFAULT_VALIDATION_SCORES
    )
    parser.add_argument("--local-artifact", type=Path, default=DEFAULT_LOCAL_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--global-figure", type=Path, default=DEFAULT_GLOBAL_FIGURE)
    parser.add_argument(
        "--direction-figure", type=Path, default=DEFAULT_DIRECTION_FIGURE
    )
    parser.add_argument("--error-figure", type=Path, default=DEFAULT_ERROR_FIGURE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_stage6_shap_analysis(
        stage5_result_path=args.stage5_result,
        model_path=args.model,
        calibrator_path=args.calibrator,
        lock_manifest_path=args.lock_manifest,
        validation_scores_path=args.validation_scores,
        local_artifact_path=args.local_artifact,
        output_path=args.output,
        report_path=args.report,
        global_figure_path=args.global_figure,
        direction_figure_path=args.direction_figure,
        error_figure_path=args.error_figure,
        progress=print,
    )
    print(
        "Stage 6 1/3 완료: "
        f"validation {result['data_scope']['validation_rows']:,}명, "
        f"test {result['data_scope']['test_feature_rows_used']}행 사용"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
