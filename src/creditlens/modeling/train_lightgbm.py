"""Stage 5 1/3 고정 설정 LightGBM을 V1·V2·V3에서 비교한다.

세 데이터 버전에 같은 모델 설정과 같은 평가 규칙을 적용한다. 모델과 전처리는
train으로만 학습하고 validation은 고정 설정의 비교에만 사용한다. test 피처,
예측과 평가는 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import duckdb
import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.data import DevelopmentDataset, load_development_data
from creditlens.modeling.feature_roles import MartVersion
from creditlens.modeling.preprocessing import (
    make_preprocessor,
    transformed_feature_names,
)


SCHEMA_VERSION = "1.0"
RUN_VERSION = "stage5-lightgbm-baseline-v1"
STAGE_PART = "1/3"
RANDOM_SEED = 42
TOP_FRACTION = 0.1
CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_OUTPUT = Path("reports/stage5_lightgbm_results.json")
DEFAULT_REPORT = Path("docs/Stage5_LightGBM_Report.md")
DEFAULT_ARTIFACT_DIR = Path("models/stage5")
DEFAULT_BASELINE_REFERENCE = Path("reports/stage4_baseline_results.json")

REQUIRED_BASELINE_KEYS = (
    "logistic_v1",
    "logistic_v2",
    "logistic_v3",
    "random_forest_v3",
)


class LightGBMTrainingError(RuntimeError):
    """Stage 5 LightGBM 실행 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    display_name: str
    version: MartVersion


@dataclass(frozen=True)
class DevelopmentAlignment:
    """V1·V2·V3에서 동일해야 하는 고객 순서와 TARGET."""

    train_ids: pd.Index
    train_y: np.ndarray
    validation_ids: pd.Index
    validation_y: np.ndarray


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("lightgbm_v1", "V1 LightGBM", "v1"),
    ExperimentSpec("lightgbm_v2", "V2 LightGBM", "v2"),
    ExperimentSpec("lightgbm_v3", "V3 LightGBM", "v3"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _peak_rss_mb() -> float | None:
    try:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ValueError):
        return None
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return round(peak / divisor, 3)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    _atomic_write_text(path, serialized + "\n")


def _git_ignored(path: Path) -> bool | None:
    """저장소 내부 경로가 미추적 상태이며 실제 ignore되는지 반환한다."""

    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        return None
    root = Path(root_result.stdout.strip()).resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return None

    tracked_result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked_result.returncode == 0:
        raise LightGBMTrainingError(
            f"모델·행별 점수 경로가 이미 Git에 추적되고 있습니다: {relative}"
        )
    if tracked_result.returncode != 1:
        raise LightGBMTrainingError("모델 산출물의 Git 추적 상태를 확인하지 못했습니다.")

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise LightGBMTrainingError("모델 산출물의 Git ignore 상태를 확인하지 못했습니다.")


def _atomic_joblib_dump(value: Any, path: Path) -> dict[str, Any]:
    ignored = _git_ignored(path)
    if ignored is False:
        raise LightGBMTrainingError(
            f"모델·행별 점수 경로는 Git에서 제외되어야 합니다: {_display_path(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".joblib.tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        joblib.dump(value, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "display_path": _display_path(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "git_ignored": ignored,
    }


def _validate_output_paths(
    output: Path,
    report: Path,
    artifacts: Path,
    baseline_reference: Path,
) -> None:
    if output.suffix.lower() != ".json":
        raise LightGBMTrainingError("집계 결과 경로는 .json이어야 합니다.")
    if report.suffix.lower() != ".md":
        raise LightGBMTrainingError("보고서 경로는 .md여야 합니다.")
    resolved = (
        output.resolve(),
        report.resolve(),
        artifacts.resolve(),
        baseline_reference.resolve(),
    )
    if len(set(resolved)) != len(resolved):
        raise LightGBMTrainingError(
            "결과·보고서·모델·Stage 4 기준 경로는 서로 달라야 합니다."
        )
    for shared_path in (output.resolve(), report.resolve()):
        try:
            shared_path.relative_to(artifacts.resolve())
        except ValueError:
            continue
        raise LightGBMTrainingError(
            "공유 JSON·Markdown은 Git 제외 모델 경로 아래에 둘 수 없습니다."
        )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _load_baseline_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LightGBMTrainingError(f"Stage 4 결과를 찾을 수 없습니다: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise LightGBMTrainingError("Stage 4 결과 JSON을 읽을 수 없습니다.") from error
    if not isinstance(payload, dict) or payload.get("run_status") != "complete":
        raise LightGBMTrainingError("완료된 Stage 4 기준 결과가 필요합니다.")

    scope = payload.get("data_scope")
    if not isinstance(scope, dict):
        raise LightGBMTrainingError("Stage 4 데이터 사용 감사 정보가 없습니다.")
    if (
        scope.get("test_feature_rows_used") != 0
        or scope.get("test_predictions_created") is not False
        or scope.get("customer_ids_in_shared_outputs") is not False
    ):
        raise LightGBMTrainingError("Stage 4 결과의 test·고객 ID 봉인 계약이 다릅니다.")

    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise LightGBMTrainingError("Stage 4 평가 설정이 없습니다.")
    expected_settings = {
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "top_fraction": TOP_FRACTION,
    }
    for key, expected in expected_settings.items():
        value = settings.get(key)
        if not _finite_number(value) or not np.isclose(
            float(value), expected, rtol=0.0, atol=1e-15
        ):
            raise LightGBMTrainingError(f"Stage 4 {key} 설정이 현재 평가와 다릅니다.")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise LightGBMTrainingError("Stage 4 실험 목록이 없습니다.")
    by_key = {
        item.get("key"): item
        for item in experiments
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }
    missing = [key for key in REQUIRED_BASELINE_KEYS if key not in by_key]
    if missing:
        raise LightGBMTrainingError(f"Stage 4 비교 모델이 없습니다: {missing}")
    expected_versions = {
        "logistic_v1": "v1",
        "logistic_v2": "v2",
        "logistic_v3": "v3",
        "random_forest_v3": "v3",
    }
    for key in REQUIRED_BASELINE_KEYS:
        if by_key[key].get("data_version") != expected_versions[key]:
            raise LightGBMTrainingError(f"Stage 4 {key} 데이터 버전이 다릅니다.")
        metrics = by_key[key].get("metrics")
        if not isinstance(metrics, dict):
            raise LightGBMTrainingError(f"Stage 4 {key} 평가 지표가 없습니다.")
        for metric in ("roc_auc", "pr_auc", "ks", "gini", "brier_score"):
            if not _finite_number(metrics.get(metric)):
                raise LightGBMTrainingError(
                    f"Stage 4 {key}.{metric}이 유한한 숫자가 아닙니다."
                )
        threshold_metrics = metrics.get("threshold_metrics")
        top_k_metrics = metrics.get("top_k_metrics")
        threshold_value = (
            threshold_metrics.get("threshold")
            if isinstance(threshold_metrics, dict)
            else None
        )
        if not _finite_number(threshold_value) or not np.isclose(
            float(threshold_value),
            CLASSIFICATION_THRESHOLD,
            rtol=0.0,
            atol=1e-15,
        ):
            raise LightGBMTrainingError(f"Stage 4 {key} threshold 지표가 다릅니다.")
        top_fraction_value = (
            top_k_metrics.get("requested_fraction")
            if isinstance(top_k_metrics, dict)
            else None
        )
        if not _finite_number(top_fraction_value) or not np.isclose(
            float(top_fraction_value),
            TOP_FRACTION,
            rtol=0.0,
            atol=1e-15,
        ):
            raise LightGBMTrainingError(f"Stage 4 {key} Top-K 비율이 다릅니다.")
        for metric in ("recall", "lift"):
            if not _finite_number(top_k_metrics.get(metric)):
                raise LightGBMTrainingError(
                    f"Stage 4 {key} Top-K {metric}이 유한한 숫자가 아닙니다."
                )
    versions = payload.get("data_versions")
    if not isinstance(versions, dict) or any(
        not isinstance(versions.get(version), dict) for version in ("v1", "v2", "v3")
    ):
        raise LightGBMTrainingError("Stage 4 데이터 버전 계약이 없습니다.")
    return payload


def _baseline_by_key(reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["key"]: item for item in reference["experiments"]}


def _baseline_snapshot(reference: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = _baseline_by_key(reference)
    return [
        {
            "key": key,
            "display_name": by_key[key]["display_name"],
            "data_version": by_key[key]["data_version"],
            "metrics": by_key[key]["metrics"],
        }
        for key in REQUIRED_BASELINE_KEYS
    ]


def _validate_dataset(dataset: DevelopmentDataset) -> None:
    audit = dataset.audit
    if audit.get("test_sealed") is not True:
        raise LightGBMTrainingError("test 봉인 상태가 확인되지 않았습니다.")
    if audit.get("test_feature_rows_used") != 0:
        raise LightGBMTrainingError("LightGBM 실행에서 test 피처가 사용되었습니다.")
    loaded = audit.get("feature_rows_loaded", {})
    if loaded.get("test") != 0:
        raise LightGBMTrainingError("개발 로더가 test 피처 행을 적재했습니다.")
    if hasattr(dataset, "test"):
        raise LightGBMTrainingError("개발 데이터 객체에 test split이 노출되었습니다.")


def _validate_baseline_dataset(
    reference: dict[str, Any],
    dataset: DevelopmentDataset,
) -> None:
    expected_scope = reference["data_scope"]
    expected = reference["data_versions"][dataset.version]
    actual_values = {
        "schema_sha256": dataset.schema_sha256,
        "parquet_sha256": dataset.parquet_sha256,
        "model_feature_columns": len(dataset.roles.model_features),
        "numeric_feature_columns": len(dataset.roles.numeric),
        "categorical_feature_columns": len(dataset.roles.categorical),
        "train_rows": len(dataset.train.y),
        "validation_rows": len(dataset.validation.y),
    }
    for key, actual in actual_values.items():
        if expected.get(key) != actual:
            raise LightGBMTrainingError(
                f"현재 {dataset.version}.{key}가 Stage 4 기준과 다릅니다."
            )
    if len(dataset.train.y) != expected_scope.get("train_rows"):
        raise LightGBMTrainingError("현재 train 행 수가 Stage 4와 다릅니다.")
    if len(dataset.validation.y) != expected_scope.get("validation_rows"):
        raise LightGBMTrainingError("현재 validation 행 수가 Stage 4와 다릅니다.")
    expected_rate = expected_scope.get("validation_positive_rate")
    actual_rate = float(dataset.validation.y.mean())
    if not isinstance(expected_rate, (int, float)) or not np.isclose(
        actual_rate, float(expected_rate), rtol=0.0, atol=1e-15
    ):
        raise LightGBMTrainingError("현재 validation 양성률이 Stage 4와 다릅니다.")


def _capture_alignment(dataset: DevelopmentDataset) -> DevelopmentAlignment:
    return DevelopmentAlignment(
        train_ids=dataset.train.customer_ids.copy(),
        train_y=dataset.train.y.to_numpy(dtype=np.int8, copy=True),
        validation_ids=dataset.validation.customer_ids.copy(),
        validation_y=dataset.validation.y.to_numpy(dtype=np.int8, copy=True),
    )


def _validate_alignment(
    reference: DevelopmentAlignment,
    dataset: DevelopmentDataset,
) -> None:
    current = _capture_alignment(dataset)
    if not reference.train_ids.equals(current.train_ids):
        raise LightGBMTrainingError("데이터 버전별 train 고객 순서가 다릅니다.")
    if not np.array_equal(reference.train_y, current.train_y):
        raise LightGBMTrainingError("데이터 버전별 train TARGET이 다릅니다.")
    if not reference.validation_ids.equals(current.validation_ids):
        raise LightGBMTrainingError("데이터 버전별 validation 고객 순서가 다릅니다.")
    if not np.array_equal(reference.validation_y, current.validation_y):
        raise LightGBMTrainingError("데이터 버전별 validation TARGET이 다릅니다.")


def _lightgbm_settings(n_jobs: int) -> dict[str, Any]:
    """V1·V2·V3에 동일하게 적용할 사전 고정 기준 설정."""

    return {
        "boosting_type": "gbdt",
        "objective": "binary",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "min_split_gain": 0.0,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "class_weight": None,
        "random_state": RANDOM_SEED,
        "n_jobs": n_jobs,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _positive_scores(model: BaseEstimator, X: Any) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise LightGBMTrainingError("LightGBM 파이프라인에 predict_proba가 없습니다.")
    classes = np.asarray(getattr(model, "classes_", []))
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1:
        raise LightGBMTrainingError("학습 모델의 양성 클래스 1을 찾을 수 없습니다.")
    probabilities = model.predict_proba(X)  # type: ignore[attr-defined]
    scores = np.asarray(probabilities[:, int(positive_indices[0])], dtype=np.float64)
    if (
        scores.ndim != 1
        or not np.isfinite(scores).all()
        or ((scores < 0.0) | (scores > 1.0)).any()
    ):
        raise LightGBMTrainingError("validation 예측이 [0, 1] 유한 확률이 아닙니다.")
    return scores


def _fit_experiment(
    spec: ExperimentSpec,
    dataset: DevelopmentDataset,
    *,
    artifact_dir: Path,
    lightgbm_jobs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    settings = _lightgbm_settings(lightgbm_jobs)
    estimator = LGBMClassifier(**settings)
    fitted = Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(dataset.roles, model_family="tree"),
            ),
            ("model", estimator),
        ]
    )

    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fitted.fit(dataset.train.X, dataset.train.y)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    scores = _positive_scores(fitted, dataset.validation.X)
    predict_seconds = time.perf_counter() - predict_started

    metrics = evaluate_binary_metrics(
        dataset.validation.y.to_numpy(),
        scores,
        threshold=CLASSIFICATION_THRESHOLD,
        top_fraction=TOP_FRACTION,
    )
    transformed_features = len(
        transformed_feature_names(fitted.named_steps["preprocessor"])
    )
    fitted_model = fitted.named_steps["model"]
    best_iteration_raw = getattr(fitted_model, "best_iteration_", 0)
    best_iteration = int(best_iteration_raw) if best_iteration_raw else None
    n_estimators_fitted = int(
        getattr(fitted_model, "n_estimators_", settings["n_estimators"])
    )
    artifact = _atomic_joblib_dump(
        fitted,
        artifact_dir / f"{spec.key}.joblib",
    )
    return (
        {
            "key": spec.key,
            "display_name": spec.display_name,
            "data_version": spec.version,
            "estimator": "lightgbm",
            "model_family": "tree",
            "settings": settings,
            "train_rows": len(dataset.train.y),
            "validation_rows": len(dataset.validation.y),
            "input_feature_columns": len(dataset.roles.model_features),
            "transformed_feature_columns": transformed_features,
            "fit_seconds": round(fit_seconds, 3),
            "validation_predict_seconds": round(predict_seconds, 3),
            "iteration_details": {
                "n_estimators_requested": settings["n_estimators"],
                "n_estimators_fitted": n_estimators_fitted,
                "best_iteration": best_iteration,
                "early_stopping_used": False,
            },
            "warnings": [
                f"{item.category.__name__}: {item.message}" for item in captured
            ],
            "metrics": metrics,
            "model_artifact": artifact,
        },
        scores.astype(np.float32),
    )


def _metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    if metric == "recall_at_10pct":
        value = metrics["top_k_metrics"].get("recall")
    elif metric == "lift_at_10pct":
        value = metrics["top_k_metrics"].get("lift")
    else:
        value = metrics.get(metric)
    return None if value is None else float(value)


def _metric_deltas(
    newer: dict[str, Any],
    older: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, float | None] = {}
    for metric in (
        "roc_auc",
        "pr_auc",
        "ks",
        "gini",
        "brier_score",
        "recall_at_10pct",
        "lift_at_10pct",
    ):
        newer_value = _metric_value(newer["metrics"], metric)
        older_value = _metric_value(older["metrics"], metric)
        result[metric] = (
            None
            if newer_value is None or older_value is None
            else newer_value - older_value
        )
    return result


def _build_comparisons(
    experiments: Sequence[dict[str, Any]],
    baseline_reference: dict[str, Any],
) -> dict[str, Any]:
    lightgbm_by_key = {item["key"]: item for item in experiments}
    baseline_by_key = _baseline_by_key(baseline_reference)
    random_forest_delta = _metric_deltas(
        lightgbm_by_key["lightgbm_v3"],
        baseline_by_key["random_forest_v3"],
    )
    random_forest_delta["brier_score"] = None
    random_forest_delta["brier_score_note"] = (
        "not_comparable_random_forest_used_balanced_subsample"
    )
    return {
        "data_value_same_lightgbm": {
            "v2_minus_v1": _metric_deltas(
                lightgbm_by_key["lightgbm_v2"], lightgbm_by_key["lightgbm_v1"]
            ),
            "v3_minus_v2": _metric_deltas(
                lightgbm_by_key["lightgbm_v3"], lightgbm_by_key["lightgbm_v2"]
            ),
            "v3_minus_v1": _metric_deltas(
                lightgbm_by_key["lightgbm_v3"], lightgbm_by_key["lightgbm_v1"]
            ),
        },
        "model_effect_same_data": {
            "lightgbm_v1_minus_logistic_v1": _metric_deltas(
                lightgbm_by_key["lightgbm_v1"], baseline_by_key["logistic_v1"]
            ),
            "lightgbm_v2_minus_logistic_v2": _metric_deltas(
                lightgbm_by_key["lightgbm_v2"], baseline_by_key["logistic_v2"]
            ),
            "lightgbm_v3_minus_logistic_v3": _metric_deltas(
                lightgbm_by_key["lightgbm_v3"], baseline_by_key["logistic_v3"]
            ),
            "lightgbm_v3_minus_random_forest_v3": random_forest_delta,
        },
    }


def _format_metric(value: Any, digits: int = 4, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    prefix = "+" if signed and float(value) >= 0 else ""
    return f"{prefix}{float(value):.{digits}f}"


def render_markdown_report(payload: dict[str, Any]) -> str:
    """집계 결과를 사람이 읽을 수 있는 한국어 Markdown으로 변환한다."""

    baseline_by_key = {
        item["key"]: item for item in payload["baseline_reference"]["experiments"]
    }
    lines = [
        "# Stage 5 1/3 LightGBM 비교 보고서",
        "",
        "> 같은 고정 설정의 LightGBM을 V1·V2·V3 train으로 학습하고 같은 validation에서 평가한 결과입니다. test는 사용하지 않았습니다.",
        "",
        "## 왜 이 실험을 했는가",
        "",
        "LightGBM은 표 형태 데이터의 비선형 관계와 변수 사이 상호작용을 학습하는 트리 기반 모델입니다. 같은 LightGBM에 아래 데이터를 차례로 추가해 모델 변화와 데이터 추가 효과를 분리했습니다.",
        "",
        "- V1: 대출 신청정보",
        "- V2: V1 + 외부 신용이력",
        "- V3: V2 + 과거 납부이력",
        "",
        "이번 결과는 튜닝 전 기준 성능입니다. 세 버전 모두 같은 설정을 사용했고 validation을 학습, early stopping 또는 설정 선택에 사용하지 않았습니다.",
        "",
        "## LightGBM validation 결과",
        "",
        "| 데이터 | ROC-AUC | PR-AUC(AP) | KS | Gini | Brier | Recall@10% | Lift@10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["experiments"]:
        metrics = result["metrics"]
        top_k = metrics["top_k_metrics"]
        lines.append(
            "| {version} | {roc} | {pr} | {ks} | {gini} | {brier} | {recall} | {lift} |".format(
                version=result["data_version"].upper(),
                roc=_format_metric(metrics["roc_auc"]),
                pr=_format_metric(metrics["pr_auc"]),
                ks=_format_metric(metrics["ks"]),
                gini=_format_metric(metrics["gini"]),
                brier=_format_metric(metrics["brier_score"]),
                recall=_format_metric(top_k["recall"]),
                lift=_format_metric(top_k["lift"]),
            )
        )

    data_deltas = payload["comparisons"]["data_value_same_lightgbm"]
    lines.extend(
        [
            "",
            "PR-AUC의 무작위 기준은 validation의 상환곤란 비율입니다. Brier Score는 낮을수록 좋고 나머지 표의 주요 지표는 높을수록 좋습니다.",
            "",
            "## 같은 LightGBM에서 데이터 추가 효과",
            "",
            "| 비교 | Δ ROC-AUC | Δ PR-AUC | Δ KS | Δ Brier | Δ Recall@10% |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("V2 - V1: 외부 신용이력 추가", "v2_minus_v1"),
        ("V3 - V2: 납부이력 추가", "v3_minus_v2"),
        ("V3 - V1: 두 정보 원천 추가", "v3_minus_v1"),
    ):
        delta = data_deltas[key]
        lines.append(
            "| {label} | {roc} | {pr} | {ks} | {brier} | {recall} |".format(
                label=label,
                roc=_format_metric(delta["roc_auc"], signed=True),
                pr=_format_metric(delta["pr_auc"], signed=True),
                ks=_format_metric(delta["ks"], signed=True),
                brier=_format_metric(delta["brier_score"], signed=True),
                recall=_format_metric(delta["recall_at_10pct"], signed=True),
            )
        )

    model_deltas = payload["comparisons"]["model_effect_same_data"]
    lines.extend(
        [
            "",
            "## 기존 모델과의 같은 데이터 비교",
            "",
            "| 비교 | Δ ROC-AUC | Δ PR-AUC | Δ KS | Δ Brier | Δ Recall@10% |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("V1 LightGBM - V1 Logistic", "lightgbm_v1_minus_logistic_v1"),
        ("V2 LightGBM - V2 Logistic", "lightgbm_v2_minus_logistic_v2"),
        ("V3 LightGBM - V3 Logistic", "lightgbm_v3_minus_logistic_v3"),
        ("V3 LightGBM - V3 Random Forest", "lightgbm_v3_minus_random_forest_v3"),
    ):
        delta = model_deltas[key]
        brier = (
            "비교 불가"
            if key == "lightgbm_v3_minus_random_forest_v3"
            else _format_metric(delta["brier_score"], signed=True)
        )
        lines.append(
            "| {label} | {roc} | {pr} | {ks} | {brier} | {recall} |".format(
                label=label,
                roc=_format_metric(delta["roc_auc"], signed=True),
                pr=_format_metric(delta["pr_auc"], signed=True),
                ks=_format_metric(delta["ks"], signed=True),
                brier=brier,
                recall=_format_metric(delta["recall_at_10pct"], signed=True),
            )
        )

    best = max(payload["experiments"], key=lambda item: item["metrics"]["pr_auc"])
    v3 = next(item for item in payload["experiments"] if item["key"] == "lightgbm_v3")
    v3_logistic = baseline_by_key["logistic_v3"]
    lines.extend(
        [
            "",
            "V3 Random Forest는 `balanced_subsample` 가중치를 사용해 현재 점수를 보정된 실제 확률로 해석할 수 없습니다. 따라서 LightGBM과 Random Forest의 Brier 차이는 확률 품질의 공정 비교에서 제외했습니다.",
            "",
            "## 현재 해석",
            "",
            f"- LightGBM 세 버전 중 validation PR-AUC가 가장 높은 데이터는 {best['data_version'].upper()}이며 값은 {_format_metric(best['metrics']['pr_auc'])}입니다.",
            f"- V3 LightGBM의 ROC-AUC/PR-AUC는 {_format_metric(v3['metrics']['roc_auc'])}/{_format_metric(v3['metrics']['pr_auc'])}, V3 Logistic은 {_format_metric(v3_logistic['metrics']['roc_auc'])}/{_format_metric(v3_logistic['metrics']['pr_auc'])}입니다.",
            "- 이것은 고정 설정의 validation 비교 결과이지 최종 모델 선정이 아닙니다. TensorFlow MLP와 train 내부 개선 실험이 남아 있습니다.",
            "- validation 결과를 보고 LightGBM 설정을 다시 바꾸지 않았습니다. 제한 튜닝은 Stage 5 3/3에서 train 내부 데이터로만 수행합니다.",
            "",
            "## 고정한 조건과 데이터 감사",
            "",
            f"- LightGBM: {payload['environment']['lightgbm']}",
            f"- 트리 수: {payload['settings']['lightgbm']['n_estimators']}",
            f"- learning rate: {payload['settings']['lightgbm']['learning_rate']}",
            f"- 재현 seed: {payload['settings']['random_seed']}",
            "- 불균형 가중치: 사용하지 않음",
            "- early stopping: 사용하지 않음",
            "- 전처리: train에서만 median 대치·결측 플래그·희소범주·원-핫 인코딩 학습",
            f"- train 고객: {payload['data_scope']['train_rows']:,}명",
            f"- validation 고객: {payload['data_scope']['validation_rows']:,}명",
            f"- test 피처 사용: {payload['data_scope']['test_feature_rows_used']}행",
            "- test 예측·평가: 없음",
            "- 공유 JSON·Markdown에 고객 ID와 행별 예측값을 저장하지 않음",
            "",
            "## 다음 작업",
            "",
            "Stage 5 2/3에서 V3 TensorFlow MLP를 train 내부 early stopping으로 학습합니다. Stage 5 3/3에서 train 내부 제한 튜닝·확률 보정·피처군 분석 후 후보를 비교합니다. test는 계속 봉인합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def run_lightgbm_experiments(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    baseline_reference_path: str | Path = DEFAULT_BASELINE_REFERENCE,
    lightgbm_jobs: int = 2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """고정 설정 LightGBM 세 실험을 순차 실행하고 집계 결과를 저장한다."""

    if lightgbm_jobs < 1 or lightgbm_jobs > 4:
        raise LightGBMTrainingError("LightGBM n_jobs는 1~4 범위여야 합니다.")
    output = Path(output_path)
    report = Path(report_path)
    artifacts = Path(artifact_dir)
    baseline_path = Path(baseline_reference_path)
    _validate_output_paths(output, report, artifacts, baseline_path)
    baseline_reference = _load_baseline_reference(baseline_path)
    settings = _lightgbm_settings(lightgbm_jobs)
    started = time.perf_counter()

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "stage_part": STAGE_PART,
        "run_status": "in_progress",
        "generated_at_utc": _utc_now(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "duckdb": duckdb.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "random_seed": RANDOM_SEED,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "top_fraction": TOP_FRACTION,
            "lightgbm": settings,
            "experiment_order": [spec.key for spec in EXPERIMENTS],
            "configuration_locked_before_validation": True,
            "validation_used_for_fit": False,
            "validation_used_for_early_stopping": False,
            "validation_result_driven_retuning": False,
        },
        "data_scope": {
            "train_rows": 0,
            "validation_rows": 0,
            "validation_positive_rate": None,
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
            "row_level_predictions_in_shared_outputs": False,
        },
        "baseline_reference": {
            "display_path": _display_path(baseline_path),
            "sha256": _sha256(baseline_path),
            "run_version": baseline_reference.get("run_version"),
            "experiments": _baseline_snapshot(baseline_reference),
        },
        "data_versions": {},
        "experiments": [],
        "comparisons": {},
        "local_prediction_artifact": None,
        "resources": {},
    }
    alignment: DevelopmentAlignment | None = None
    prediction_scores: dict[str, np.ndarray] = {}

    for spec in EXPERIMENTS:
        if progress is not None:
            progress(f"{spec.version.upper()} train·validation 로딩 및 계약 검증")
        dataset = load_development_data(spec.version)
        _validate_dataset(dataset)
        _validate_baseline_dataset(baseline_reference, dataset)
        if alignment is None:
            alignment = _capture_alignment(dataset)
            payload["data_scope"].update(
                {
                    "train_rows": len(dataset.train.y),
                    "validation_rows": len(dataset.validation.y),
                    "validation_positive_rate": float(dataset.validation.y.mean()),
                }
            )
        else:
            _validate_alignment(alignment, dataset)

        payload["data_versions"][spec.version] = {
            "schema_sha256": dataset.schema_sha256,
            "parquet_sha256": dataset.parquet_sha256,
            "model_feature_columns": len(dataset.roles.model_features),
            "numeric_feature_columns": len(dataset.roles.numeric),
            "categorical_feature_columns": len(dataset.roles.categorical),
            "train_rows": len(dataset.train.y),
            "validation_rows": len(dataset.validation.y),
            "test_feature_rows_used": dataset.audit["test_feature_rows_used"],
            "stage3_summary_verified": dataset.audit["stage3_summary_verified"],
            "parquet_sha256_verified": dataset.audit["parquet_sha256_verified"],
            "stage4_reference_verified": True,
        }

        if progress is not None:
            progress(f"학습 시작: {spec.display_name}")
        result, scores = _fit_experiment(
            spec,
            dataset,
            artifact_dir=artifacts,
            lightgbm_jobs=lightgbm_jobs,
        )
        payload["experiments"].append(result)
        prediction_scores[spec.key] = scores
        if alignment is None:
            raise LightGBMTrainingError("validation 정답이 준비되지 않았습니다.")
        prediction_payload = {
            "schema_version": SCHEMA_VERSION,
            "run_version": RUN_VERSION,
            "split": "validation",
            "customer_ids_included": False,
            "y_true": alignment.validation_y,
            "scores": prediction_scores,
        }
        payload["local_prediction_artifact"] = _atomic_joblib_dump(
            prediction_payload,
            artifacts / "lightgbm_validation_scores.joblib",
        )
        _atomic_write_json(output, payload)
        if progress is not None:
            metrics = result["metrics"]
            progress(
                f"학습 완료: {spec.display_name} "
                f"(ROC-AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}, "
                f"fit={result['fit_seconds']:.1f}s)"
            )
        del dataset
        gc.collect()

    expected_keys = [spec.key for spec in EXPERIMENTS]
    actual_keys = [item["key"] for item in payload["experiments"]]
    if actual_keys != expected_keys:
        raise LightGBMTrainingError(
            f"LightGBM 실험 실행 순서가 계약과 다릅니다: {actual_keys}"
        )
    payload["comparisons"] = _build_comparisons(
        payload["experiments"], baseline_reference
    )
    payload["resources"] = {
        "total_seconds": round(time.perf_counter() - started, 3),
        "process_peak_rss_mb": _peak_rss_mb(),
        "measurement_scope": "current_process_lifetime",
    }
    payload["run_status"] = "complete"
    payload["completed_at_utc"] = _utc_now()
    _atomic_write_json(output, payload)
    _atomic_write_text(report, render_markdown_report(payload))
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5 1/3 고정 설정 LightGBM V1·V2·V3 비교를 실행합니다."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        default=DEFAULT_BASELINE_REFERENCE,
    )
    parser.add_argument("--lightgbm-jobs", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_lightgbm_experiments(
        output_path=args.output,
        report_path=args.report,
        artifact_dir=args.artifact_dir,
        baseline_reference_path=args.baseline_reference,
        lightgbm_jobs=args.lightgbm_jobs,
        progress=lambda message: print(message, flush=True),
    )
    print(
        "Stage 5 1/3 LightGBM 완료: "
        f"{len(result['experiments'])}개 실험, "
        f"test 피처 {result['data_scope']['test_feature_rows_used']}행 사용"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
