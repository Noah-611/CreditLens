"""Stage 4 기준 모델을 train에 적합하고 validation에서만 평가한다.

실험 결과 JSON에는 집계 지표와 재현 정보만 기록한다. 고객 ID와 행별 예측은
포함하지 않으며, 학습 파이프라인과 validation 점수는 Git에서 제외되는
``models/stage4`` 아래에만 저장한다.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import duckdb
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.data import DevelopmentDataset, load_development_data
from creditlens.modeling.feature_roles import MartVersion
from creditlens.modeling.preprocessing import (
    ModelFamily,
    make_preprocessor,
    transformed_feature_names,
)


SCHEMA_VERSION = "1.0"
RUN_VERSION = "stage4-baseline-v2"
RANDOM_SEED = 42
DEFAULT_OUTPUT = Path("reports/stage4_baseline_results.json")
DEFAULT_REPORT = Path("docs/Stage4_Baseline_Model_Report.md")
DEFAULT_ARTIFACT_DIR = Path("models/stage4")
TOP_FRACTION = 0.1
CLASSIFICATION_THRESHOLD = 0.5

EstimatorKind = Literal["dummy", "logistic", "random_forest"]


class BaselineTrainingError(RuntimeError):
    """기준 모델 실험 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    display_name: str
    version: MartVersion
    estimator_kind: EstimatorKind
    model_family: ModelFamily | None


@dataclass(frozen=True)
class DevelopmentAlignment:
    """데이터 버전 사이에서 같아야 하는 고객 순서와 정답."""

    train_ids: pd.Index
    train_y: np.ndarray
    validation_ids: pd.Index
    validation_y: np.ndarray


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("dummy_prior", "Dummy Prior", "v1", "dummy", None),
    ExperimentSpec(
        "logistic_v1", "V1 Logistic Regression", "v1", "logistic", "linear"
    ),
    ExperimentSpec(
        "logistic_v2", "V2 Logistic Regression", "v2", "logistic", "linear"
    ),
    ExperimentSpec(
        "logistic_v3", "V3 Logistic Regression", "v3", "logistic", "linear"
    ),
    ExperimentSpec(
        "random_forest_v3", "V3 Random Forest", "v3", "random_forest", "tree"
    ),
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


def _atomic_joblib_dump(value: Any, path: Path) -> dict[str, Any]:
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
        "git_ignored": True,
    }


def _make_estimator(
    spec: ExperimentSpec,
    *,
    random_forest_jobs: int,
) -> tuple[BaseEstimator, dict[str, Any]]:
    if spec.estimator_kind == "dummy":
        settings = {"strategy": "prior", "random_state": RANDOM_SEED}
        return DummyClassifier(**settings), settings
    if spec.estimator_kind == "logistic":
        settings = {
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 600,
            "tol": 0.0001,
            "class_weight": None,
        }
        return LogisticRegression(**settings), settings
    if spec.estimator_kind == "random_forest":
        settings = {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "max_samples": 0.7,
            "bootstrap": True,
            "class_weight": "balanced_subsample",
            "n_jobs": random_forest_jobs,
            "random_state": RANDOM_SEED,
        }
        return RandomForestClassifier(**settings), settings
    raise BaselineTrainingError(f"지원하지 않는 estimator입니다: {spec.estimator_kind}")


def _positive_scores(model: BaseEstimator, X: Any) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise BaselineTrainingError("기준 모델은 predict_proba를 제공해야 합니다.")
    classes = np.asarray(getattr(model, "classes_", []))
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1:
        raise BaselineTrainingError("학습 모델의 양성 클래스 1을 찾을 수 없습니다.")
    probabilities = model.predict_proba(X)  # type: ignore[attr-defined]
    scores = np.asarray(probabilities[:, int(positive_indices[0])], dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise BaselineTrainingError("validation 예측확률이 유한한 1차원 배열이 아닙니다.")
    return scores


def _validate_dataset(dataset: DevelopmentDataset) -> None:
    audit = dataset.audit
    if audit.get("test_sealed") is not True:
        raise BaselineTrainingError("test 봉인 상태가 확인되지 않았습니다.")
    if audit.get("test_feature_rows_used") != 0:
        raise BaselineTrainingError("기준 모델 실행에서 test 피처가 사용되었습니다.")
    loaded = audit.get("feature_rows_loaded", {})
    if loaded.get("test") != 0:
        raise BaselineTrainingError("기준 모델 로더가 test 피처 행을 적재했습니다.")
    if hasattr(dataset, "test"):
        raise BaselineTrainingError("개발 데이터 객체에 test split이 노출되었습니다.")


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
    """V1·V2·V3가 같은 개발 고객과 정답을 같은 순서로 쓰는지 검사한다."""

    current = _capture_alignment(dataset)
    if not reference.train_ids.equals(current.train_ids):
        raise BaselineTrainingError("데이터 버전별 train 고객 순서가 다릅니다.")
    if not np.array_equal(reference.train_y, current.train_y):
        raise BaselineTrainingError("데이터 버전별 train TARGET이 다릅니다.")
    if not reference.validation_ids.equals(current.validation_ids):
        raise BaselineTrainingError("데이터 버전별 validation 고객 순서가 다릅니다.")
    if not np.array_equal(reference.validation_y, current.validation_y):
        raise BaselineTrainingError("데이터 버전별 validation TARGET이 다릅니다.")


def _convergence_details(
    estimator: BaseEstimator,
    captured: Sequence[warnings.WarningMessage],
) -> dict[str, Any] | None:
    if not isinstance(estimator, LogisticRegression):
        return None
    n_iter = [int(value) for value in np.asarray(estimator.n_iter_).ravel()]
    messages = [
        str(item.message)
        for item in captured
        if issubclass(item.category, ConvergenceWarning)
    ]
    return {
        "n_iter": n_iter,
        "max_iter": int(estimator.max_iter),
        "converged": not messages and max(n_iter, default=0) < estimator.max_iter,
        "warnings": messages,
    }


def _fit_experiment(
    spec: ExperimentSpec,
    dataset: DevelopmentDataset,
    *,
    artifact_dir: Path,
    random_forest_jobs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    estimator, settings = _make_estimator(
        spec,
        random_forest_jobs=random_forest_jobs,
    )
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if spec.estimator_kind == "dummy":
            train_input = np.zeros((len(dataset.train.y), 1), dtype=np.float32)
            validation_input = np.zeros(
                (len(dataset.validation.y), 1), dtype=np.float32
            )
            estimator.fit(train_input, dataset.train.y)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            scores = _positive_scores(estimator, validation_input)
            predict_seconds = time.perf_counter() - predict_started
            fitted: BaseEstimator = estimator
            transformed_features = 0
        else:
            if spec.model_family is None:
                raise BaselineTrainingError("피처 모델에는 전처리 계열이 필요합니다.")
            fitted = Pipeline(
                [
                    (
                        "preprocessor",
                        make_preprocessor(
                            dataset.roles,
                            model_family=spec.model_family,
                        ),
                    ),
                    ("model", estimator),
                ]
            )
            fitted.fit(dataset.train.X, dataset.train.y)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            scores = _positive_scores(fitted, dataset.validation.X)
            predict_seconds = time.perf_counter() - predict_started
            transformed_features = len(
                transformed_feature_names(fitted.named_steps["preprocessor"])
            )

    metrics = evaluate_binary_metrics(
        dataset.validation.y.to_numpy(),
        scores,
        threshold=CLASSIFICATION_THRESHOLD,
        top_fraction=TOP_FRACTION,
    )
    model_path = artifact_dir / f"{spec.key}.joblib"
    artifact = _atomic_joblib_dump(fitted, model_path)
    convergence = _convergence_details(estimator, captured)
    warning_messages = [
        f"{item.category.__name__}: {item.message}"
        for item in captured
        if not issubclass(item.category, ConvergenceWarning)
    ]
    result = {
        "key": spec.key,
        "display_name": spec.display_name,
        "data_version": spec.version,
        "estimator": spec.estimator_kind,
        "model_family": spec.model_family,
        "settings": settings,
        "train_rows": len(dataset.train.y),
        "validation_rows": len(dataset.validation.y),
        "input_feature_columns": 0
        if spec.estimator_kind == "dummy"
        else len(dataset.roles.model_features),
        "transformed_feature_columns": transformed_features,
        "fit_seconds": round(fit_seconds, 3),
        "validation_predict_seconds": round(predict_seconds, 3),
        "convergence": convergence,
        "warnings": warning_messages,
        "metrics": metrics,
        "model_artifact": artifact,
    }
    return result, scores.astype(np.float32)


def _metric_delta(
    experiments: dict[str, dict[str, Any]],
    newer: str,
    older: str,
    metric: str,
) -> float | None:
    newer_value = experiments[newer]["metrics"].get(metric)
    older_value = experiments[older]["metrics"].get(metric)
    if newer_value is None or older_value is None:
        return None
    return float(newer_value) - float(older_value)


def _build_comparisons(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_key = {result["key"]: result for result in results}
    metrics = ("roc_auc", "pr_auc", "ks", "gini", "brier_score")

    def deltas(newer: str, older: str) -> dict[str, float | None]:
        return {
            metric: _metric_delta(by_key, newer, older, metric)
            for metric in metrics
        }

    return {
        "logistic_v2_minus_v1": deltas("logistic_v2", "logistic_v1"),
        "logistic_v3_minus_v2": deltas("logistic_v3", "logistic_v2"),
        "logistic_v3_minus_v1": deltas("logistic_v3", "logistic_v1"),
        "random_forest_v3_minus_logistic_v3": deltas(
            "random_forest_v3", "logistic_v3"
        ),
    }


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def render_markdown_report(payload: dict[str, Any]) -> str:
    """집계 결과를 사람이 읽기 쉬운 Markdown 보고서로 변환한다."""

    lines = [
        "# Stage 4 기준 모델 학습 보고서",
        "",
        "> train으로만 모델과 전처리를 학습하고 validation으로 비교한 실제 실행 결과입니다. test 피처·예측·평가는 사용하지 않았습니다.",
        "",
        "## 실험 목적",
        "",
        "- Dummy Prior로 아무 피처도 사용하지 않는 최저 기준을 확인합니다.",
        "- V1·V2·V3 Logistic Regression을 비교해 외부 신용이력과 납부이력의 추가 가치를 측정합니다.",
        "- V3 Random Forest를 제한된 복잡도의 비선형 기준 모델로 비교합니다.",
        "- 이번 단계에서는 하이퍼파라미터 탐색, cutoff 선택, 확률 보정과 test 평가를 수행하지 않습니다.",
        "",
        "## Validation 결과",
        "",
        "| 실험 | ROC-AUC | PR-AUC(AP) | KS | Gini | Brier | Recall@10% | Lift@10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["experiments"]:
        metrics = result["metrics"]
        top_k = metrics["top_k_metrics"]
        lines.append(
            "| {name} | {roc} | {pr} | {ks} | {gini} | {brier} | {recall} | {lift} |".format(
                name=result["display_name"],
                roc=_format_metric(metrics["roc_auc"]),
                pr=_format_metric(metrics["pr_auc"]),
                ks=_format_metric(metrics["ks"]),
                gini=_format_metric(metrics["gini"]),
                brier=_format_metric(metrics["brier_score"]),
                recall=_format_metric(top_k["recall"]),
                lift=_format_metric(top_k["lift"]),
            )
        )

    v2_delta = payload["comparisons"]["logistic_v2_minus_v1"]
    v3_delta = payload["comparisons"]["logistic_v3_minus_v2"]
    lines.extend(
        [
            "",
            "PR-AUC의 무작위 기준선은 validation의 TARGET=1 비율입니다. Gini는 `2 × ROC-AUC - 1`이며 독립 지표로 과장하지 않습니다.",
            "",
            "## 데이터 버전별 추가 가치",
            "",
            "| 비교 | Δ ROC-AUC | Δ PR-AUC | Δ KS | Δ Brier |",
            "|---|---:|---:|---:|---:|",
            "| V2 Logistic - V1 Logistic | {roc} | {pr} | {ks} | {brier} |".format(
                roc=_format_metric(v2_delta["roc_auc"]),
                pr=_format_metric(v2_delta["pr_auc"]),
                ks=_format_metric(v2_delta["ks"]),
                brier=_format_metric(v2_delta["brier_score"]),
            ),
            "| V3 Logistic - V2 Logistic | {roc} | {pr} | {ks} | {brier} |".format(
                roc=_format_metric(v3_delta["roc_auc"]),
                pr=_format_metric(v3_delta["pr_auc"]),
                ks=_format_metric(v3_delta["ks"]),
                brier=_format_metric(v3_delta["brier_score"]),
            ),
            "",
            "Brier Score는 낮을수록 좋으므로 음수 변화가 개선입니다. 위 차이는 동일한 Logistic Regression에서 데이터 원천만 추가한 결과입니다.",
            "",
            "## 고정한 학습 조건",
            "",
            f"- 재현 seed: `{payload['settings']['random_seed']}`",
            f"- 분류 임계값: `{payload['settings']['classification_threshold']}`",
            f"- Top-K 비율: `{payload['settings']['top_fraction']}`",
            "- Logistic Regression은 확률 기준선을 확인하기 위해 class weight 없이 학습했습니다.",
            "- Random Forest는 제한된 깊이와 트리 수, `balanced_subsample`을 사용했습니다.",
            "- 각 데이터 버전의 전처리기는 해당 train에만 fit했습니다.",
            "- 선형 수치 피처는 `StandardScaler(with_mean=False)`로 크기를 맞췄고 결측 indicator는 0/1을 유지했습니다.",
            "",
            "## 수렴과 실행 자원",
            "",
            "| 실험 | 반복 횟수 | 최대 반복 | 수렴 | 학습시간(초) |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for result in payload["experiments"]:
        convergence = result["convergence"]
        if convergence is None:
            iteration = "-"
            maximum = "-"
            converged = "해당 없음"
        else:
            iteration = str(max(convergence["n_iter"], default=0))
            maximum = str(convergence["max_iter"])
            converged = "예" if convergence["converged"] else "아니요"
        lines.append(
            f"| {result['display_name']} | {iteration} | {maximum} | "
            f"{converged} | {result['fit_seconds']:.3f} |"
        )

    peak_rss = payload["resources"]["process_peak_rss_mb"]
    peak_rss_text = "측정 불가" if peak_rss is None else f"{peak_rss:.3f} MiB"
    lines.extend(
        [
            "",
            f"- 전체 실행시간: {payload['resources']['total_seconds']:.3f}초",
            f"- 프로세스 최대 RSS: {peak_rss_text}",
            "",
            "초기 검증에서는 사분위 범위가 0인 희소 금액 피처가 `RobustScaler`에서 수천만 단위로 남아 V2·V3 Logistic 최적화를 방해했습니다. 모든 비상수 수치 피처를 표준편차 기준으로 맞추는 현재 방식으로 수정한 뒤 V1→V2→V3의 일관된 비교 결과를 얻었습니다.",
            "",
            "## 데이터 사용 감사",
            "",
            f"- train 고객: {payload['data_scope']['train_rows']:,}명",
            f"- validation 고객: {payload['data_scope']['validation_rows']:,}명",
            f"- test 피처 사용: {payload['data_scope']['test_feature_rows_used']}행",
            f"- test 예측·평가: {'없음' if payload['data_scope']['test_predictions_created'] is False else '있음'}",
            "- 고객 ID와 행별 예측값은 공유용 JSON·Markdown에 저장하지 않았습니다.",
            "",
            "## 해석 범위",
            "",
            "이 결과는 튜닝 전 기준 성능입니다. ROC·PR·Calibration 곡선, 위험도 decile, Top-K 상세 시나리오와 모델 선택 해석은 다음 분석 단계에서 작성합니다. Random Forest는 class weight를 사용했으므로 Brier Score를 보정된 확률 품질로 해석하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def run_baseline_experiments(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    random_forest_jobs: int = 2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """고정된 다섯 기준 실험을 순차 실행하고 집계 결과를 저장한다."""

    if random_forest_jobs < 1 or random_forest_jobs > 4:
        raise BaselineTrainingError("Random Forest n_jobs는 1~4 범위여야 합니다.")
    output = Path(output_path)
    report = Path(report_path)
    artifacts = Path(artifact_dir)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "run_status": "in_progress",
        "generated_at_utc": _utc_now(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "duckdb": duckdb.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "random_seed": RANDOM_SEED,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "top_fraction": TOP_FRACTION,
            "random_forest_jobs": random_forest_jobs,
            "experiment_order": [spec.key for spec in EXPERIMENTS],
        },
        "data_scope": {
            "train_rows": 0,
            "validation_rows": 0,
            "validation_positive_rate": None,
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
        },
        "data_versions": {},
        "experiments": [],
        "comparisons": {},
        "local_prediction_artifact": None,
        "resources": {},
    }
    alignment: DevelopmentAlignment | None = None
    prediction_scores: dict[str, np.ndarray] = {}

    for version in ("v1", "v2", "v3"):
        if progress is not None:
            progress(f"{version.upper()} train·validation 로딩 및 무결성 검증")
        dataset = load_development_data(version)
        _validate_dataset(dataset)
        if alignment is None:
            alignment = _capture_alignment(dataset)
            payload["data_scope"].update(
                {
                    "train_rows": len(dataset.train.y),
                    "validation_rows": len(dataset.validation.y),
                    "validation_positive_rate": float(
                        alignment.validation_y.mean()
                    ),
                }
            )
        else:
            _validate_alignment(alignment, dataset)

        payload["data_versions"][version] = {
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
        }

        for spec in (item for item in EXPERIMENTS if item.version == version):
            if progress is not None:
                progress(f"학습 시작: {spec.display_name}")
            result, scores = _fit_experiment(
                spec,
                dataset,
                artifact_dir=artifacts,
                random_forest_jobs=random_forest_jobs,
            )
            payload["experiments"].append(result)
            prediction_scores[spec.key] = scores
            if progress is not None:
                metrics = result["metrics"]
                progress(
                    f"학습 완료: {spec.display_name} "
                    f"(ROC-AUC={metrics['roc_auc']:.4f}, "
                    f"PR-AUC={metrics['pr_auc']:.4f}, "
                    f"fit={result['fit_seconds']:.1f}s)"
                )
            if alignment is None:
                raise BaselineTrainingError("validation 정답이 준비되지 않았습니다.")
            prediction_payload = {
                "schema_version": SCHEMA_VERSION,
                "split": "validation",
                "customer_ids_included": False,
                "y_true": alignment.validation_y,
                "scores": prediction_scores,
            }
            prediction_artifact = _atomic_joblib_dump(
                prediction_payload,
                artifacts / "validation_scores.joblib",
            )
            payload["local_prediction_artifact"] = prediction_artifact
            _atomic_write_json(output, payload)

        del dataset
        gc.collect()

    expected_keys = [spec.key for spec in EXPERIMENTS]
    actual_keys = [result["key"] for result in payload["experiments"]]
    if actual_keys != expected_keys:
        raise BaselineTrainingError(
            f"기준 실험 실행 순서가 계약과 다릅니다: {actual_keys}"
        )
    payload["comparisons"] = _build_comparisons(payload["experiments"])
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
        description="Stage 4 Dummy·Logistic·Random Forest 기준 모델을 학습합니다."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--random-forest-jobs", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_baseline_experiments(
        output_path=args.output,
        report_path=args.report,
        artifact_dir=args.artifact_dir,
        random_forest_jobs=args.random_forest_jobs,
        progress=lambda message: print(message, flush=True),
    )
    print(
        "Stage 4 기준 모델 완료: "
        f"{len(result['experiments'])}개 실험, "
        f"test 피처 {result['data_scope']['test_feature_rows_used']}행 사용"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
