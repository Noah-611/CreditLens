"""Stage 5 3/3 제한 튜닝·피처군 ablation·확률 보정을 수행한다.

공식 train 안에서만 LightGBM 설정과 확률 보정 방법을 선택한다. 공식
validation은 모든 설정과 로컬 모델 산출물을 고정한 뒤 한 번 예측하며, test
피처·예측·평가는 계속 봉인한다. 공유 JSON과 Markdown에는 집계 결과만 남긴다.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "creditlens-matplotlib"),
)

import duckdb
import joblib
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from creditlens.analysis.stage4_validation_analysis import build_calibration_summary
from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.calibration import (
    IdentityCalibrator,
    IsotonicScoreCalibrator,
    ScoreCalibrator,
    SigmoidLogitCalibrator,
    validate_score_labels as _validate_score_labels,
    validate_scores as _validate_scores,
)
from creditlens.modeling.data import (
    DEFAULT_MART_PATHS,
    ModelSplit,
    load_model_split,
)
from creditlens.modeling.feature_roles import (
    FeatureRoleError,
    FeatureRoles,
    MartVersion,
    resolve_feature_roles,
    schema_sha256,
)
from creditlens.modeling.preprocessing import (
    make_preprocessor,
    transformed_feature_names,
)
from creditlens.modeling.train_lightgbm import (
    _atomic_joblib_dump as _lightgbm_atomic_joblib_dump,
    _atomic_write_json,
    _atomic_write_text,
    _display_path,
    _finite_number,
    _git_ignored,
    _load_baseline_reference,
    _lightgbm_settings as _stage5_baseline_settings,
    _peak_rss_mb,
    _positive_scores,
    _sha256,
)


SCHEMA_VERSION = "1.0"
RUN_VERSION = "stage5-finalization-v1"
STAGE_PART = "3/3"
RANDOM_SEED = 42
CALIBRATION_SEED = 43
ABLATION_SEED = 44
TUNING_FOLDS = 3
CALIBRATION_FOLDS = 5
CLASSIFICATION_THRESHOLD = 0.5
TOP_FRACTION = 0.1
CALIBRATION_BINS = 10

MIN_PR_AUC_IMPROVEMENT = 0.001
MIN_WINNING_FOLDS = 2
MAX_ROC_AUC_DEGRADATION = 0.001
MAX_RECALL_AT_10PCT_DEGRADATION = 0.005
MIN_BRIER_IMPROVEMENT = 0.0001
MAX_LOG_LOSS_DEGRADATION = 0.0001
TUNING_PR_AUC_TIE_TOLERANCE = 0.0005
CALIBRATION_BRIER_TIE_TOLERANCE = 0.0001
MIN_CALIBRATION_WINNING_FOLDS = 3
SCORE_EPSILON = 1e-6

EXPECTED_GROUP_COUNTS = {
    "application": 132,
    "bureau": 37,
    "installments": 29,
}

DEFAULT_OUTPUT = Path("reports/stage5_final_results.json")
DEFAULT_REPORT = Path("docs/Stage5_Final_Model_Selection_Report.md")
DEFAULT_CALIBRATION_FIGURE = Path(
    "reports/figures/stage5_calibration_comparison.png"
)
DEFAULT_ABLATION_FIGURE = Path("reports/figures/stage5_feature_ablation.png")
DEFAULT_ARTIFACT_DIR = Path("models/stage5")
DEFAULT_STAGE4_REFERENCE = Path("reports/stage4_baseline_results.json")
DEFAULT_LIGHTGBM_REFERENCE = Path("reports/stage5_lightgbm_results.json")
DEFAULT_MLP_REFERENCE = Path("reports/stage5_mlp_results.json")


class Stage5FinalizationError(RuntimeError):
    """Stage 5 최종화 실행 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class TuningCandidate:
    key: str
    display_name: str
    settings: Mapping[str, Any]
    complexity_order: int


@dataclass(frozen=True)
class FeatureGroupContract:
    application: tuple[str, ...]
    bureau: tuple[str, ...]
    installments: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "application": self.application,
            "bureau": self.bureau,
            "installments": self.installments,
        }


@dataclass(frozen=True)
class AblationSpec:
    key: str
    display_name: str
    removed_group: str | None
    included_groups: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationResult:
    selected_method: str
    decision: dict[str, Any]
    experiments: tuple[dict[str, Any], ...]
    crossfit_scores: Mapping[str, np.ndarray]
    fitted_calibrator: Any


def _base_lightgbm_settings(n_jobs: int) -> dict[str, Any]:
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not 1 <= n_jobs <= 4:
        raise Stage5FinalizationError("LightGBM n_jobs는 1~4 범위여야 합니다.")
    # Stage 5 1/3의 baseline과 정확히 같은 설정이어야 비교 기준이 움직이지 않는다.
    return dict(_stage5_baseline_settings(n_jobs))


def _lightgbm_candidates(n_jobs: int = 2) -> tuple[TuningCandidate, ...]:
    """공식 validation 확인 전에 고정한 세 개의 제한 후보."""

    baseline = _base_lightgbm_settings(n_jobs)
    regularized_sampling = {
        **baseline,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "bagging_seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
    }
    higher_capacity = {
        **baseline,
        "num_leaves": 63,
        "min_child_samples": 150,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "bagging_seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
    }
    return (
        TuningCandidate("baseline", "기존 고정 설정", baseline, 0),
        TuningCandidate(
            "regularized_sampling",
            "규제·행/열 표본추출",
            regularized_sampling,
            1,
        ),
        TuningCandidate(
            "higher_capacity_regularized",
            "용량 확대·강한 규제",
            higher_capacity,
            2,
        ),
    )


def _positions_sha256(values: np.ndarray) -> str:
    positions = np.asarray(values, dtype="<i8")
    return hashlib.sha256(positions.tobytes()).hexdigest()


def _class_distribution(y: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise Stage5FinalizationError("TARGET은 0/1의 1차원 배열이어야 합니다.")
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(labels.size - positives)
    if not positives or not negatives:
        raise Stage5FinalizationError("교차검증에는 TARGET 두 클래스가 모두 필요합니다.")
    return {
        "rows": int(labels.size),
        "negative_count": negatives,
        "positive_count": positives,
        "positive_rate": positives / labels.size,
    }


def _make_cv_splits(
    y: np.ndarray,
    n_splits: int = TUNING_FOLDS,
    seed: int = RANDOM_SEED,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    labels = np.asarray(y, dtype=np.int8)
    distribution = _class_distribution(labels)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise Stage5FinalizationError("교차검증 fold 수는 2 이상의 정수여야 합니다.")
    if min(distribution["positive_count"], distribution["negative_count"]) < n_splits:
        raise Stage5FinalizationError("각 TARGET 클래스 수가 교차검증 fold 수보다 작습니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    seen = np.zeros(labels.size, dtype=np.int8)
    for fit_positions, holdout_positions in splitter.split(
        np.zeros(labels.size, dtype=np.int8), labels
    ):
        fit_positions = np.asarray(fit_positions, dtype=np.int64)
        holdout_positions = np.asarray(holdout_positions, dtype=np.int64)
        if np.intersect1d(fit_positions, holdout_positions).size:
            raise Stage5FinalizationError("교차검증 fit과 holdout 위치가 겹칩니다.")
        seen[holdout_positions] += 1
        splits.append((fit_positions, holdout_positions))
    if not np.all(seen == 1):
        raise Stage5FinalizationError("교차검증 holdout이 train 전체를 정확히 한 번 덮지 않습니다.")
    return tuple(splits)


def _cv_audit(
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int8)
    return {
        "method": "StratifiedKFold",
        "n_splits": len(splits),
        "shuffle": True,
        "random_seed": seed,
        "covers_each_train_row_once_as_holdout": True,
        "customer_ids_in_shared_output": False,
        "folds": [
            {
                "fold": index,
                "fit": _class_distribution(labels[fit_positions]),
                "holdout": _class_distribution(labels[holdout_positions]),
                "fit_positions_sha256": _positions_sha256(fit_positions),
                "holdout_positions_sha256": _positions_sha256(holdout_positions),
                "disjoint": True,
            }
            for index, (fit_positions, holdout_positions) in enumerate(splits, 1)
        ],
    }


def _load_roles_from_schema(
    version: MartVersion,
) -> tuple[FeatureRoles, str]:
    path = DEFAULT_MART_PATHS[version]
    if not path.is_file():
        raise Stage5FinalizationError(f"{version} 분석 마트를 찾을 수 없습니다: {path}")
    with duckdb.connect(database=":memory:") as connection:
        rows = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
    schema = [(str(row[0]), str(row[1])) for row in rows]
    try:
        roles = resolve_feature_roles(version, schema)
    except FeatureRoleError as error:
        raise Stage5FinalizationError(str(error)) from error
    return roles, schema_sha256(schema)


def _resolve_feature_groups(
    v1_roles: FeatureRoles,
    v2_roles: FeatureRoles,
    v3_roles: FeatureRoles,
) -> FeatureGroupContract:
    """V1⊂V2⊂V3 스키마 차집합으로 정보 원천 피처군을 고정한다."""

    if (v1_roles.version, v2_roles.version, v3_roles.version) != ("v1", "v2", "v3"):
        raise Stage5FinalizationError("피처군 계약에는 V1·V2·V3 역할이 순서대로 필요합니다.")
    v1 = v1_roles.model_features
    v2 = v2_roles.model_features
    v3 = v3_roles.model_features
    v1_set, v2_set, v3_set = set(v1), set(v2), set(v3)
    if not v1_set < v2_set or not v2_set < v3_set:
        raise Stage5FinalizationError("V1⊂V2⊂V3 피처 포함관계가 깨졌습니다.")
    contract = FeatureGroupContract(
        application=tuple(column for column in v3 if column in v1_set),
        bureau=tuple(column for column in v3 if column in v2_set - v1_set),
        installments=tuple(column for column in v3 if column in v3_set - v2_set),
    )
    groups = contract.as_dict()
    if any(len(groups[name]) != EXPECTED_GROUP_COUNTS[name] for name in groups):
        observed = {name: len(columns) for name, columns in groups.items()}
        raise Stage5FinalizationError(f"피처군 컬럼 수가 계약과 다릅니다: {observed}")
    flattened = [column for columns in groups.values() for column in columns]
    if len(flattened) != len(set(flattened)) or set(flattened) != v3_set:
        raise Stage5FinalizationError("피처군이 V3 피처를 중복 없이 모두 덮지 않습니다.")
    for group_name in ("bureau", "installments"):
        if any(column not in v3_roles.numeric for column in groups[group_name]):
            raise Stage5FinalizationError(f"{group_name} 피처군에 수치형이 아닌 피처가 있습니다.")
    return contract


def _roles_for_columns(
    source: FeatureRoles,
    columns: Sequence[str],
) -> FeatureRoles:
    requested = set(columns)
    if not requested or len(requested) != len(tuple(columns)):
        raise Stage5FinalizationError("ablation 피처는 비어 있거나 중복될 수 없습니다.")
    unknown = requested.difference(source.model_features)
    if unknown:
        raise Stage5FinalizationError(f"V3에 없는 ablation 피처가 있습니다: {sorted(unknown)}")
    numeric = tuple(column for column in source.numeric if column in requested)
    categorical = tuple(column for column in source.categorical if column in requested)
    if set(numeric).union(categorical) != requested:
        raise Stage5FinalizationError("ablation 피처 역할을 모두 분류하지 못했습니다.")
    return FeatureRoles(version="v3", numeric=numeric, categorical=categorical)


def _ablation_specs() -> tuple[AblationSpec, ...]:
    return (
        AblationSpec(
            "full",
            "전체 V3",
            None,
            ("application", "bureau", "installments"),
        ),
        AblationSpec(
            "without_application",
            "신청정보 제외",
            "application",
            ("bureau", "installments"),
        ),
        AblationSpec(
            "without_bureau",
            "외부 신용이력 제외",
            "bureau",
            ("application", "installments"),
        ),
        AblationSpec(
            "without_installments",
            "납부이력 제외",
            "installments",
            ("application", "bureau"),
        ),
    )


def _columns_for_ablation(
    contract: FeatureGroupContract,
    spec: AblationSpec,
) -> tuple[str, ...]:
    included = set(spec.included_groups)
    groups = contract.as_dict()
    if not included or not included.issubset(groups):
        raise Stage5FinalizationError(f"잘못된 ablation 피처군입니다: {spec.key}")
    # 실제 모델 입력 순서는 _roles_for_columns가 V3 수치형·범주형 순서로 재정렬한다.
    return tuple(
        column
        for name, columns in groups.items()
        if name in included
        for column in columns
    )


def _validate_transformed(values: Any, *, label: str) -> Any:
    if sparse.issparse(values):
        checked = values.tocsr().astype(np.float32, copy=False)
        finite = np.isfinite(checked.data).all()
    else:
        checked = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(checked).all()
    if checked.ndim != 2 or not finite:
        raise Stage5FinalizationError(f"{label} 전처리 결과가 유한한 2차원 행렬이 아닙니다.")
    return checked


def _score_summary(scores: np.ndarray) -> dict[str, float]:
    values = _validate_scores(scores)
    return {
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
    }


def _metric_value(metrics: Mapping[str, Any], name: str) -> float:
    if name == "recall_at_10pct":
        value = metrics["top_k_metrics"]["recall"]
    elif name == "lift_at_10pct":
        value = metrics["top_k_metrics"]["lift"]
    else:
        value = metrics[name]
    if not _finite_number(value):
        raise Stage5FinalizationError(f"평가 지표 {name}이 유한한 숫자가 아닙니다.")
    return float(value)


def _aggregate_fold_metrics(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for metric in (
        "roc_auc",
        "pr_auc",
        "ks",
        "gini",
        "brier_score",
        "recall_at_10pct",
        "lift_at_10pct",
    ):
        values = np.asarray(
            [_metric_value(fold["metrics"], metric) for fold in folds],
            dtype=np.float64,
        )
        aggregates[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return aggregates


def _candidate_result(
    candidate: TuningCandidate,
    folds: list[dict[str, Any]],
    oof_scores: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "display_name": candidate.display_name,
        "settings": dict(candidate.settings),
        "complexity_order": candidate.complexity_order,
        "folds": folds,
        "fold_aggregate": _aggregate_fold_metrics(folds),
        "oof_metrics": evaluate_binary_metrics(
            y,
            oof_scores,
            threshold=CLASSIFICATION_THRESHOLD,
            top_fraction=TOP_FRACTION,
        ),
        "oof_score_summary": _score_summary(oof_scores),
    }


def _select_tuning_candidate(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_key = {str(result["key"]): result for result in results}
    if len(by_key) != len(results) or "baseline" not in by_key:
        raise Stage5FinalizationError("튜닝 결과에는 중복 없는 baseline 후보가 필요합니다.")
    baseline = by_key["baseline"]
    baseline_aggregate = baseline["fold_aggregate"]
    checks: list[dict[str, Any]] = []
    eligible: list[Mapping[str, Any]] = []
    for result in results:
        if result["key"] == "baseline":
            continue
        pr_delta = (
            result["fold_aggregate"]["pr_auc"]["mean"]
            - baseline_aggregate["pr_auc"]["mean"]
        )
        roc_delta = (
            result["fold_aggregate"]["roc_auc"]["mean"]
            - baseline_aggregate["roc_auc"]["mean"]
        )
        recall_delta = (
            result["fold_aggregate"]["recall_at_10pct"]["mean"]
            - baseline_aggregate["recall_at_10pct"]["mean"]
        )
        fold_wins = sum(
            candidate_fold["metrics"]["pr_auc"]
            > baseline_fold["metrics"]["pr_auc"]
            for candidate_fold, baseline_fold in zip(
                result["folds"], baseline["folds"], strict=True
            )
        )
        passed = {
            "minimum_pr_auc_improvement": pr_delta >= MIN_PR_AUC_IMPROVEMENT,
            "minimum_winning_folds": fold_wins >= MIN_WINNING_FOLDS,
            "roc_auc_guard": roc_delta >= -MAX_ROC_AUC_DEGRADATION,
            "recall_at_10pct_guard": recall_delta
            >= -MAX_RECALL_AT_10PCT_DEGRADATION,
        }
        check = {
            "key": result["key"],
            "pr_auc_delta_vs_baseline": float(pr_delta),
            "roc_auc_delta_vs_baseline": float(roc_delta),
            "recall_at_10pct_delta_vs_baseline": float(recall_delta),
            "pr_auc_winning_folds": int(fold_wins),
            "guards": passed,
            "eligible": all(passed.values()),
        }
        checks.append(check)
        if check["eligible"]:
            eligible.append(result)

    if eligible:
        best_pr_auc = max(
            float(item["fold_aggregate"]["pr_auc"]["mean"])
            for item in eligible
        )
        practically_tied = [
            item
            for item in eligible
            if best_pr_auc
            - float(item["fold_aggregate"]["pr_auc"]["mean"])
            <= TUNING_PR_AUC_TIE_TOLERANCE
        ]
        selected = sorted(
            practically_tied,
            key=lambda item: (
                int(item["complexity_order"]),
                -float(item["fold_aggregate"]["pr_auc"]["mean"]),
                str(item["key"]),
            ),
        )[0]
        reason = (
            "보수적 개선 기준을 통과한 후보 중 최고 PR-AUC와 실질적 동률인 "
            "가장 단순한 설정"
        )
    else:
        selected = baseline
        reason = "개선 후보가 보수적 기준을 모두 통과하지 못해 기준 설정 유지"
    return {
        "selected_key": selected["key"],
        "selected_display_name": selected["display_name"],
        "selection_metric": "mean_fold_pr_auc",
        "reason": reason,
        "guard_policy": {
            "minimum_pr_auc_improvement_vs_baseline": MIN_PR_AUC_IMPROVEMENT,
            "minimum_pr_auc_winning_folds": MIN_WINNING_FOLDS,
            "maximum_roc_auc_degradation": MAX_ROC_AUC_DEGRADATION,
            "maximum_recall_at_10pct_degradation": MAX_RECALL_AT_10PCT_DEGRADATION,
            "pr_auc_practical_tie_tolerance": TUNING_PR_AUC_TIE_TOLERANCE,
        },
        "candidate_checks": checks,
        "official_validation_used": False,
    }


def _validate_oof_complete(scores: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.isnan(values).any():
        raise Stage5FinalizationError(f"{label} OOF 점수가 train 전체를 덮지 않습니다.")
    return _validate_scores(values)


def _fit_candidate_on_matrix(
    candidate: TuningCandidate,
    X_fit: Any,
    y_fit: np.ndarray,
    X_holdout: Any,
) -> tuple[np.ndarray, float, list[str]]:
    estimator = LGBMClassifier(**dict(candidate.settings))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    scores = _positive_scores(estimator, X_holdout)
    warning_messages = sorted(
        {f"{item.category.__name__}: {item.message}" for item in captured}
    )
    return scores, fit_seconds, warning_messages


def _run_full_feature_tuning(
    X: pd.DataFrame,
    y: np.ndarray,
    roles: FeatureRoles,
    candidates: Sequence[TuningCandidate],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if tuple(X.columns) != roles.model_features:
        raise Stage5FinalizationError("튜닝 X 컬럼과 V3 피처 역할 순서가 다릅니다.")
    if not candidates or candidates[0].key != "baseline":
        raise Stage5FinalizationError("첫 튜닝 후보는 baseline이어야 합니다.")
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise Stage5FinalizationError("튜닝 후보 key가 중복되었습니다.")
    labels = np.asarray(y, dtype=np.int8)
    _class_distribution(labels)
    oof_by_key = {
        candidate.key: np.full(labels.size, np.nan, dtype=np.float64)
        for candidate in candidates
    }
    folds_by_key: dict[str, list[dict[str, Any]]] = {
        candidate.key: [] for candidate in candidates
    }

    for fold_number, (fit_positions, holdout_positions) in enumerate(splits, 1):
        if progress is not None:
            progress(f"튜닝 fold {fold_number}/{len(splits)} 전처리")
        fold_fit_X = X.iloc[fit_positions]
        fold_holdout_X = X.iloc[holdout_positions]
        fold_fit_y = labels[fit_positions]
        fold_holdout_y = labels[holdout_positions]
        preprocessor = make_preprocessor(roles, model_family="tree")
        preprocess_started = time.perf_counter()
        transformed_fit = _validate_transformed(
            preprocessor.fit_transform(fold_fit_X, fold_fit_y),
            label=f"tuning fold {fold_number} fit",
        )
        transformed_holdout = _validate_transformed(
            preprocessor.transform(fold_holdout_X),
            label=f"tuning fold {fold_number} holdout",
        )
        preprocess_seconds = time.perf_counter() - preprocess_started
        transformed_columns = len(transformed_feature_names(preprocessor))

        for candidate in candidates:
            if progress is not None:
                progress(
                    f"튜닝 fold {fold_number}/{len(splits)}: {candidate.key} 학습"
                )
            scores, fit_seconds, captured = _fit_candidate_on_matrix(
                candidate,
                transformed_fit,
                fold_fit_y,
                transformed_holdout,
            )
            oof_by_key[candidate.key][holdout_positions] = scores
            folds_by_key[candidate.key].append(
                {
                    "fold": fold_number,
                    "fit_rows": int(fit_positions.size),
                    "holdout_rows": int(holdout_positions.size),
                    "input_feature_columns": len(roles.model_features),
                    "transformed_feature_columns": transformed_columns,
                    "shared_preprocess_seconds": round(preprocess_seconds, 3),
                    "fit_seconds": round(fit_seconds, 3),
                    "warnings": captured,
                    "metrics": evaluate_binary_metrics(
                        fold_holdout_y,
                        scores,
                        threshold=CLASSIFICATION_THRESHOLD,
                        top_fraction=TOP_FRACTION,
                    ),
                }
            )
            del scores
            gc.collect()
        del (
            preprocessor,
            transformed_fit,
            transformed_holdout,
            fold_fit_X,
            fold_holdout_X,
        )
        gc.collect()

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        complete = _validate_oof_complete(
            oof_by_key[candidate.key], label=candidate.key
        )
        oof_by_key[candidate.key] = complete
        results.append(
            _candidate_result(
                candidate,
                folds_by_key[candidate.key],
                complete,
                labels,
            )
        )
    return results, oof_by_key


def _run_one_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    source_roles: FeatureRoles,
    contract: FeatureGroupContract,
    spec: AblationSpec,
    candidate: TuningCandidate,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    requested_columns = _columns_for_ablation(contract, spec)
    roles = _roles_for_columns(source_roles, requested_columns)
    selected_X = X.loc[:, roles.model_features]
    labels = np.asarray(y, dtype=np.int8)
    oof_scores = np.full(labels.size, np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []

    for fold_number, (fit_positions, holdout_positions) in enumerate(splits, 1):
        if progress is not None:
            progress(
                f"피처군 분석 {spec.key} fold {fold_number}/{len(splits)}"
            )
        fold_fit_X = selected_X.iloc[fit_positions]
        fold_holdout_X = selected_X.iloc[holdout_positions]
        fold_fit_y = labels[fit_positions]
        fold_holdout_y = labels[holdout_positions]
        preprocessor = make_preprocessor(roles, model_family="tree")
        preprocess_started = time.perf_counter()
        transformed_fit = _validate_transformed(
            preprocessor.fit_transform(fold_fit_X, fold_fit_y),
            label=f"{spec.key} fold {fold_number} fit",
        )
        transformed_holdout = _validate_transformed(
            preprocessor.transform(fold_holdout_X),
            label=f"{spec.key} fold {fold_number} holdout",
        )
        preprocess_seconds = time.perf_counter() - preprocess_started
        transformed_columns = len(transformed_feature_names(preprocessor))
        scores, fit_seconds, captured = _fit_candidate_on_matrix(
            candidate,
            transformed_fit,
            fold_fit_y,
            transformed_holdout,
        )
        oof_scores[holdout_positions] = scores
        folds.append(
            {
                "fold": fold_number,
                "fit_rows": int(fit_positions.size),
                "holdout_rows": int(holdout_positions.size),
                "input_feature_columns": len(roles.model_features),
                "transformed_feature_columns": transformed_columns,
                "preprocess_seconds": round(preprocess_seconds, 3),
                "fit_seconds": round(fit_seconds, 3),
                "warnings": captured,
                "metrics": evaluate_binary_metrics(
                    fold_holdout_y,
                    scores,
                    threshold=CLASSIFICATION_THRESHOLD,
                    top_fraction=TOP_FRACTION,
                ),
            }
        )
        del (
            preprocessor,
            transformed_fit,
            transformed_holdout,
            fold_fit_X,
            fold_holdout_X,
            scores,
        )
        gc.collect()

    complete = _validate_oof_complete(oof_scores, label=spec.key)
    result = {
        "key": spec.key,
        "display_name": spec.display_name,
        "removed_group": spec.removed_group,
        "included_groups": list(spec.included_groups),
        "input_feature_columns": len(roles.model_features),
        "numeric_feature_columns": len(roles.numeric),
        "categorical_feature_columns": len(roles.categorical),
        "selected_candidate_key": candidate.key,
        "settings": dict(candidate.settings),
        "folds": folds,
        "fold_aggregate": _aggregate_fold_metrics(folds),
        "oof_metrics": evaluate_binary_metrics(
            labels,
            complete,
            threshold=CLASSIFICATION_THRESHOLD,
            top_fraction=TOP_FRACTION,
        ),
        "oof_score_summary": _score_summary(complete),
        "official_validation_used": False,
    }
    return result, complete


def _ablation_deltas(
    experiments: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    by_key = {str(item["key"]): item for item in experiments}
    if "full" not in by_key:
        raise Stage5FinalizationError("피처군 분석에 full 결과가 없습니다.")
    full = by_key["full"]["oof_metrics"]
    deltas: dict[str, dict[str, float]] = {}
    for key, experiment in by_key.items():
        if key == "full":
            continue
        values: dict[str, float] = {}
        for metric in (
            "roc_auc",
            "pr_auc",
            "ks",
            "gini",
            "brier_score",
            "recall_at_10pct",
            "lift_at_10pct",
        ):
            values[metric] = _metric_value(full, metric) - _metric_value(
                experiment["oof_metrics"], metric
            )
        deltas[f"full_minus_{key}"] = values
    return deltas


def _run_tuning_and_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    roles: FeatureRoles,
    groups: FeatureGroupContract,
    candidates: Sequence[TuningCandidate],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    ablation_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], TuningCandidate, dict[str, np.ndarray]]:
    """train-only 튜닝과 별도 fold의 leave-one-source-out 분석을 실행한다."""

    tuning_results, candidate_oof = _run_full_feature_tuning(
        X,
        y,
        roles,
        candidates,
        splits,
        progress=progress,
    )
    decision = _select_tuning_candidate(tuning_results)
    candidate_by_key = {candidate.key: candidate for candidate in candidates}
    selected = candidate_by_key.get(decision["selected_key"])
    if selected is None:
        raise Stage5FinalizationError("선택된 튜닝 후보 설정을 찾을 수 없습니다.")

    tuning_by_key = {result["key"]: result for result in tuning_results}
    selected_tuning = tuning_by_key[selected.key]
    reuse_tuning_oof = ablation_splits is None
    effective_ablation_splits = splits if ablation_splits is None else ablation_splits
    ablation_results: list[dict[str, Any]] = []
    selected_oof: dict[str, np.ndarray] = {}
    specs = _ablation_specs()
    if reuse_tuning_oof:
        full_result = {
            "key": "full",
            "display_name": "전체 V3",
            "removed_group": None,
            "included_groups": ["application", "bureau", "installments"],
            "input_feature_columns": len(roles.model_features),
            "numeric_feature_columns": len(roles.numeric),
            "categorical_feature_columns": len(roles.categorical),
            "selected_candidate_key": selected.key,
            "settings": dict(selected.settings),
            "folds": selected_tuning["folds"],
            "fold_aggregate": selected_tuning["fold_aggregate"],
            "oof_metrics": selected_tuning["oof_metrics"],
            "oof_score_summary": selected_tuning["oof_score_summary"],
            "official_validation_used": False,
            "oof_reused_from_tuning": True,
        }
        ablation_results.append(full_result)
        selected_oof["full"] = candidate_oof[selected.key].copy()
        specs = specs[1:]
    for spec in specs:
        result, scores = _run_one_ablation(
            X,
            y,
            roles,
            groups,
            spec,
            selected,
            effective_ablation_splits,
            progress=progress,
        )
        ablation_results.append(result)
        selected_oof[spec.key] = scores

    return (
        {
            "tuning": {
                "candidates": tuning_results,
                "decision": decision,
                "candidate_count": len(tuning_results),
                "official_validation_used": False,
                "estimate_scope": "train_only_candidate_selection_development_estimate",
            },
            "feature_ablation": {
                "method": "leave_one_source_group_out_on_train_oof",
                "experiments": ablation_results,
                "deltas": _ablation_deltas(ablation_results),
                "same_selected_settings_for_all_experiments": True,
                "same_cv_folds_for_all_experiments": True,
                "full_oof_reused_from_tuning": reuse_tuning_oof,
                "cv_separate_from_tuning": not reuse_tuning_oof,
                "interpretation_scope": (
                    "train_only_exploratory_association_not_causal_or_external_validation"
                ),
                "winner_selection_bias_still_possible": True,
                "official_validation_used": False,
            },
        },
        selected,
        selected_oof,
    )


def _new_calibrator(method: str) -> ScoreCalibrator:
    if method == "identity":
        return IdentityCalibrator()
    if method == "sigmoid":
        return SigmoidLogitCalibrator()
    if method == "isotonic":
        return IsotonicScoreCalibrator()
    raise Stage5FinalizationError(f"지원하지 않는 확률 보정 방법입니다: {method}")


def _safe_log_loss(y: np.ndarray, scores: np.ndarray) -> float:
    values = np.clip(_validate_scores(scores), SCORE_EPSILON, 1.0 - SCORE_EPSILON)
    return float(log_loss(y, values, labels=[0, 1]))


def _select_calibration_method(
    experiments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_method = {str(item["method"]): item for item in experiments}
    if len(by_method) != len(experiments) or "identity" not in by_method:
        raise Stage5FinalizationError("보정 결과에는 중복 없는 identity 기준이 필요합니다.")
    identity = by_method["identity"]
    identity_brier = float(identity["crossfit_metrics"]["brier_score"])
    identity_log_loss = float(identity["crossfit_log_loss"])
    identity_folds = identity.get("folds")
    if not isinstance(identity_folds, list) or len(identity_folds) < MIN_CALIBRATION_WINNING_FOLDS:
        raise Stage5FinalizationError("identity 보정의 fold 결과가 계약과 다릅니다.")
    checks: list[dict[str, Any]] = []
    eligible: list[Mapping[str, Any]] = []
    for method in ("sigmoid", "isotonic"):
        experiment = by_method.get(method)
        if experiment is None:
            raise Stage5FinalizationError(f"{method} 보정 결과가 없습니다.")
        brier_improvement = identity_brier - float(
            experiment["crossfit_metrics"]["brier_score"]
        )
        log_loss_delta = float(experiment["crossfit_log_loss"]) - identity_log_loss
        candidate_folds = experiment.get("folds")
        if not isinstance(candidate_folds, list) or len(candidate_folds) != len(
            identity_folds
        ):
            raise Stage5FinalizationError(f"{method} 보정의 fold 결과가 계약과 다릅니다.")
        winning_folds = sum(
            float(candidate_fold["metrics"]["brier_score"])
            < float(identity_fold["metrics"]["brier_score"])
            for candidate_fold, identity_fold in zip(
                candidate_folds, identity_folds, strict=True
            )
        )
        passed = {
            "minimum_brier_improvement": brier_improvement
            >= MIN_BRIER_IMPROVEMENT,
            "minimum_winning_folds": winning_folds
            >= MIN_CALIBRATION_WINNING_FOLDS,
            "log_loss_guard": log_loss_delta <= MAX_LOG_LOSS_DEGRADATION,
        }
        check = {
            "method": method,
            "brier_improvement_vs_identity": brier_improvement,
            "log_loss_delta_vs_identity": log_loss_delta,
            "brier_winning_folds": int(winning_folds),
            "guards": passed,
            "eligible": all(passed.values()),
        }
        checks.append(check)
        if check["eligible"]:
            eligible.append(experiment)
    if eligible:
        best_brier = min(
            float(item["crossfit_metrics"]["brier_score"]) for item in eligible
        )
        practically_tied = [
            item
            for item in eligible
            if float(item["crossfit_metrics"]["brier_score"]) - best_brier
            <= CALIBRATION_BRIER_TIE_TOLERANCE
        ]
        preference = {"sigmoid": 0, "isotonic": 1}
        selected = sorted(
            practically_tied,
            key=lambda item: (
                preference[str(item["method"])],
                float(item["crossfit_metrics"]["brier_score"]),
                float(item["crossfit_log_loss"]),
            ),
        )[0]
        reason = (
            "교차적합 Brier 개선·fold 우세·log loss 보호 기준을 통과하고, "
            "실질적 동률에서는 단순한 방법을 우선"
        )
    else:
        selected = identity
        reason = "보정 후보가 보수적 개선 기준을 통과하지 못해 원 점수 유지"
    return {
        "selected_method": selected["method"],
        "selection_metric": "cross_fitted_brier_score",
        "reason": reason,
        "guard_policy": {
            "minimum_brier_improvement_vs_identity": MIN_BRIER_IMPROVEMENT,
            "minimum_brier_winning_folds": MIN_CALIBRATION_WINNING_FOLDS,
            "maximum_log_loss_degradation": MAX_LOG_LOSS_DEGRADATION,
            "brier_practical_tie_tolerance": CALIBRATION_BRIER_TIE_TOLERANCE,
        },
        "candidate_checks": checks,
        "official_validation_used": False,
    }


def _crossfit_calibrators(
    y: np.ndarray,
    raw_oof_scores: np.ndarray,
    *,
    n_splits: int = CALIBRATION_FOLDS,
    seed: int = CALIBRATION_SEED,
) -> CalibrationResult:
    """base-model OOF 점수에서 보정기 자체도 교차적합해 선택한다."""

    scores, labels = _validate_score_labels(raw_oof_scores, y)
    splits = _make_cv_splits(labels, n_splits=n_splits, seed=seed)
    methods = ("identity", "sigmoid", "isotonic")
    crossfit_scores = {
        method: np.full(labels.size, np.nan, dtype=np.float64)
        for method in methods
    }
    fold_results: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for fold_number, (fit_positions, holdout_positions) in enumerate(splits, 1):
        for method in methods:
            calibrator = _new_calibrator(method)
            calibrator.fit(scores[fit_positions], labels[fit_positions])
            predicted = calibrator.predict(scores[holdout_positions])
            crossfit_scores[method][holdout_positions] = predicted
            fold_metrics = evaluate_binary_metrics(
                labels[holdout_positions],
                predicted,
                threshold=CLASSIFICATION_THRESHOLD,
                top_fraction=TOP_FRACTION,
            )
            fold_results[method].append(
                {
                    "fold": fold_number,
                    "fit_rows": int(fit_positions.size),
                    "holdout_rows": int(holdout_positions.size),
                    "metrics": fold_metrics,
                    "log_loss": _safe_log_loss(labels[holdout_positions], predicted),
                }
            )
    experiments: list[dict[str, Any]] = []
    for method in methods:
        complete = _validate_oof_complete(
            crossfit_scores[method], label=f"calibration {method}"
        )
        crossfit_scores[method] = complete
        experiments.append(
            {
                "method": method,
                "folds": fold_results[method],
                "crossfit_metrics": evaluate_binary_metrics(
                    labels,
                    complete,
                    threshold=CLASSIFICATION_THRESHOLD,
                    top_fraction=TOP_FRACTION,
                ),
                "crossfit_log_loss": _safe_log_loss(labels, complete),
                "calibration_summary": build_calibration_summary(
                    labels,
                    complete,
                    n_bins=CALIBRATION_BINS,
                ),
            }
        )
    decision = _select_calibration_method(experiments)
    selected_method = str(decision["selected_method"])
    fitted = _new_calibrator(selected_method)
    fitted.fit(scores, labels)
    return CalibrationResult(
        selected_method=selected_method,
        decision=decision,
        experiments=tuple(experiments),
        crossfit_scores=crossfit_scores,
        fitted_calibrator=fitted,
    )


def _load_complete_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Stage5FinalizationError(f"{label} 파일을 찾을 수 없습니다: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise Stage5FinalizationError(f"{label} JSON을 읽을 수 없습니다.") from error
    if not isinstance(payload, dict) or payload.get("run_status") != "complete":
        raise Stage5FinalizationError(f"완료된 {label} 결과가 필요합니다.")
    return payload


def _validate_reference_scope(payload: Mapping[str, Any], *, label: str) -> None:
    scope = payload.get("data_scope")
    if not isinstance(scope, Mapping) or (
        scope.get("test_feature_rows_used") != 0
        or scope.get("test_predictions_created") is not False
        or scope.get("customer_ids_in_shared_outputs") is not False
    ):
        raise Stage5FinalizationError(f"{label}의 test·공유 출력 계약이 다릅니다.")
    if "row_level_predictions_in_shared_outputs" in scope and (
        scope.get("row_level_predictions_in_shared_outputs") is not False
    ):
        raise Stage5FinalizationError(f"{label} 공유 결과에 행별 예측이 포함됐습니다.")


def _validate_reference_metrics(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Stage5FinalizationError(f"{label} 평가 지표가 없습니다.")
    for metric in ("roc_auc", "pr_auc", "ks", "gini", "brier_score"):
        if not _finite_number(value.get(metric)):
            raise Stage5FinalizationError(f"{label}.{metric}이 유한한 숫자가 아닙니다.")
    top_k = value.get("top_k_metrics")
    if not isinstance(top_k, dict):
        raise Stage5FinalizationError(f"{label} Top-K 지표가 없습니다.")
    for metric in ("recall", "precision", "lift"):
        if not _finite_number(top_k.get(metric)):
            raise Stage5FinalizationError(f"{label} Top-K {metric}이 유한하지 않습니다.")
    return value


def _find_experiment(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise Stage5FinalizationError(f"{label} 실험 목록이 없습니다.")
    matches = [
        item
        for item in experiments
        if isinstance(item, dict) and item.get("key") == key
    ]
    if len(matches) != 1:
        raise Stage5FinalizationError(f"{label}에서 {key} 결과 하나를 찾을 수 없습니다.")
    return matches[0]


def _validate_common_settings(payload: Mapping[str, Any], *, label: str) -> None:
    settings = payload.get("settings")
    if not isinstance(settings, Mapping):
        raise Stage5FinalizationError(f"{label} 평가 설정이 없습니다.")
    for key, expected in (
        ("classification_threshold", CLASSIFICATION_THRESHOLD),
        ("top_fraction", TOP_FRACTION),
    ):
        value = settings.get(key)
        if not _finite_number(value) or not np.isclose(
            float(value), expected, rtol=0.0, atol=1e-15
        ):
            raise Stage5FinalizationError(f"{label}.{key} 설정이 다릅니다.")


def _load_references(
    stage4_path: Path,
    lightgbm_path: Path,
    mlp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        stage4 = _load_baseline_reference(stage4_path)
    except RuntimeError as error:
        raise Stage5FinalizationError(str(error)) from error
    lightgbm_payload = _load_complete_json(lightgbm_path, label="Stage 5 LightGBM")
    mlp = _load_complete_json(mlp_path, label="Stage 5 MLP")
    if lightgbm_payload.get("stage_part") != "1/3":
        raise Stage5FinalizationError("LightGBM 기준 결과의 단계가 1/3이 아닙니다.")
    if mlp.get("stage_part") != "2/3":
        raise Stage5FinalizationError("MLP 기준 결과의 단계가 2/3이 아닙니다.")
    for payload, label in (
        (stage4, "Stage 4"),
        (lightgbm_payload, "Stage 5 LightGBM"),
        (mlp, "Stage 5 MLP"),
    ):
        _validate_reference_scope(payload, label=label)
        _validate_common_settings(payload, label=label)

    baseline_reference = lightgbm_payload.get("baseline_reference")
    if not isinstance(baseline_reference, Mapping) or baseline_reference.get(
        "sha256"
    ) != _sha256(stage4_path):
        raise Stage5FinalizationError("LightGBM이 참조한 Stage 4 파일이 현재 파일과 다릅니다.")
    mlp_references = mlp.get("references")
    if not isinstance(mlp_references, Mapping):
        raise Stage5FinalizationError("MLP 기준 결과에 참조 파일 정보가 없습니다.")
    for name, path in (("stage4", stage4_path), ("lightgbm", lightgbm_path)):
        reference = mlp_references.get(name)
        if not isinstance(reference, Mapping) or reference.get("sha256") != _sha256(path):
            raise Stage5FinalizationError(f"MLP가 참조한 {name} 파일이 현재 파일과 다릅니다.")

    stage4_versions = stage4.get("data_versions")
    lightgbm_versions = lightgbm_payload.get("data_versions")
    mlp_version = mlp.get("data_version")
    if (
        not isinstance(stage4_versions, Mapping)
        or not isinstance(lightgbm_versions, Mapping)
        or not isinstance(mlp_version, Mapping)
    ):
        raise Stage5FinalizationError("기준 결과의 데이터 버전 계약이 없습니다.")
    for version in ("v1", "v2", "v3"):
        stage4_version = stage4_versions.get(version)
        lightgbm_version = lightgbm_versions.get(version)
        if not isinstance(stage4_version, Mapping) or not isinstance(
            lightgbm_version, Mapping
        ):
            raise Stage5FinalizationError(f"기준 결과에 {version} 계약이 없습니다.")
        for key in (
            "schema_sha256",
            "parquet_sha256",
            "model_feature_columns",
            "numeric_feature_columns",
            "categorical_feature_columns",
            "train_rows",
            "validation_rows",
        ):
            if stage4_version.get(key) != lightgbm_version.get(key):
                raise Stage5FinalizationError(f"Stage 4와 LightGBM의 {version}.{key}가 다릅니다.")
            if (
                version == "v3"
                and key
                in {
                    "schema_sha256",
                    "parquet_sha256",
                    "model_feature_columns",
                    "numeric_feature_columns",
                    "categorical_feature_columns",
                }
                and mlp_version.get(key) != stage4_version.get(key)
            ):
                raise Stage5FinalizationError(f"MLP와 Stage 4의 V3.{key}가 다릅니다.")
    mlp_scope = mlp["data_scope"]
    for key in ("train_rows", "validation_rows"):
        if mlp_scope.get(key) != stage4["data_scope"].get(key):
            raise Stage5FinalizationError(f"MLP와 Stage 4의 {key}가 다릅니다.")

    for key, label in (
        ("logistic_v3", "V3 Logistic Regression"),
        ("random_forest_v3", "V3 Random Forest"),
    ):
        experiment = _find_experiment(stage4, key, label="Stage 4")
        _validate_reference_metrics(experiment.get("metrics"), label=label)
    lightgbm_experiment = _find_experiment(
        lightgbm_payload, "lightgbm_v3", label="Stage 5 LightGBM"
    )
    _validate_reference_metrics(
        lightgbm_experiment.get("metrics"), label="V3 LightGBM"
    )
    mlp_experiment = mlp.get("experiment")
    if not isinstance(mlp_experiment, dict) or mlp_experiment.get("key") != "mlp_v3":
        raise Stage5FinalizationError("Stage 5 MLP에 mlp_v3 결과가 없습니다.")
    _validate_reference_metrics(mlp_experiment.get("metrics"), label="V3 MLP")
    return stage4, lightgbm_payload, mlp


def _reference_candidates(
    stage4: Mapping[str, Any],
    lightgbm_payload: Mapping[str, Any],
    mlp: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = (
        _find_experiment(stage4, "logistic_v3", label="Stage 4"),
        _find_experiment(stage4, "random_forest_v3", label="Stage 4"),
        _find_experiment(
            lightgbm_payload, "lightgbm_v3", label="Stage 5 LightGBM"
        ),
        mlp["experiment"],
    )
    return [
        {
            "key": item["key"],
            "display_name": item["display_name"],
            "data_version": "v3",
            "metrics": item["metrics"],
        }
        for item in selected
    ]


def _candidate_registry_audit(
    candidates: Sequence[TuningCandidate],
    lightgbm_payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = [
        "baseline",
        "regularized_sampling",
        "higher_capacity_regularized",
    ]
    if [candidate.key for candidate in candidates] != expected_keys:
        raise Stage5FinalizationError("사전 고정 LightGBM 후보 3개가 계약과 다릅니다.")
    reference_settings = lightgbm_payload.get("settings", {}).get("lightgbm")
    if not isinstance(reference_settings, Mapping):
        raise Stage5FinalizationError("Stage 5 1/3 LightGBM 설정이 없습니다.")
    baseline = dict(candidates[0].settings)
    baseline["n_jobs"] = reference_settings.get("n_jobs")
    if baseline != dict(reference_settings):
        raise Stage5FinalizationError(
            "3/3 baseline이 Stage 5 1/3 고정 설정과 정확히 일치하지 않습니다."
        )
    registry = [
        {
            "key": candidate.key,
            "display_name": candidate.display_name,
            "complexity_order": candidate.complexity_order,
            "settings": dict(candidate.settings),
        }
        for candidate in candidates
    ]
    serialized = json.dumps(
        registry,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "candidates": registry,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "serialization": "canonical JSON UTF-8 with sorted keys and compact separators",
        "baseline_matches_stage5_part_1_except_runtime_n_jobs": True,
    }


def _validate_current_train_contract(
    train: ModelSplit,
    roles: FeatureRoles,
    stage4: Mapping[str, Any],
    lightgbm_payload: Mapping[str, Any],
    mlp: Mapping[str, Any],
    role_hashes: Mapping[str, str],
) -> None:
    if train.name != "train" or roles.version != "v3":
        raise Stage5FinalizationError("Stage 5 개발에는 V3 train만 필요합니다.")
    if tuple(train.X.columns) != roles.model_features:
        raise Stage5FinalizationError("현재 V3 train 피처 순서가 역할 계약과 다릅니다.")
    if train.customer_ids.has_duplicates or train.customer_ids.hasnans:
        raise Stage5FinalizationError("현재 V3 train 고객 키가 유일하지 않습니다.")
    if not np.isin(train.y.to_numpy(), (0, 1)).all():
        raise Stage5FinalizationError("현재 V3 train TARGET이 0/1이 아닙니다.")
    versions = stage4["data_versions"]
    for version in ("v1", "v2", "v3"):
        if versions[version].get("schema_sha256") != role_hashes[version]:
            raise Stage5FinalizationError(f"현재 {version} 스키마가 Stage 4 기준과 다릅니다.")
    expected = versions["v3"]
    observed = {
        "schema_sha256": role_hashes["v3"],
        "parquet_sha256": expected["parquet_sha256"],
        "model_feature_columns": len(roles.model_features),
        "numeric_feature_columns": len(roles.numeric),
        "categorical_feature_columns": len(roles.categorical),
        "train_rows": len(train.y),
        "validation_rows": stage4["data_scope"]["validation_rows"],
    }
    for key, value in observed.items():
        if expected.get(key) != value:
            raise Stage5FinalizationError(f"현재 V3.{key}가 Stage 4 기준과 다릅니다.")
        if lightgbm_payload["data_versions"]["v3"].get(key) != value:
            raise Stage5FinalizationError(f"현재 V3.{key}가 LightGBM 기준과 다릅니다.")
        if key in {"train_rows", "validation_rows"}:
            if mlp["data_scope"].get(key) != value:
                raise Stage5FinalizationError(f"현재 V3.{key}가 MLP 기준과 다릅니다.")
        elif mlp["data_version"].get(key) != value:
            raise Stage5FinalizationError(f"현재 V3.{key}가 MLP 기준과 다릅니다.")


def _validate_current_validation_contract(
    train: ModelSplit,
    validation: ModelSplit,
    roles: FeatureRoles,
    stage4: Mapping[str, Any],
) -> None:
    if validation.name != "validation":
        raise Stage5FinalizationError("설정 잠금 뒤 validation split을 불러와야 합니다.")
    if tuple(validation.X.columns) != roles.model_features:
        raise Stage5FinalizationError("현재 V3 validation 피처 순서가 역할 계약과 다릅니다.")
    if validation.customer_ids.has_duplicates or validation.customer_ids.hasnans:
        raise Stage5FinalizationError("현재 V3 validation 고객 키가 유일하지 않습니다.")
    if train.customer_ids.intersection(validation.customer_ids).size:
        raise Stage5FinalizationError("현재 train과 validation 고객이 겹칩니다.")
    labels = validation.y.to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise Stage5FinalizationError("현재 V3 validation TARGET이 0/1이 아닙니다.")
    expected_scope = stage4["data_scope"]
    if len(validation.y) != expected_scope.get("validation_rows"):
        raise Stage5FinalizationError("현재 validation 행 수가 Stage 4 기준과 다릅니다.")
    expected_rate = expected_scope.get("validation_positive_rate")
    if not _finite_number(expected_rate) or not np.isclose(
        float(validation.y.mean()), float(expected_rate), rtol=0.0, atol=1e-15
    ):
        raise Stage5FinalizationError("현재 validation 양성률이 Stage 4 기준과 다릅니다.")


def _validate_paths(
    *,
    output: Path,
    report: Path,
    calibration_figure: Path,
    ablation_figure: Path,
    artifact_dir: Path,
    references: Sequence[Path],
) -> None:
    if output.suffix.lower() != ".json":
        raise Stage5FinalizationError("집계 결과 경로는 .json이어야 합니다.")
    if report.suffix.lower() != ".md":
        raise Stage5FinalizationError("보고서 경로는 .md여야 합니다.")
    if any(path.suffix.lower() != ".png" for path in (calibration_figure, ablation_figure)):
        raise Stage5FinalizationError("그림 경로는 .png여야 합니다.")
    all_paths = (
        output,
        report,
        calibration_figure,
        ablation_figure,
        artifact_dir,
        *references,
    )
    resolved = [path.resolve() for path in all_paths]
    if len(resolved) != len(set(resolved)):
        raise Stage5FinalizationError("공유 결과·모델·기준 파일 경로는 서로 달라야 합니다.")
    for shared in (output, report, calibration_figure, ablation_figure):
        try:
            shared.resolve().relative_to(artifact_dir.resolve())
        except ValueError:
            continue
        raise Stage5FinalizationError("공유 산출물은 Git 제외 모델 경로 아래에 둘 수 없습니다.")


def _artifact_ignore_status(path: Path) -> bool | None:
    try:
        ignored = _git_ignored(path)
    except RuntimeError as error:
        raise Stage5FinalizationError(str(error)) from error
    if ignored is False:
        raise Stage5FinalizationError(
            f"모델·행별 점수 경로는 Git에서 제외되어야 합니다: {_display_path(path)}"
        )
    return ignored


def _artifact_metadata(path: Path) -> dict[str, Any]:
    ignored = _artifact_ignore_status(path)
    if not path.is_file():
        raise Stage5FinalizationError(f"로컬 산출물을 찾을 수 없습니다: {path}")
    return {
        "display_path": _display_path(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "git_ignored": ignored,
    }


def _atomic_joblib_dump(value: Any, path: Path) -> dict[str, Any]:
    try:
        return _lightgbm_atomic_joblib_dump(value, path)
    except RuntimeError as error:
        raise Stage5FinalizationError(str(error)) from error


def _atomic_local_json(value: Mapping[str, Any], path: Path) -> dict[str, Any]:
    _artifact_ignore_status(path)
    _atomic_write_json(path, dict(value))
    return _artifact_metadata(path)


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


def _fit_full_candidate(
    train: ModelSplit,
    roles: FeatureRoles,
    candidate: TuningCandidate,
) -> tuple[Pipeline, dict[str, Any]]:
    """선택 완료 후 새 파이프라인을 공식 train 전체에 적합한다."""

    fitted = Pipeline(
        [
            ("preprocessor", make_preprocessor(roles, model_family="tree")),
            ("model", LGBMClassifier(**dict(candidate.settings))),
        ]
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fitted.fit(train.X, train.y)
    fit_seconds = time.perf_counter() - started
    names = transformed_feature_names(fitted.named_steps["preprocessor"])
    return fitted, {
        "train_rows": len(train.y),
        "input_feature_columns": len(roles.model_features),
        "transformed_feature_columns": len(names),
        "transformed_feature_names_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
        "fit_seconds": round(fit_seconds, 3),
        "warnings": sorted(
            {f"{item.category.__name__}: {item.message}" for item in captured}
        ),
        "preprocessor_fit_scope": "full_train_only",
        "official_validation_used_for_fit": False,
        "test_used": False,
    }


def _reference_delta(
    newer: Mapping[str, Any],
    older: Mapping[str, Any],
) -> dict[str, float]:
    return {
        metric: _metric_value(newer, metric) - _metric_value(older, metric)
        for metric in (
            "roc_auc",
            "pr_auc",
            "ks",
            "gini",
            "brier_score",
            "recall_at_10pct",
            "lift_at_10pct",
        )
    }


def _build_official_validation(
    reference_candidates: Sequence[Mapping[str, Any]],
    *,
    selected_key: str,
    raw_metrics: Mapping[str, Any],
    calibrated_metrics: Mapping[str, Any],
    calibration_method: str,
    predict_seconds: float,
) -> dict[str, Any]:
    comparisons = [dict(candidate) for candidate in reference_candidates]
    comparisons.extend(
        [
            {
                "key": "stage5_selected_lightgbm_v3_raw",
                "display_name": "Stage 5 선택 V3 LightGBM (원 확률)",
                "data_version": "v3",
                "metrics": dict(raw_metrics),
            },
            {
                "key": "stage5_selected_lightgbm_v3_calibrated",
                "display_name": (
                    f"Stage 5 선택 V3 LightGBM ({calibration_method} 보정)"
                ),
                "data_version": "v3",
                "metrics": dict(calibrated_metrics),
            },
        ]
    )
    old_lightgbm = next(
        candidate for candidate in reference_candidates if candidate["key"] == "lightgbm_v3"
    )
    raw_comparisons = [
        item
        for item in comparisons
        if item["key"] != "stage5_selected_lightgbm_v3_calibrated"
    ]
    ranking = sorted(
        raw_comparisons,
        key=lambda item: (-float(item["metrics"]["pr_auc"]), str(item["key"])),
    )
    return {
        "prediction_calls": 1,
        "base_model_prediction_seconds": round(predict_seconds, 3),
        "selected_tuning_key": selected_key,
        "calibration_method_locked_before_prediction": calibration_method,
        "raw_metrics": dict(raw_metrics),
        "calibrated_metrics": dict(calibrated_metrics),
        "selected_raw_minus_stage5_1_baseline": _reference_delta(
            raw_metrics, old_lightgbm["metrics"]
        ),
        "candidate_comparison": comparisons,
        "pr_auc_ranking": [item["key"] for item in ranking],
        "pr_auc_ranking_scope": "raw_model_outputs_only",
        "used_for_tuning": False,
        "used_for_calibrator_fit": False,
        "test_used": False,
    }


def _plot_calibration(
    official: Mapping[str, Any],
    y: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_summary = build_calibration_summary(y, raw_scores, n_bins=CALIBRATION_BINS)
    calibrated_summary = build_calibration_summary(
        y, calibrated_scores, n_bins=CALIBRATION_BINS
    )
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    # 실행 환경에 한글 글꼴이 없어도 공유 그림이 깨지지 않도록 그림 안의
    # 문구는 영문으로 고정하고, 한국어 해설은 Markdown 보고서에 둔다.
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#777777",
        label="Perfect calibration",
    )
    for label, summary, color in (
        ("Raw probability", raw_summary, "#4C78A8"),
        (
            f"{official['calibration_method_locked_before_prediction']} output",
            calibrated_summary,
            "#F58518",
        ),
    ):
        axis.plot(
            [item["mean_predicted_probability"] for item in summary["bins"]],
            [item["observed_positive_rate"] for item in summary["bins"]],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed difficulty rate")
    axis.set_title("Stage 5 validation calibration")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    metadata = _atomic_save_figure(figure, path)
    return metadata, {
        "raw": raw_summary,
        "calibrated": calibrated_summary,
    }


def _plot_ablation(
    feature_ablation: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    experiments = feature_ablation["experiments"]
    label_by_key = {
        "full": "Full V3",
        "without_application": "Without application",
        "without_bureau": "Without bureau",
        "without_installments": "Without installments",
    }
    labels = [label_by_key[str(item["key"])] for item in experiments]
    values = [float(item["oof_metrics"]["pr_auc"]) for item in experiments]
    colors = ["#4C78A8" if item["key"] == "full" else "#BAB0AC" for item in experiments]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    positions = np.arange(len(experiments))
    bars = axis.bar(positions, values, color=colors)
    axis.set_xticks(positions, labels, rotation=12, ha="right")
    axis.set_ylabel("train 3-fold OOF PR-AUC")
    axis.set_title("Feature-group ablation")
    lower = max(0.0, min(values) - 0.02)
    upper = min(1.0, max(values) + 0.02)
    axis.set_ylim(lower, upper)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (upper - lower) * 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _atomic_save_figure(figure, path)


def _format_metric(value: Any, digits: int = 4, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    prefix = "+" if signed and float(value) >= 0 else ""
    return f"{prefix}{float(value):.{digits}f}"


def _markdown_relative_path(report_path: Path, target_path: Path) -> str:
    return os.path.relpath(
        target_path.resolve(), start=report_path.resolve().parent
    ).replace(os.sep, "/")


def render_markdown_report(
    payload: Mapping[str, Any],
    *,
    report_path: str | Path = DEFAULT_REPORT,
    calibration_figure_path: str | Path = DEFAULT_CALIBRATION_FIGURE,
    ablation_figure_path: str | Path = DEFAULT_ABLATION_FIGURE,
) -> str:
    report = Path(report_path)
    calibration_figure_link = _markdown_relative_path(
        report, Path(calibration_figure_path)
    )
    ablation_figure_link = _markdown_relative_path(
        report, Path(ablation_figure_path)
    )
    tuning = payload["inner_development"]["tuning"]
    ablation = payload["inner_development"]["feature_ablation"]
    calibration = payload["calibration"]
    official = payload["official_validation"]
    groups = payload["data_version"]["feature_groups"]
    lines = [
        "# Stage 5 3/3 제한 개선·확률 보정·피처군 분석 보고서",
        "",
        "> Stage 4·5의 이전 validation 비교 결과는 이미 존재합니다. 이번 3/3 실행에서는 모든 설정 선택과 확률 보정 학습을 train 내부에서 끝낸 뒤 현재 validation 피처·TARGET을 처음 로드해 base model 예측을 한 번 수행했습니다. test는 사용하지 않았습니다.",
        "",
        "## 왜 이 단계를 했는가",
        "",
        "튜닝 전 비교에서 V3 LightGBM이 가장 높았지만, 우연한 설정 하나만 보고 후보를 고정할 수는 없습니다. 그래서 작은 후보군만 train 내부 교차검증으로 비교하고, 각 정보 원천을 하나씩 제거해 예측 신호와의 연관성을 확인했습니다. 위험 순서뿐 아니라 점수를 확률로 해석할 수 있도록 보정 방법도 train 내부에서 선택했습니다.",
        "",
        "## 피처군 계약",
        "",
        "| 피처군 | 정의 | 피처 수 | 순서 포함 SHA-256 |",
        "|---|---|---:|---|",
        "| 신청정보 | V1 전체 모델 피처 | {count} | `{digest}` |".format(
            count=groups["application"]["columns"],
            digest=groups["application"]["columns_sha256"],
        ),
        "| 외부 신용이력 | V2 − V1 스키마 차집합 | {count} | `{digest}` |".format(
            count=groups["bureau"]["columns"],
            digest=groups["bureau"]["columns_sha256"],
        ),
        "| 납부이력 | V3 − V2 스키마 차집합 | {count} | `{digest}` |".format(
            count=groups["installments"]["columns"],
            digest=groups["installments"]["columns_sha256"],
        ),
        "",
        "세 그룹은 서로 겹치지 않고 합치면 V3 모델 피처 198개가 됩니다. 해시는 V3 역할 순서대로 정렬된 컬럼명을 UTF-8로 인코딩하고 줄바꿈으로 연결하되 마지막 줄바꿈은 넣지 않아 계산했습니다.",
        "",
        "## 제한 튜닝: train 3-fold 결과",
        "",
        "| 후보 | 평균 ROC-AUC | 평균 PR-AUC | 평균 Recall@10% | 선택 |",
        "|---|---:|---:|---:|---|",
    ]
    for item in tuning["candidates"]:
        aggregate = item["fold_aggregate"]
        lines.append(
            "| {name} | {roc} | {pr} | {recall} | {selected} |".format(
                name=item["display_name"],
                roc=_format_metric(aggregate["roc_auc"]["mean"]),
                pr=_format_metric(aggregate["pr_auc"]["mean"]),
                recall=_format_metric(aggregate["recall_at_10pct"]["mean"]),
                selected="채택" if item["key"] == tuning["decision"]["selected_key"] else "-",
            )
        )
    lines.extend(
        [
            "",
            f"선택 결과는 **{tuning['decision']['selected_display_name']}**입니다. {tuning['decision']['reason']}.",
            "",
            "후보를 바꾸려면 기준 설정보다 평균 PR-AUC가 0.001 이상 높고, 3개 중 2개 이상의 fold에서 우세하며, ROC-AUC와 Recall@10% 보호 기준도 통과해야 했습니다.",
            "최고 후보 간 PR-AUC 차이가 0.0005 이내면 더 단순한 설정을 선택했습니다. 아래 내부 수치는 후보 선택용 개발 추정치이며 외부 성능의 비편향 추정치로 해석하지 않습니다.",
            "",
            "## 피처군 ablation: train OOF 결과",
            "",
            "| 입력 | 피처 수 | ROC-AUC | PR-AUC | Recall@10% | full−제외 ΔPR-AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in ablation["experiments"]:
        metrics = item["oof_metrics"]
        delta = ablation["deltas"].get(f"full_minus_{item['key']}")
        lines.append(
            "| {name} | {features} | {roc} | {pr} | {recall} | {delta} |".format(
                name=item["display_name"],
                features=item["input_feature_columns"],
                roc=_format_metric(metrics["roc_auc"]),
                pr=_format_metric(metrics["pr_auc"]),
                recall=_format_metric(metrics["top_k_metrics"]["recall"]),
                delta=(
                    "-"
                    if delta is None
                    else _format_metric(delta["pr_auc"], signed=True)
                ),
            )
        )
    lines.extend(
        [
            "",
            f"![피처군 제거 비교]({ablation_figure_link})",
            "",
            "`full - 제외 모델`의 양수 차이는 해당 정보 원천을 포함한 모델이 train 내부에서 더 높았다는 뜻입니다. 튜닝과 다른 fold 배치를 사용했지만 같은 train에서 선택된 설정을 이용한 탐색적 연관성 분석이므로 인과효과나 외부 검증 결과로 해석하지 않습니다.",
            "",
            "## 확률 보정: train OOF 안의 5-fold 교차적합",
            "",
            "| 방법 | Brier | Log loss | ECE | 선택 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in calibration["experiments"]:
        lines.append(
            "| {method} | {brier} | {logloss} | {ece} | {selected} |".format(
                method=item["method"],
                brier=_format_metric(item["crossfit_metrics"]["brier_score"], 6),
                logloss=_format_metric(item["crossfit_log_loss"], 6),
                ece=_format_metric(
                    item["calibration_summary"]["expected_calibration_error"], 6
                ),
                selected=(
                    "채택"
                    if item["method"] == calibration["decision"]["selected_method"]
                    else "-"
                ),
            )
        )
    raw = official["raw_metrics"]
    calibrated = official["calibrated_metrics"]
    selected_calibration = calibration["decision"]["selected_method"]
    transformed_label = (
        "선택 변환 후(identity: 원 확률 유지)"
        if selected_calibration == "identity"
        else f"선택 변환 후({selected_calibration})"
    )
    comparison_table = [
        "| 모델·출력 | ROC-AUC | PR-AUC | KS | Brier | Recall@10% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in official["candidate_comparison"]:
        metrics = item["metrics"]
        comparison_table.append(
            "| {name} | {roc} | {pr} | {ks} | {brier} | {recall} |".format(
                name=item["display_name"],
                roc=_format_metric(metrics["roc_auc"]),
                pr=_format_metric(metrics["pr_auc"]),
                ks=_format_metric(metrics["ks"]),
                brier=(
                    "비교 제외"
                    if item["key"] in {"random_forest_v3", "mlp_v3"}
                    else _format_metric(metrics["brier_score"])
                ),
                recall=_format_metric(metrics["top_k_metrics"]["recall"]),
            )
        )
    lines.extend(
        [
            "",
            f"보정 선택은 **{selected_calibration}**입니다. {calibration['decision']['reason']}.",
            (
                "identity는 점수를 바꾸지 않는 방법이므로 원 확률과 선택 변환 후 결과가 같습니다."
                if selected_calibration == "identity"
                else "선택한 보정기는 위험순위용 원 점수와 별도로 확률 출력에만 적용합니다."
            ),
            "이 비교는 base OOF 점수 위에서 보정기만 교차적합한 train 내부 선택용 추정치이며 완전한 nested CV 결과는 아닙니다. 위험순위에는 보정 전 raw 점수를 계속 사용합니다.",
            "",
            f"![확률 보정 비교]({calibration_figure_link})",
            "",
            "## 잠금 뒤 공식 validation 결과",
            "",
            "| 출력 | ROC-AUC | PR-AUC | KS | Brier | Recall@10% | Lift@10% |",
            "|---|---:|---:|---:|---:|---:|---:|",
            "| 원 LightGBM 확률 | {roc} | {pr} | {ks} | {brier} | {recall} | {lift} |".format(
                roc=_format_metric(raw["roc_auc"]),
                pr=_format_metric(raw["pr_auc"]),
                ks=_format_metric(raw["ks"]),
                brier=_format_metric(raw["brier_score"]),
                recall=_format_metric(raw["top_k_metrics"]["recall"]),
                lift=_format_metric(raw["top_k_metrics"]["lift"]),
            ),
            "| {label} | {roc} | {pr} | {ks} | {brier} | {recall} | {lift} |".format(
                label=transformed_label,
                roc=_format_metric(calibrated["roc_auc"]),
                pr=_format_metric(calibrated["pr_auc"]),
                ks=_format_metric(calibrated["ks"]),
                brier=_format_metric(calibrated["brier_score"]),
                recall=_format_metric(calibrated["top_k_metrics"]["recall"]),
                lift=_format_metric(calibrated["top_k_metrics"]["lift"]),
            ),
            "",
            "## 전체 V3 모델 비교",
            "",
            "\n".join(comparison_table),
            "",
            "Random Forest와 MLP는 class weight를 사용한 미보정 점수이므로 Brier를 다른 비가중 모델과 직접 비교하지 않습니다. 확률 보정은 순위 성능을 높이는 작업이 아니므로 모델 순위는 raw score의 PR-AUC·ROC-AUC·KS·Top-K로 비교합니다.",
            "",
            "## Stage 6 전달 후보",
            "",
            f"- 기반 모델: {payload['stage6_candidate']['base_model_display_name']}",
            f"- 확률 변환: {payload['stage6_candidate']['calibration_method']}",
            f"- 선정 근거: {payload['stage6_candidate']['selection_rationale']}",
            "- 용도: 자동 승인·거절이 아니라 위험순위와 우선검토를 돕는 분석 후보",
            "- 상태: validation 기반 개발 후보이며 봉인 test의 최종 평가는 아직 수행하지 않음",
            "",
            "## 누수·데이터 사용 감사",
            "",
            f"- train: {payload['data_scope']['train_rows']:,}명",
            f"- validation: {payload['data_scope']['validation_rows']:,}명",
            f"- 공식 validation base-model 예측 호출: {official['prediction_calls']}회",
            "- 이번 실행의 validation 피처·TARGET은 설정 잠금 파일 저장 뒤 처음 로드",
            "- Stage 4·5의 이전 validation 비교 결과는 참조값으로만 사용",
            f"- test 피처 사용: {payload['data_scope']['test_feature_rows_used']}행",
            "- fold마다 전처리를 해당 fit 부분에만 학습",
            "- 튜닝·ablation·보정 방법 선택에 공식 validation을 사용하지 않음",
            "- 공유 결과에 고객 ID·행별 정답·행별 점수를 저장하지 않음",
            "",
        ]
    )
    return "\n".join(lines)


def _write_completed_outputs(
    *,
    output: Path,
    report: Path,
    payload: Mapping[str, Any],
    calibration_figure: Path,
    ablation_figure: Path,
) -> None:
    """보고서를 먼저 저장하고 complete JSON을 마지막에 확정한다."""

    report_content = render_markdown_report(
        payload,
        report_path=report,
        calibration_figure_path=calibration_figure,
        ablation_figure_path=ablation_figure,
    )
    _atomic_write_text(report, report_content)
    _atomic_write_json(output, dict(payload))


def run_stage5_finalization(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    calibration_figure_path: str | Path = DEFAULT_CALIBRATION_FIGURE,
    ablation_figure_path: str | Path = DEFAULT_ABLATION_FIGURE,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    stage4_reference_path: str | Path = DEFAULT_STAGE4_REFERENCE,
    lightgbm_reference_path: str | Path = DEFAULT_LIGHTGBM_REFERENCE,
    mlp_reference_path: str | Path = DEFAULT_MLP_REFERENCE,
    lightgbm_jobs: int = 2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stage 5 3/3 전체 프로토콜을 실행하고 집계 산출물을 저장한다."""

    output = Path(output_path)
    report = Path(report_path)
    calibration_figure = Path(calibration_figure_path)
    ablation_figure = Path(ablation_figure_path)
    artifacts = Path(artifact_dir)
    stage4_path = Path(stage4_reference_path)
    lightgbm_path = Path(lightgbm_reference_path)
    mlp_path = Path(mlp_reference_path)
    _validate_paths(
        output=output,
        report=report,
        calibration_figure=calibration_figure,
        ablation_figure=ablation_figure,
        artifact_dir=artifacts,
        references=(stage4_path, lightgbm_path, mlp_path),
    )
    candidates = _lightgbm_candidates(lightgbm_jobs)
    started = time.perf_counter()
    stage4, lightgbm_payload, mlp = _load_references(
        stage4_path, lightgbm_path, mlp_path
    )
    candidate_registry = _candidate_registry_audit(candidates, lightgbm_payload)
    roles_by_version: dict[str, FeatureRoles] = {}
    role_hashes: dict[str, str] = {}
    for version in ("v1", "v2", "v3"):
        roles, digest = _load_roles_from_schema(version)  # type: ignore[arg-type]
        roles_by_version[version] = roles
        role_hashes[version] = digest
    groups = _resolve_feature_groups(
        roles_by_version["v1"],
        roles_by_version["v2"],
        roles_by_version["v3"],
    )
    if progress is not None:
        progress("V3 train 전용 로딩 및 Stage 4·5 기준 계약 검증")
    train = load_model_split(DEFAULT_MART_PATHS["v3"], "v3", "train")
    v3_roles = roles_by_version["v3"]
    _validate_current_train_contract(
        train,
        v3_roles,
        stage4,
        lightgbm_payload,
        mlp,
        role_hashes,
    )
    labels = train.y.to_numpy(dtype=np.int8, copy=False)
    tuning_cv_splits = _make_cv_splits(
        labels, n_splits=TUNING_FOLDS, seed=RANDOM_SEED
    )
    ablation_cv_splits = _make_cv_splits(
        labels, n_splits=TUNING_FOLDS, seed=ABLATION_SEED
    )
    references = _reference_candidates(stage4, lightgbm_payload, mlp)

    group_audit = {
        name: {
            "columns": len(columns),
            "columns_sha256": hashlib.sha256(
                "\n".join(columns).encode("utf-8")
            ).hexdigest(),
        }
        for name, columns in groups.as_dict().items()
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "stage_part": STAGE_PART,
        "run_status": "in_progress",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "duckdb": duckdb.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "random_seed": RANDOM_SEED,
            "calibration_seed": CALIBRATION_SEED,
            "ablation_seed": ABLATION_SEED,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "top_fraction": TOP_FRACTION,
            "tuning_folds": TUNING_FOLDS,
            "calibration_folds": CALIBRATION_FOLDS,
            "candidate_order": [candidate.key for candidate in candidates],
            "candidate_registry": candidate_registry,
            "configuration_locked_before_official_validation": True,
            "official_validation_loaded_before_configuration_lock": False,
            "official_validation_used_for_tuning": False,
            "official_validation_used_for_calibration_fit": False,
        },
        "data_scope": {
            "train_rows": len(train.y),
            "validation_rows": stage4["data_scope"]["validation_rows"],
            "validation_positive_rate": stage4["data_scope"][
                "validation_positive_rate"
            ],
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
            "row_level_predictions_in_shared_outputs": False,
        },
        "references": {
            "stage4": {
                "display_path": _display_path(stage4_path),
                "sha256": _sha256(stage4_path),
                "run_version": stage4.get("run_version"),
            },
            "lightgbm": {
                "display_path": _display_path(lightgbm_path),
                "sha256": _sha256(lightgbm_path),
                "run_version": lightgbm_payload.get("run_version"),
            },
            "mlp": {
                "display_path": _display_path(mlp_path),
                "sha256": _sha256(mlp_path),
                "run_version": mlp.get("run_version"),
            },
            "candidate_snapshot": references,
        },
        "data_version": {
            "version": "v3",
            "schema_sha256": role_hashes["v3"],
            "parquet_sha256": stage4["data_versions"]["v3"]["parquet_sha256"],
            "model_feature_columns": len(v3_roles.model_features),
            "numeric_feature_columns": len(v3_roles.numeric),
            "categorical_feature_columns": len(v3_roles.categorical),
            "feature_groups": group_audit,
            "feature_group_hash_serialization": (
                "sha256 of UTF-8 ordered column names joined by newline, no terminal newline"
            ),
            "v1_v2_v3_schema_difference_verified": True,
            "stage3_summary_verified": True,
            "parquet_sha256_verified_by_train_loader": True,
        },
        "cv": {
            "tuning": _cv_audit(labels, tuning_cv_splits, seed=RANDOM_SEED),
            "ablation": _cv_audit(
                labels, ablation_cv_splits, seed=ABLATION_SEED
            ),
            "ablation_uses_separate_fold_assignment": True,
        },
        "inner_development": {},
        "calibration": {},
        "full_train_refit": {},
        "official_validation": {},
        "stage6_candidate": {},
        "artifacts": {},
        "figures": {},
        "resources": {},
    }
    _atomic_write_json(output, payload)

    if progress is not None:
        progress("train 3-fold 제한 튜닝과 피처군 제거 분석 시작")
    inner, selected, ablation_oof = _run_tuning_and_ablation(
        train.X,
        labels,
        v3_roles,
        groups,
        candidates,
        tuning_cv_splits,
        ablation_splits=ablation_cv_splits,
        progress=progress,
    )
    payload["inner_development"] = inner
    _atomic_write_json(output, payload)

    if progress is not None:
        progress("선택 후보 OOF 점수로 5-fold 확률 보정 비교")
    calibration_result = _crossfit_calibrators(
        labels,
        ablation_oof["full"],
        n_splits=CALIBRATION_FOLDS,
        seed=CALIBRATION_SEED,
    )
    payload["calibration"] = {
        "method": "five_fold_calibrator_cross_fit_on_train_base_oof",
        "base_oof_source": (
            "selected_lightgbm_from_separate_three_fold_ablation_run"
        ),
        "cv": _cv_audit(
            labels,
            _make_cv_splits(
                labels,
                n_splits=CALIBRATION_FOLDS,
                seed=CALIBRATION_SEED,
            ),
            seed=CALIBRATION_SEED,
        ),
        "experiments": list(calibration_result.experiments),
        "decision": calibration_result.decision,
        "final_calibrator_fit_scope": "all_selected_train_oof_scores",
        "class_or_sample_weight_used": False,
        "estimate_scope": "train_only_method_selection_heuristic",
        "fully_nested_unbiased_estimate": False,
        "official_validation_is_final_confirmation": True,
        "raw_score_remains_ranking_output": True,
        "official_validation_used": False,
    }
    _atomic_write_json(output, payload)

    oof_path = artifacts / "stage5_final_oof_scores.joblib"
    oof_artifact = _atomic_joblib_dump(
        {
            "schema_version": SCHEMA_VERSION,
            "run_version": RUN_VERSION,
            "split": "train_oof",
            "customer_ids_included": False,
            "y_true": labels.copy(),
            "ablation_scores": {
                key: values.astype(np.float32)
                for key, values in ablation_oof.items()
            },
            "calibration_crossfit_scores": {
                key: values.astype(np.float32)
                for key, values in calibration_result.crossfit_scores.items()
            },
        },
        oof_path,
    )

    if progress is not None:
        progress(
            f"{selected.key} 선택 완료; 새 파이프라인을 train 전체로 재학습"
        )
    fitted, refit_audit = _fit_full_candidate(train, v3_roles, selected)
    model_path = artifacts / "stage5_v3_lightgbm_candidate.joblib"
    calibrator_path = artifacts / "stage5_v3_probability_calibrator.joblib"
    model_artifact = _atomic_joblib_dump(
        fitted,
        model_path,
    )
    calibrator_artifact = _atomic_joblib_dump(
        calibration_result.fitted_calibrator,
        calibrator_path,
    )
    lock_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "locked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "locked_before_official_validation_prediction": True,
        "data_version": payload["data_version"],
        "selected_tuning_candidate": {
            "key": selected.key,
            "settings": dict(selected.settings),
        },
        "selected_calibration_method": calibration_result.selected_method,
        "model_artifact": model_artifact,
        "calibrator_artifact": calibrator_artifact,
        "oof_artifact": oof_artifact,
        "test_feature_rows_used": 0,
    }
    lock_artifact = _atomic_local_json(
        lock_payload,
        artifacts / "stage5_v3_candidate_lock_manifest.json",
    )
    payload["full_train_refit"] = {
        "selected_candidate_key": selected.key,
        "selected_candidate_display_name": selected.display_name,
        "settings": dict(selected.settings),
        **refit_audit,
        "model_artifact": model_artifact,
        "calibrator_artifact": calibrator_artifact,
        "lock_manifest": lock_artifact,
        "settings_locked_before_official_validation": True,
    }
    payload["artifacts"].update(
        {
            "model": model_artifact,
            "calibrator": calibrator_artifact,
            "train_oof_scores": oof_artifact,
            "lock_manifest": lock_artifact,
        }
    )
    _atomic_write_json(output, payload)

    # 공식 validation이 확인하는 객체와 Stage 6에 넘길 로컬 산출물이 정확히
    # 같도록 잠금 뒤 직렬화 파일을 다시 읽어 그 객체만 평가한다.
    if _artifact_metadata(model_path) != model_artifact:
        raise Stage5FinalizationError("잠금 뒤 모델 산출물 해시가 달라졌습니다.")
    if _artifact_metadata(calibrator_path) != calibrator_artifact:
        raise Stage5FinalizationError("잠금 뒤 보정기 산출물 해시가 달라졌습니다.")
    locked_fitted = joblib.load(model_path)
    locked_calibrator = joblib.load(calibrator_path)
    payload["full_train_refit"][
        "official_validation_uses_reloaded_locked_artifacts"
    ] = True

    if progress is not None:
        progress("설정 잠금 완료; 공식 validation을 처음 로드하고 예측 1회 실행")
    validation = load_model_split(
        DEFAULT_MART_PATHS["v3"], "v3", "validation"
    )
    _validate_current_validation_contract(train, validation, v3_roles, stage4)
    payload["data_scope"]["validation_positive_rate"] = float(
        validation.y.mean()
    )
    predict_started = time.perf_counter()
    raw_validation_scores = _positive_scores(locked_fitted, validation.X)
    predict_seconds = time.perf_counter() - predict_started
    calibrated_validation_scores = locked_calibrator.predict(raw_validation_scores)
    validation_y = validation.y.to_numpy(dtype=np.int8, copy=False)
    raw_metrics = evaluate_binary_metrics(
        validation_y,
        raw_validation_scores,
        threshold=CLASSIFICATION_THRESHOLD,
        top_fraction=TOP_FRACTION,
    )
    calibrated_metrics = evaluate_binary_metrics(
        validation_y,
        calibrated_validation_scores,
        threshold=CLASSIFICATION_THRESHOLD,
        top_fraction=TOP_FRACTION,
    )
    official = _build_official_validation(
        references,
        selected_key=selected.key,
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        calibration_method=calibration_result.selected_method,
        predict_seconds=predict_seconds,
    )
    payload["official_validation"] = official
    validation_artifact = _atomic_joblib_dump(
        {
            "schema_version": SCHEMA_VERSION,
            "run_version": RUN_VERSION,
            "split": "validation",
            "customer_ids_included": False,
            "y_true": validation_y.copy(),
            "scores": {
                "raw": raw_validation_scores.astype(np.float32),
                "calibrated": calibrated_validation_scores.astype(np.float32),
            },
        },
        artifacts / "stage5_final_validation_scores.joblib",
    )
    payload["artifacts"]["validation_scores"] = validation_artifact

    calibration_figure_artifact, official_calibration = _plot_calibration(
        official,
        validation_y,
        raw_validation_scores,
        calibrated_validation_scores,
        calibration_figure,
    )
    official["calibration_summary"] = official_calibration
    ablation_figure_artifact = _plot_ablation(
        inner["feature_ablation"], ablation_figure
    )
    payload["figures"] = {
        "calibration": calibration_figure_artifact,
        "feature_ablation": ablation_figure_artifact,
    }
    payload["stage6_candidate"] = {
        "base_model_key": "stage5_selected_lightgbm_v3",
        "base_model_display_name": f"V3 LightGBM ({selected.display_name})",
        "selected_tuning_candidate": selected.key,
        "calibration_method": calibration_result.selected_method,
        "ranking_score_output": "raw_lightgbm_probability",
        "probability_output": "selected_train_oof_calibration_transform",
        "selection_scope": "development_train_and_validation_only",
        "selection_rationale": (
            "기존 고정 비교에서 V3 LightGBM이 가장 높은 순위 성능을 보였고, "
            "3/3에서는 공식 validation을 사용하지 않은 train 내부 규칙으로 "
            "설정과 확률 보정 방법을 고정함"
        ),
        "model_family_selection_source": (
            "stage4_and_stage5_fixed_official_validation_comparison_before_part_3"
        ),
        "test_evaluated": False,
        "decision_status": "candidate_for_stage6_not_final_test_result",
        "automatic_credit_decision_allowed": False,
    }
    payload["data_scope"]["official_validation_prediction_calls"] = 1
    payload["data_scope"]["official_validation_loaded_after_lock"] = True
    payload["resources"] = {
        "total_seconds": round(time.perf_counter() - started, 3),
        "process_peak_rss_mb": _peak_rss_mb(),
        "measurement_scope": "current_process_lifetime",
        "execution_strategy": "sequential_folds_and_candidates",
    }
    payload["run_status"] = "complete"
    payload["completed_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    # JSON의 complete 상태는 보고서까지 안전하게 저장된 뒤 마지막에 쓴다.
    # 보고서 생성/저장 실패 시 디스크의 JSON은 in_progress 상태로 남는다.
    _write_completed_outputs(
        output=output,
        report=report,
        payload=payload,
        calibration_figure=calibration_figure,
        ablation_figure=ablation_figure,
    )
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5 3/3 제한 개선·확률 보정·피처군 분석을 실행합니다."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--calibration-figure", type=Path, default=DEFAULT_CALIBRATION_FIGURE
    )
    parser.add_argument("--ablation-figure", type=Path, default=DEFAULT_ABLATION_FIGURE)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--stage4-reference", type=Path, default=DEFAULT_STAGE4_REFERENCE
    )
    parser.add_argument(
        "--lightgbm-reference", type=Path, default=DEFAULT_LIGHTGBM_REFERENCE
    )
    parser.add_argument("--mlp-reference", type=Path, default=DEFAULT_MLP_REFERENCE)
    parser.add_argument("--lightgbm-jobs", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_stage5_finalization(
        output_path=args.output,
        report_path=args.report,
        calibration_figure_path=args.calibration_figure,
        ablation_figure_path=args.ablation_figure,
        artifact_dir=args.artifact_dir,
        stage4_reference_path=args.stage4_reference,
        lightgbm_reference_path=args.lightgbm_reference,
        mlp_reference_path=args.mlp_reference,
        lightgbm_jobs=args.lightgbm_jobs,
        progress=lambda message: print(message, flush=True),
    )
    official = result["official_validation"]
    print(
        "Stage 5 3/3 완료: "
        f"{result['stage6_candidate']['base_model_display_name']}, "
        f"PR-AUC={official['raw_metrics']['pr_auc']:.4f}, "
        f"test 피처 {result['data_scope']['test_feature_rows_used']}행 사용"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
