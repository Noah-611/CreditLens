"""Stage 2 학습 파티션 전용 탐색적 데이터 분석(EDA).

분할표를 먼저 검증하고 ``train`` 고객의 피처만 메모리에 적재한다. 검증 및
테스트 파티션은 키와 TARGET의 계약 확인에만 사용하며 피처 통계에는 절대
사용하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "creditlens-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ASSIGNMENT_COLUMNS = ("SK_ID_CURR", "TARGET", "SPLIT")
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
IDENTIFIER_PREFIX = "SK_ID_"
DAYS_EMPLOYED_SENTINEL = 365_243
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_CORRELATION_SAMPLE_SIZE = 250_000
DEFAULT_MAX_CORRELATION_COLUMNS = 120
DEFAULT_TOP_CATEGORIES = 10
FIGURE_FILENAMES = (
    "stage2_target_distribution.png",
    "stage2_missingness_top20.png",
    "stage2_numeric_distributions.png",
)
PREFERRED_NUMERIC_PLOTS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "EXT_SOURCE_2",
)

_ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:/|[A-Za-z]:[\\/]|\\\\)")


class Stage2EDAError(ValueError):
    """분할 또는 입력 데이터 계약이 안전한 EDA 실행을 허용하지 않을 때 발생한다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    """JSON 표준에 맞는 유한 실수로 변환한다."""

    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _normalize_integer_series(series: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.isna() | numeric.isna() | ~np.isclose(numeric % 1, 0)
    if bool(invalid.any()):
        raise Stage2EDAError(f"{column}에는 결측치 또는 정수가 아닌 값이 없어야 합니다.")
    return numeric.astype("int64")


def _normalize_target(series: pd.Series, source: str) -> pd.Series:
    target = _normalize_integer_series(series, "TARGET")
    invalid = ~target.isin([0, 1])
    if bool(invalid.any()):
        raise Stage2EDAError(f"{source}의 TARGET은 0 또는 1이어야 합니다.")
    return target


def _read_and_validate_assignments(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise Stage2EDAError(f"분할표를 찾을 수 없습니다: {path.name}")

    assignments = pd.read_csv(path)
    if set(assignments.columns) != set(ASSIGNMENT_COLUMNS):
        raise Stage2EDAError(
            "분할표 컬럼은 SK_ID_CURR, TARGET, SPLIT 세 개만 포함해야 합니다."
        )
    assignments = assignments.loc[:, ASSIGNMENT_COLUMNS].copy()
    if assignments.empty:
        raise Stage2EDAError("분할표가 비어 있습니다.")

    assignments["SK_ID_CURR"] = _normalize_integer_series(
        assignments["SK_ID_CURR"], "SK_ID_CURR"
    )
    assignments["TARGET"] = _normalize_target(assignments["TARGET"], "분할표")
    if bool(assignments["SK_ID_CURR"].duplicated().any()):
        raise Stage2EDAError("분할표의 고객 식별자는 중복될 수 없습니다.")
    if bool(assignments["SPLIT"].isna().any()):
        raise Stage2EDAError("분할표의 SPLIT에는 결측치가 없어야 합니다.")

    assignments["SPLIT"] = assignments["SPLIT"].astype("string")
    observed_splits = set(assignments["SPLIT"].tolist())
    unknown = observed_splits - ALLOWED_SPLITS
    missing = ALLOWED_SPLITS - observed_splits
    if unknown:
        raise Stage2EDAError(f"허용되지 않은 SPLIT 값이 있습니다: {sorted(unknown)}")
    if missing:
        raise Stage2EDAError(f"필수 SPLIT 값이 없습니다: {sorted(missing)}")
    return assignments


def _validate_application_contract(
    application_path: Path, assignments: pd.DataFrame
) -> list[str]:
    """전체 파일에서는 키와 TARGET만 읽어 분할표 계약을 확인한다."""

    if not application_path.is_file():
        raise Stage2EDAError(f"신청 데이터를 찾을 수 없습니다: {application_path.name}")

    columns = pd.read_csv(application_path, nrows=0).columns.tolist()
    required = {"SK_ID_CURR", "TARGET"}
    missing = required - set(columns)
    if missing:
        raise Stage2EDAError(f"신청 데이터 필수 컬럼이 없습니다: {sorted(missing)}")

    application_keys = pd.read_csv(
        application_path, usecols=["SK_ID_CURR", "TARGET"]
    )
    application_keys["SK_ID_CURR"] = _normalize_integer_series(
        application_keys["SK_ID_CURR"], "SK_ID_CURR"
    )
    application_keys["TARGET"] = _normalize_target(
        application_keys["TARGET"], "신청 데이터"
    )
    if bool(application_keys["SK_ID_CURR"].duplicated().any()):
        raise Stage2EDAError("신청 데이터의 고객 식별자는 중복될 수 없습니다.")

    compared = assignments.merge(
        application_keys,
        on="SK_ID_CURR",
        how="outer",
        suffixes=("_assignment", "_application"),
        indicator=True,
        validate="one_to_one",
    )
    if not bool(compared["_merge"].eq("both").all()):
        raise Stage2EDAError("분할표와 신청 데이터의 고객 집합이 정확히 일치해야 합니다.")
    if not bool(
        compared["TARGET_assignment"].eq(compared["TARGET_application"]).all()
    ):
        raise Stage2EDAError("분할표와 신청 데이터의 TARGET 값이 일치하지 않습니다.")
    return columns


def _load_train_features(
    application_path: Path,
    assignments: pd.DataFrame,
    *,
    chunk_size: int,
) -> pd.DataFrame:
    """CSV를 청크로 읽고 train 고객 행만 보존한다."""

    if chunk_size <= 0:
        raise Stage2EDAError("chunk_size는 양수여야 합니다.")

    train_assignments = assignments.loc[
        assignments["SPLIT"].eq("train"), ["SK_ID_CURR", "TARGET"]
    ].copy()
    train_ids = frozenset(train_assignments["SK_ID_CURR"].tolist())
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(application_path, chunksize=chunk_size):
        chunk_ids = pd.to_numeric(chunk["SK_ID_CURR"], errors="coerce")
        selected = chunk.loc[chunk_ids.isin(train_ids)].copy()
        if not selected.empty:
            selected["SK_ID_CURR"] = _normalize_integer_series(
                selected["SK_ID_CURR"], "SK_ID_CURR"
            )
            parts.append(selected)

    if not parts:
        raise Stage2EDAError("train 파티션이 비어 있습니다.")
    train = pd.concat(parts, ignore_index=True)
    if len(train) != len(train_assignments):
        raise Stage2EDAError("train 파티션 적재 행 수가 분할표와 일치하지 않습니다.")

    train_targets = train.set_index("SK_ID_CURR")["TARGET"]
    expected_targets = train_assignments.set_index("SK_ID_CURR")["TARGET"]
    aligned = pd.to_numeric(train_targets, errors="coerce").reindex(
        expected_targets.index
    )
    if not bool(aligned.eq(expected_targets).all()):
        raise Stage2EDAError("train 파티션의 TARGET 정합성 검증에 실패했습니다.")
    return train


def _safe_category_label(value: Any) -> str:
    if pd.isna(value):
        return "(결측)"
    label = str(value).replace("\n", " ").replace("\r", " ").strip()
    if _ABSOLUTE_PATH_PATTERN.match(label) or "file://" in label.lower():
        return "[경로 값 숨김]"
    return label[:120]


def _is_identifier(column: str) -> bool:
    return column.startswith(IDENTIFIER_PREFIX)


def _missingness_statistics(features: pd.DataFrame) -> dict[str, Any]:
    row_count = len(features)
    records = [
        {
            "column": column,
            "missing_count": int(features[column].isna().sum()),
            "missing_rate": _ratio(int(features[column].isna().sum()), row_count),
        }
        for column in features.columns
    ]
    records.sort(key=lambda item: (-item["missing_rate"], item["column"]))
    return {
        "columns_with_missing": int(
            sum(item["missing_count"] > 0 for item in records)
        ),
        "top20": records[:20],
        "all_columns": records,
    }


def _numeric_statistics(features: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    result: dict[str, Any] = {}
    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        valid = values.dropna()
        non_null_count = int(valid.size)
        unique_count = int(valid.nunique())
        quantiles = valid.quantile([0.25, 0.5, 0.75]) if non_null_count else pd.Series(dtype=float)
        q1 = _finite_float(quantiles.get(0.25))
        median = _finite_float(quantiles.get(0.5))
        q3 = _finite_float(quantiles.get(0.75))

        lower_bound: float | None = None
        upper_bound: float | None = None
        outlier_count: int | None = None
        outlier_rate: float | None = None
        if unique_count > 2 and q1 is not None and q3 is not None:
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = int(((valid < lower_bound) | (valid > upper_bound)).sum())
            outlier_rate = _ratio(outlier_count, non_null_count)

        result[column] = {
            "non_null_count": non_null_count,
            "missing_count": int(values.isna().sum()),
            "unique_count": unique_count,
            "mean": _finite_float(valid.mean()) if non_null_count else None,
            "std": _finite_float(valid.std()) if non_null_count > 1 else None,
            "min": _finite_float(valid.min()) if non_null_count else None,
            "p1": _finite_float(valid.quantile(0.01)) if non_null_count else None,
            "q1": q1,
            "median": median,
            "q3": q3,
            "p95": _finite_float(valid.quantile(0.95)) if non_null_count else None,
            "p99": _finite_float(valid.quantile(0.99)) if non_null_count else None,
            "p99_9": _finite_float(valid.quantile(0.999)) if non_null_count else None,
            "max": _finite_float(valid.max()) if non_null_count else None,
            "iqr_outlier_lower_bound": _finite_float(lower_bound),
            "iqr_outlier_upper_bound": _finite_float(upper_bound),
            "iqr_outlier_count": outlier_count,
            "iqr_outlier_rate": outlier_rate,
        }
    return result, numeric_columns


def _categorical_statistics(
    features: pd.DataFrame,
    numeric_columns: Iterable[str],
    *,
    top_categories: int,
) -> tuple[dict[str, Any], list[str]]:
    numeric_set = set(numeric_columns)
    categorical_columns = [
        column for column in features.columns if column not in numeric_set
    ]
    row_count = len(features)
    result: dict[str, Any] = {}
    for column in categorical_columns:
        series = features[column]
        counts = series.value_counts(dropna=False).head(top_categories)
        top_values = [
            {
                "value": _safe_category_label(value),
                "count": int(count),
                "rate": _ratio(int(count), row_count),
            }
            for value, count in counts.items()
        ]
        result[column] = {
            "missing_count": int(series.isna().sum()),
            "missing_rate": _ratio(int(series.isna().sum()), row_count),
            "unique_count": int(series.nunique(dropna=True)),
            "top_values": top_values,
        }
    return result, categorical_columns


def _correlation_statistics(
    features: pd.DataFrame,
    numeric_columns: Sequence[str],
    *,
    sample_size: int,
    max_columns: int,
) -> dict[str, Any]:
    eligible = [
        column
        for column in numeric_columns
        if pd.to_numeric(features[column], errors="coerce").nunique(dropna=True) > 2
    ]
    preferred = [column for column in PREFERRED_NUMERIC_PLOTS if column in eligible]
    remaining = sorted(
        (column for column in eligible if column not in preferred),
        key=lambda column: (-int(features[column].notna().sum()), column),
    )
    selected = (preferred + remaining)[:max_columns]
    if len(features) > sample_size:
        sample = features.loc[:, selected].sample(n=sample_size, random_state=42)
    else:
        sample = features.loc[:, selected]
    numeric_sample = sample.apply(pd.to_numeric, errors="coerce")
    correlation = numeric_sample.corr(method="pearson")

    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            value = _finite_float(correlation.loc[left, right])
            if value is not None:
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    pairs.sort(
        key=lambda item: (
            -item["absolute_correlation"],
            item["left"],
            item["right"],
        )
    )
    high_correlation_pairs = [
        item for item in pairs if item["absolute_correlation"] >= 0.95
    ]
    return {
        "method": "pearson",
        "sample_rows": int(len(sample)),
        "selected_column_count": len(selected),
        "absolute_correlation_at_least_0_95_pair_count": len(
            high_correlation_pairs
        ),
        "absolute_pairs_at_least_0_95": high_correlation_pairs,
        "top20_absolute_pairs": pairs[:20],
    }


def _sentinel_statistics(train: pd.DataFrame) -> dict[str, Any]:
    if "DAYS_EMPLOYED" not in train.columns:
        return {
            "available": False,
            "sentinel": DAYS_EMPLOYED_SENTINEL,
            "count": 0,
            "rate": 0.0,
        }
    values = pd.to_numeric(train["DAYS_EMPLOYED"], errors="coerce")
    count = int(values.eq(DAYS_EMPLOYED_SENTINEL).sum())
    return {
        "available": True,
        "sentinel": DAYS_EMPLOYED_SENTINEL,
        "count": count,
        "rate": _ratio(count, len(train)),
    }


def _target_relationships(
    features: pd.DataFrame,
    target: pd.Series,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> dict[str, Any]:
    """train 안에서만 피처와 TARGET의 단변량 관계를 집계한다."""

    minimum_support = max(20, math.ceil(len(features) * 0.001))
    overall_rate = float(target.mean())

    numeric_result: dict[str, Any] = {}
    numeric_ranking: list[dict[str, Any]] = []
    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        by_target: dict[str, Any] = {}
        for target_value in (0, 1):
            selected = values.loc[target.eq(target_value)].dropna()
            by_target[str(target_value)] = {
                "count": int(len(selected)),
                "mean": _finite_float(selected.mean()) if not selected.empty else None,
                "median": _finite_float(selected.median()) if not selected.empty else None,
            }
        valid = values.notna()
        valid_values = values.loc[valid]
        valid_target = target.loc[valid]
        point_biserial = (
            _finite_float(valid_values.corr(valid_target, method="pearson"))
            if int(valid.sum()) >= 2
            and int(valid_values.nunique()) > 1
            and int(valid_target.nunique()) > 1
            else None
        )
        numeric_result[column] = {
            "by_target": by_target,
            "point_biserial_correlation": point_biserial,
        }
        if point_biserial is not None:
            numeric_ranking.append(
                {
                    "column": column,
                    "correlation": point_biserial,
                    "absolute_correlation": abs(point_biserial),
                }
            )
    numeric_ranking.sort(
        key=lambda item: (-item["absolute_correlation"], item["column"])
    )

    missing_result: list[dict[str, Any]] = []
    for column in features.columns:
        missing = features[column].isna()
        missing_count = int(missing.sum())
        observed_count = int((~missing).sum())
        if missing_count < minimum_support or observed_count < minimum_support:
            continue
        missing_rate = float(target.loc[missing].mean())
        observed_rate = float(target.loc[~missing].mean())
        missing_result.append(
            {
                "column": column,
                "missing_count": missing_count,
                "observed_count": observed_count,
                "target_rate_when_missing": missing_rate,
                "target_rate_when_observed": observed_rate,
                "target_rate_difference": missing_rate - observed_rate,
            }
        )
    missing_result.sort(
        key=lambda item: (-abs(item["target_rate_difference"]), item["column"])
    )

    categorical_result: dict[str, Any] = {}
    category_ranking: list[dict[str, Any]] = []
    for column in categorical_columns:
        values = features[column].astype("object").where(features[column].notna(), "[결측]")
        counts = values.value_counts(dropna=False)
        rare_values = set(counts.loc[counts < minimum_support].index.tolist())
        grouped_values = values.map(
            lambda value: "[기타 희소 범주]" if value in rare_values else value
        )
        grouped = (
            pd.DataFrame({"category": grouped_values, "target": target})
            .groupby("category", dropna=False, sort=False)["target"]
            .agg(["count", "mean"])
            .reset_index()
        )
        records: list[dict[str, Any]] = []
        for row in grouped.itertuples(index=False):
            count = int(row.count)
            if count < minimum_support:
                continue
            rate = float(row.mean)
            record = {
                "value": _safe_category_label(row.category),
                "count": count,
                "target_rate": rate,
                "difference_from_train_rate": rate - overall_rate,
            }
            records.append(record)
            category_ranking.append({"column": column, **record})
        records.sort(key=lambda item: (-item["count"], item["value"]))
        categorical_result[column] = records
    category_ranking.sort(
        key=lambda item: (
            -abs(item["difference_from_train_rate"]),
            -item["count"],
            item["column"],
            item["value"],
        )
    )

    return {
        "scope": "train_only",
        "minimum_support": minimum_support,
        "train_target_rate": overall_rate,
        "numeric": {
            "method": "TARGET별 평균·중앙값과 point-biserial 상관계수",
            "columns": numeric_result,
            "top20_absolute_correlations": numeric_ranking[:20],
        },
        "missingness": {
            "method": "결측/관측 집단의 TARGET=1 비율 차이",
            "columns_meeting_minimum_support": missing_result,
        },
        "categorical": {
            "method": "최소 지지 건수 미만 범주를 기타 희소 범주로 합산한 TARGET=1 비율",
            "columns": categorical_result,
            "top20_rate_differences": category_ranking[:20],
        },
    }


def _category_count(
    features: pd.DataFrame,
    column: str,
    expected_value: str,
) -> dict[str, Any]:
    if column not in features.columns:
        return {"available": False, "count": 0, "rate": 0.0}
    count = int(features[column].eq(expected_value).sum())
    return {
        "available": True,
        "count": count,
        "rate": _ratio(count, len(features)),
    }


def _data_quality_observations(
    features: pd.DataFrame,
    categorical_columns: Sequence[str],
    missingness: Mapping[str, Any],
) -> dict[str, Any]:
    """Home Credit 고유 이상값과 일반적인 희소·거의 상수 피처를 집계한다."""

    near_constant: list[dict[str, Any]] = []
    for column in features.columns:
        counts = features[column].value_counts(dropna=False)
        if counts.empty:
            continue
        top_value = counts.index[0]
        top_count = int(counts.iloc[0])
        top_rate = _ratio(top_count, len(features))
        if top_rate >= 0.99:
            near_constant.append(
                {
                    "column": column,
                    "dominant_value": _safe_category_label(top_value),
                    "count": top_count,
                    "rate": top_rate,
                }
            )
    near_constant.sort(key=lambda item: (-item["rate"], item["column"]))

    rare_threshold = max(2, math.ceil(len(features) * 0.001))
    rare_categories: list[dict[str, Any]] = []
    for column in categorical_columns:
        series = features[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count > 100:
            continue
        counts = series.value_counts(dropna=True)
        for value, count in counts.items():
            count = int(count)
            if count <= rare_threshold:
                rare_categories.append(
                    {
                        "column": column,
                        "value": _safe_category_label(value),
                        "count": count,
                        "rate": _ratio(count, len(features)),
                    }
                )
    rare_categories.sort(key=lambda item: (item["count"], item["column"], item["value"]))

    car_age_missing: dict[str, Any] = {
        "available": False,
        "car_owner_count": 0,
        "missing_count_among_car_owners": 0,
        "rate_among_car_owners": 0.0,
    }
    if {"FLAG_OWN_CAR", "OWN_CAR_AGE"}.issubset(features.columns):
        car_owner = features["FLAG_OWN_CAR"].eq("Y")
        owner_count = int(car_owner.sum())
        missing_count = int(features.loc[car_owner, "OWN_CAR_AGE"].isna().sum())
        car_age_missing = {
            "available": True,
            "car_owner_count": owner_count,
            "missing_count_among_car_owners": missing_count,
            "rate_among_car_owners": _ratio(missing_count, owner_count),
        }

    high_missing = [
        item
        for item in missingness["all_columns"]
        if float(item["missing_rate"]) >= 0.40
    ]
    return {
        "known_category_values": {
            "CODE_GENDER_XNA": _category_count(features, "CODE_GENDER", "XNA"),
            "NAME_FAMILY_STATUS_Unknown": _category_count(
                features, "NAME_FAMILY_STATUS", "Unknown"
            ),
        },
        "own_car_age_missingness": car_age_missing,
        "high_missing_columns_at_least_40_percent": high_missing,
        "near_constant_columns_at_least_99_percent": near_constant,
        "rare_category_rule": {
            "maximum_count": rare_threshold,
            "maximum_unique_values_for_labels": 100,
        },
        "rare_categories": rare_categories[:100],
    }


def _build_statistics(
    train: pd.DataFrame,
    *,
    application_name: str,
    assignment_name: str,
    application_sha256: str,
    assignment_sha256: str,
    correlation_sample_size: int,
    max_correlation_columns: int,
    top_categories: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    feature_columns = [
        column
        for column in train.columns
        if column != "TARGET" and not _is_identifier(column)
    ]
    features = train.loc[:, feature_columns]
    target = _normalize_target(train["TARGET"], "train 파티션")
    target_counts = target.value_counts().reindex([0, 1], fill_value=0)
    missingness = _missingness_statistics(features)
    numeric, numeric_columns = _numeric_statistics(features)
    categorical, categorical_columns = _categorical_statistics(
        features, numeric_columns, top_categories=top_categories
    )
    correlation = _correlation_statistics(
        features,
        numeric_columns,
        sample_size=correlation_sample_size,
        max_columns=max_correlation_columns,
    )
    target_relationships = _target_relationships(
        features, target, numeric_columns, categorical_columns
    )

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "Stage 2",
        "analysis_scope": {
            "split": "train",
            "train_rows": int(len(train)),
            "feature_columns": len(feature_columns),
            "identifier_values_included": False,
            "non_train_feature_rows_used": 0,
        },
        "source": {
            "application_file": application_name,
            "application_sha256": application_sha256,
            "assignment_file": assignment_name,
            "assignment_sha256": assignment_sha256,
        },
        "target_distribution": {
            "0": {
                "count": int(target_counts.loc[0]),
                "rate": _ratio(int(target_counts.loc[0]), len(target)),
            },
            "1": {
                "count": int(target_counts.loc[1]),
                "rate": _ratio(int(target_counts.loc[1]), len(target)),
            },
        },
        "column_summary": {
            "numeric_columns": len(numeric_columns),
            "categorical_columns": len(categorical_columns),
            "columns_with_missing": missingness["columns_with_missing"],
        },
        "missingness": missingness,
        "numeric_statistics": {
            "outlier_method": "IQR 1.5배 경계(고유값 3개 이상인 수치형 컬럼)",
            "columns": numeric,
        },
        "categorical_statistics": {
            "top_value_limit": top_categories,
            "columns": categorical,
        },
        "correlation": correlation,
        "target_relationships": target_relationships,
        "sentinel_statistics": {
            "DAYS_EMPLOYED_365243": _sentinel_statistics(train)
        },
        "data_quality_observations": _data_quality_observations(
            features, categorical_columns, missingness
        ),
        "figures": list(FIGURE_FILENAMES),
    }
    return result, features, numeric_columns


def _plot_target_distribution(train: pd.DataFrame, output_path: Path) -> None:
    counts = (
        pd.to_numeric(train["TARGET"], errors="coerce")
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    fig, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(["0: Repayment", "1: Difficulty"], counts.values, color=["#3973AC", "#D95F59"])
    axis.set_title("Train target distribution")
    axis.set_ylabel("Customers")
    axis.grid(axis="y", alpha=0.2)
    for bar, count in zip(bars, counts.values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_missingness(features: pd.DataFrame, output_path: Path) -> None:
    rates = features.isna().mean().sort_values(ascending=False).head(20).sort_values()
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(rates.index, rates.values * 100, color="#5C8D89")
    axis.set_title("Train missingness: top 20")
    axis.set_xlabel("Missing rate (%)")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _select_plot_columns(features: pd.DataFrame, numeric_columns: Sequence[str]) -> list[str]:
    eligible = [
        column
        for column in numeric_columns
        if pd.to_numeric(features[column], errors="coerce").nunique(dropna=True) > 2
    ]
    preferred = [column for column in PREFERRED_NUMERIC_PLOTS if column in eligible]
    fallback = [column for column in eligible if column not in preferred]
    return (preferred + fallback)[:6]


def _plot_numeric_distributions(
    features: pd.DataFrame,
    numeric_columns: Sequence[str],
    output_path: Path,
) -> None:
    selected = _select_plot_columns(features, numeric_columns)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    flat_axes = axes.ravel()
    for axis, column in zip(flat_axes, selected, strict=False):
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        if values.empty:
            axis.set_visible(False)
            continue
        lower, upper = values.quantile([0.01, 0.99])
        clipped = values.clip(lower=lower, upper=upper)
        axis.hist(clipped, bins=40, color="#3973AC", alpha=0.85)
        axis.set_title(column)
        axis.set_ylabel("Rows")
        axis.grid(axis="y", alpha=0.2)
    for axis in flat_axes[len(selected) :]:
        axis.set_visible(False)
    fig.suptitle("Train numeric distributions (1st-99th percentile clipped)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_figures(
    train: pd.DataFrame,
    features: pd.DataFrame,
    numeric_columns: Sequence[str],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_target_distribution(train, figures_dir / FIGURE_FILENAMES[0])
    _plot_missingness(features, figures_dir / FIGURE_FILENAMES[1])
    _plot_numeric_distributions(
        features, numeric_columns, figures_dir / FIGURE_FILENAMES[2]
    )


def _format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown_report(result: Mapping[str, Any]) -> str:
    """집계 JSON과 동일한 train-only 내용을 한국어 Markdown으로 렌더링한다."""

    scope = result["analysis_scope"]
    target = result["target_distribution"]
    summary = result["column_summary"]
    sentinel = result["sentinel_statistics"]["DAYS_EMPLOYED_365243"]
    quality = result["data_quality_observations"]
    known_categories = quality["known_category_values"]
    car_age_missing = quality["own_car_age_missingness"]
    missing = result["missingness"]["top20"]
    correlations = result["correlation"]["top20_absolute_pairs"][:10]
    target_relationships = result["target_relationships"]
    numeric_target = target_relationships["numeric"]["top20_absolute_correlations"][:10]
    missing_target = target_relationships["missingness"][
        "columns_meeting_minimum_support"
    ][:10]
    category_target = target_relationships["categorical"]["top20_rate_differences"][:10]

    lines = [
        "# Stage 2 학습 데이터 EDA 보고서",
        "",
        "> 이 보고서의 모든 피처 통계와 그래프는 `train` 파티션만 사용했습니다. ",
        "> 검증 및 테스트 고객의 피처는 분석에 포함하지 않았으며 고객 식별자도 저장하지 않았습니다.",
        "",
        "## 분석 범위",
        "",
        f"- 학습 행 수: **{int(scope['train_rows']):,}**",
        f"- 분석 피처 수: **{int(scope['feature_columns']):,}**",
        f"- 수치형 / 범주형: **{int(summary['numeric_columns'])} / {int(summary['categorical_columns'])}**",
        f"- 결측치가 있는 피처: **{int(summary['columns_with_missing'])}**",
        "- 수행 범위: 분포·결측·상관·IQR 이상치와 알려진 sentinel 확인",
        "- 제외 범위: 전처리 적용, 피처 생성, 모델 학습 및 성능 평가",
        "",
        "## TARGET 분포",
        "",
        "| TARGET | 고객 수 | 비율 |",
        "|---:|---:|---:|",
        f"| 0 | {int(target['0']['count']):,} | {float(target['0']['rate']):.2%} |",
        f"| 1 | {int(target['1']['count']):,} | {float(target['1']['rate']):.2%} |",
        "",
        f"![학습 TARGET 분포](../reports/figures/{FIGURE_FILENAMES[0]})",
        "",
        "## 결측률 상위 피처",
        "",
        "| 순위 | 피처 | 결측 수 | 결측률 |",
        "|---:|---|---:|---:|",
    ]
    for index, item in enumerate(missing, start=1):
        lines.append(
            f"| {index} | `{_markdown_escape(item['column'])}` | "
            f"{int(item['missing_count']):,} | {float(item['missing_rate']):.2%} |"
        )
    lines.extend(
        [
            "",
            f"![학습 결측률 상위 20개](../reports/figures/{FIGURE_FILENAMES[1]})",
            "",
            "## 수치형 분포와 이상치",
            "",
            "수치형 피처의 평균·표준편차·사분위수와 IQR 1.5배 경계 밖의 건수를 JSON에 기록했습니다. "
            "행을 제거하거나 값을 보정하지 않았으며, 그래프만 보기 쉽도록 1~99백분위 구간으로 표시했습니다.",
            "",
            f"![학습 주요 수치형 분포](../reports/figures/{FIGURE_FILENAMES[2]})",
            "",
            "## `DAYS_EMPLOYED` sentinel",
            "",
            f"- 값 `{int(sentinel['sentinel']):,}` 건수: **{int(sentinel['count']):,}**",
            f"- 학습 고객 대비 비율: **{float(sentinel['rate']):.2%}**",
            "- 이 값은 실제 근속일로 해석하지 않고 Stage 2 전처리 정책에서 별도 처리해야 합니다.",
            "",
            "## 알려진 데이터 품질 확인 항목",
            "",
            f"- `CODE_GENDER=XNA`: **{int(known_categories['CODE_GENDER_XNA']['count']):,}건**",
            f"- `NAME_FAMILY_STATUS=Unknown`: **{int(known_categories['NAME_FAMILY_STATUS_Unknown']['count']):,}건**",
            f"- 차량 보유 고객의 `OWN_CAR_AGE` 결측: **{int(car_age_missing['missing_count_among_car_owners']):,}건** "
            f"({float(car_age_missing['rate_among_car_owners']):.2%})",
            f"- 결측률 40% 이상 피처: **{len(quality['high_missing_columns_at_least_40_percent']):,}개**",
            f"- 한 값이 99% 이상인 거의 상수 피처: **{len(quality['near_constant_columns_at_least_99_percent']):,}개**",
            f"- 희소 범주 후보: **{len(quality['rare_categories']):,}개**(JSON에 최대 100개 기록)",
            "- 위 항목은 처리 대상을 찾기 위한 관찰이며 이 보고서에서 값을 바꾸거나 행을 삭제하지 않았습니다.",
            "",
            "## 피처와 TARGET의 관계",
            "",
            f"최소 지지 건수는 **{int(target_relationships['minimum_support']):,}건**입니다. "
            "이보다 작은 범주는 `[기타 희소 범주]`로 합쳤으며 모든 계산은 train 안에서만 수행했습니다.",
            "",
            "### 수치형 피처",
            "",
            "| 피처 | point-biserial 상관계수 |",
            "|---|---:|",
        ]
    )
    if numeric_target:
        for item in numeric_target:
            lines.append(
                f"| `{_markdown_escape(item['column'])}` | {_format_number(item['correlation'])} |"
            )
    else:
        lines.append("| - | 계산 가능한 피처 없음 |")
    lines.extend(
        [
            "",
            "### 결측 여부",
            "",
            "| 피처 | 결측 시 TARGET=1 | 관측 시 TARGET=1 | 차이 |",
            "|---|---:|---:|---:|",
        ]
    )
    if missing_target:
        for item in missing_target:
            lines.append(
                f"| `{_markdown_escape(item['column'])}` | "
                f"{float(item['target_rate_when_missing']):.2%} | "
                f"{float(item['target_rate_when_observed']):.2%} | "
                f"{float(item['target_rate_difference']):+.2%} |"
            )
    else:
        lines.append("| - | - | - | 비교 가능한 피처 없음 |")
    lines.extend(
        [
            "",
            "### 범주형 피처",
            "",
            "| 피처 | 범주 | 건수 | TARGET=1 | 학습 전체 대비 차이 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    if category_target:
        for item in category_target:
            lines.append(
                f"| `{_markdown_escape(item['column'])}` | "
                f"{_markdown_escape(item['value'])} | {int(item['count']):,} | "
                f"{float(item['target_rate']):.2%} | "
                f"{float(item['difference_from_train_rate']):+.2%} |"
            )
    else:
        lines.append("| - | - | - | - | 비교 가능한 범주 없음 |")
    lines.extend(
        [
            "",
            "## 절댓값 상관계수 상위",
            "",
            f"train **{int(result['correlation']['sample_rows']):,}행**, "
            f"수치 피처 **{int(result['correlation']['selected_column_count']):,}개**로 계산했습니다. "
            "절댓값 0.95 이상인 피처 쌍은 "
            f"**{int(result['correlation']['absolute_correlation_at_least_0_95_pair_count']):,}개**입니다.",
            "전체 0.95 이상 목록은 `reports/stage2_eda.json`의 "
            "`correlation.absolute_pairs_at_least_0_95`에 기록했습니다. 아래 표는 상위 10개입니다.",
            "",
            "| 피처 A | 피처 B | Pearson 상관계수 |",
            "|---|---|---:|",
        ]
    )
    if correlations:
        for item in correlations:
            lines.append(
                f"| `{_markdown_escape(item['left'])}` | `{_markdown_escape(item['right'])}` | "
                f"{_format_number(item['correlation'])} |"
            )
    else:
        lines.append("| - | - | 계산 가능한 피처 쌍 없음 |")
    lines.extend(
        [
            "",
            "## 해석 시 주의사항",
            "",
            "- 상관관계는 인과관계를 뜻하지 않습니다.",
            "- IQR 밖의 값은 검토 후보이며 자동 삭제 대상이 아닙니다.",
            "- 검증·테스트 파티션은 전처리 정책을 확정하는 근거로 사용하지 않습니다.",
            "- 이 단계에서는 모델을 학습하지 않았습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_strings(child)


def _assert_output_safety(result: Mapping[str, Any], markdown: str) -> None:
    if bool(result["analysis_scope"]["identifier_values_included"]):
        raise Stage2EDAError("집계 결과에 고객 식별자 값이 포함될 수 없습니다.")
    if int(result["analysis_scope"]["non_train_feature_rows_used"]) != 0:
        raise Stage2EDAError("train 이외 파티션의 피처가 통계에 포함되었습니다.")
    for value in [*_walk_strings(result), markdown]:
        for line in value.splitlines():
            stripped = line.strip().strip("`[]()")
            if _ABSOLUTE_PATH_PATTERN.match(stripped) or "file://" in stripped.lower():
                raise Stage2EDAError("출력에 절대경로가 포함될 수 없습니다.")


def _resolved_path(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise Stage2EDAError(f"경로를 안전하게 확인할 수 없습니다: {path.name}") from error


def _paths_collide(first: Path, second: Path) -> bool:
    """상대경로·심볼릭 링크·기존 하드 링크까지 같은 파일인지 확인한다."""

    if _resolved_path(first) == _resolved_path(second):
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def _check_output_paths(
    application: Path,
    assignment: Path,
    json_output: Path,
    report_output: Path,
    figures_output: Path,
) -> None:
    """잘못된 출력 인자로 입력 파일이나 다른 결과물을 덮어쓰지 않게 한다."""

    inputs = {
        "원본 신청 데이터": application,
        "고객 분할표": assignment,
    }
    outputs = {
        "EDA JSON": json_output,
        "EDA Markdown": report_output,
        **{
            f"EDA 그림 {filename}": figures_output / filename
            for filename in FIGURE_FILENAMES
        },
    }

    for output_label, output_path in outputs.items():
        for input_label, input_path in inputs.items():
            if _paths_collide(output_path, input_path):
                raise Stage2EDAError(
                    f"입력·출력 경로 충돌: {output_label}은 {input_label}과 "
                    "같은 경로일 수 없습니다."
                )

    output_items = list(outputs.items())
    for index, (left_label, left_path) in enumerate(output_items):
        for right_label, right_path in output_items[index + 1 :]:
            if _paths_collide(left_path, right_path):
                raise Stage2EDAError(
                    f"출력 경로 충돌: {left_label}과 {right_label}은 "
                    "서로 다른 경로여야 합니다."
                )

    for input_label, input_path in inputs.items():
        if _paths_collide(figures_output, input_path):
            raise Stage2EDAError(
                f"입력·출력 경로 충돌: 그림 디렉터리는 {input_label}과 "
                "같은 경로일 수 없습니다."
            )
    for output_label, output_path in (
        ("EDA JSON", json_output),
        ("EDA Markdown", report_output),
    ):
        if _paths_collide(figures_output, output_path):
            raise Stage2EDAError(
                f"출력 경로 충돌: 그림 디렉터리와 {output_label}은 "
                "같은 경로일 수 없습니다."
            )


def run_stage2_eda(
    application_path: str | Path = "data/raw/application_train.csv",
    assignment_path: str | Path = "data/interim/customer_splits.csv",
    json_path: str | Path = "reports/stage2_eda.json",
    report_path: str | Path = "docs/Stage2_EDA_Report.md",
    figures_dir: str | Path = "reports/figures",
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    correlation_sample_size: int = DEFAULT_CORRELATION_SAMPLE_SIZE,
    max_correlation_columns: int = DEFAULT_MAX_CORRELATION_COLUMNS,
    top_categories: int = DEFAULT_TOP_CATEGORIES,
) -> dict[str, Any]:
    """train 전용 EDA를 실행하고 JSON, Markdown, PNG 세 개를 생성한다."""

    application = Path(application_path)
    assignment = Path(assignment_path)
    json_output = Path(json_path)
    report_output = Path(report_path)
    figures_output = Path(figures_dir)

    _check_output_paths(
        application,
        assignment,
        json_output,
        report_output,
        figures_output,
    )
    assignments = _read_and_validate_assignments(assignment)
    application_sha256 = _sha256(application)
    assignment_sha256 = _sha256(assignment)
    _validate_application_contract(application, assignments)
    train = _load_train_features(application, assignments, chunk_size=chunk_size)
    result, features, numeric_columns = _build_statistics(
        train,
        application_name=application.name,
        assignment_name=assignment.name,
        application_sha256=application_sha256,
        assignment_sha256=assignment_sha256,
        correlation_sample_size=correlation_sample_size,
        max_correlation_columns=max_correlation_columns,
        top_categories=top_categories,
    )
    if _sha256(application) != application_sha256:
        raise Stage2EDAError("EDA 실행 중 원본 신청 데이터가 변경되었습니다.")
    if _sha256(assignment) != assignment_sha256:
        raise Stage2EDAError("EDA 실행 중 고객 분할표가 변경되었습니다.")
    markdown = render_markdown_report(result)
    _assert_output_safety(result, markdown)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_output.write_text(markdown, encoding="utf-8")
    _write_figures(train, features, numeric_columns, figures_output)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CreditLens Stage 2 train 파티션 전용 EDA를 실행합니다."
    )
    parser.add_argument(
        "--application",
        type=Path,
        default=Path("data/raw/application_train.csv"),
        help="application_train.csv 경로",
    )
    parser.add_argument(
        "--assignment",
        type=Path,
        default=Path("data/interim/customer_splits.csv"),
        help="고객 분할표 경로",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=Path("reports/stage2_eda.json"),
        help="집계 JSON 출력 경로",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/Stage2_EDA_Report.md"),
        help="한국어 Markdown 보고서 출력 경로",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
        help="PNG 그래프 출력 디렉터리",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="application CSV 청크 행 수",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_stage2_eda(
        application_path=args.application,
        assignment_path=args.assignment,
        json_path=args.json_path,
        report_path=args.report,
        figures_dir=args.figures_dir,
        chunk_size=args.chunk_size,
    )
    print(
        "Stage 2 train-only EDA 완료: "
        f"{result['analysis_scope']['train_rows']:,}행, "
        f"{result['analysis_scope']['feature_columns']:,}개 피처"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
