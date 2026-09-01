"""Stage 3 파생 피처를 학습 파티션에서만 프로파일링한다.

V3 분석 마트 전체에서는 고객 키, TARGET, split 계약만 검증한다. 실제 피처
분포와 TARGET 관계는 ``SPLIT='train'`` 행만 조회하여 계산하며 고객별 값은
공유용 JSON에 기록하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd

from creditlens.data.build_feature_mart import (
    BUREAU_FEATURES,
    INSTALLMENT_FEATURES,
    V1_DERIVED_FEATURES,
)


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
SPLIT_COLUMN = "SPLIT"
ALLOWED_SPLITS = ("train", "validation", "test")
FEATURE_GROUPS = {
    "application": V1_DERIVED_FEATURES,
    "bureau": BUREAU_FEATURES,
    "installments": INSTALLMENT_FEATURES,
}
PROFILE_FEATURES = tuple(
    feature for features in FEATURE_GROUPS.values() for feature in features
)
DEFAULT_INPUT = Path("data/processed/feature_mart_v3.parquet")
DEFAULT_OUTPUT = Path("reports/stage3_feature_profile.json")
SCHEMA_VERSION = "1.0"


class Stage3FeatureProfileError(ValueError):
    """마트 계약 또는 train-only 분석 경계가 깨졌을 때 발생한다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _safe_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return path.name
    return relative.as_posix() if ".." not in relative.parts else path.name


def _finite(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _feature_family(feature: str) -> str:
    if (
        feature.endswith("_HAS_HISTORY")
        or feature.endswith("_SENTINEL")
        or feature.endswith("_NOT_APPLICABLE")
        or feature.endswith("_MISSING")
    ):
        return "binary_flag"
    if "COUNT" in feature:
        return "count"
    if "RATIO" in feature:
        return "ratio"
    if "AMOUNT" in feature or "INCOME_PER" in feature:
        return "amount"
    if "DAYS" in feature or "AGE" in feature or "YEARS" in feature:
        return "duration"
    return "continuous"


def _validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise Stage3FeatureProfileError(
            f"V3 분석 마트를 찾을 수 없습니다: {input_path.name}"
        )
    if input_path.suffix.lower() != ".parquet":
        raise Stage3FeatureProfileError("입력 분석 마트는 Parquet 파일이어야 합니다.")
    if output_path.suffix.lower() != ".json":
        raise Stage3FeatureProfileError("프로파일 출력은 JSON 파일이어야 합니다.")
    if input_path.resolve() == output_path.resolve():
        raise Stage3FeatureProfileError("프로파일 출력이 입력 마트를 덮어쓸 수 없습니다.")


def _validate_contract(
    connection: duckdb.DuckDBPyConnection, relation: str
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    described = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    schema = {str(row[0]): str(row[1]) for row in described}
    required = {ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN, *PROFILE_FEATURES}
    missing = sorted(required - set(schema))
    if missing:
        raise Stage3FeatureProfileError(
            f"V3 분석 마트 필수 컬럼이 없습니다: {', '.join(missing)}"
        )

    non_numeric = sorted(
        feature
        for feature in PROFILE_FEATURES
        if not any(
            token in schema[feature]
            for token in (
                "INT",
                "FLOAT",
                "DOUBLE",
                "DECIMAL",
                "BOOLEAN",
            )
        )
    )
    if non_numeric:
        raise Stage3FeatureProfileError(
            f"Stage 3 파생 피처는 수치형이어야 합니다: {', '.join(non_numeric)}"
        )

    metrics = connection.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT {ID_COLUMN}) AS unique_customers,
            COUNT(*) FILTER (WHERE {ID_COLUMN} IS NULL) AS missing_ids,
            COUNT(*) FILTER (
                WHERE {TARGET_COLUMN} IS NULL OR {TARGET_COLUMN} NOT IN (0, 1)
            ) AS invalid_targets,
            COUNT(*) FILTER (
                WHERE {SPLIT_COLUMN} IS NULL
                   OR {SPLIT_COLUMN} NOT IN {ALLOWED_SPLITS}
            ) AS invalid_splits,
            COUNT(DISTINCT {SPLIT_COLUMN}) AS split_count
        FROM {relation}
        """
    ).fetchone()
    if (
        not metrics[0]
        or metrics[0] != metrics[1]
        or metrics[2]
        or metrics[3]
        or metrics[4]
        or metrics[5] != len(ALLOWED_SPLITS)
    ):
        raise Stage3FeatureProfileError("V3 고객 키·TARGET·split 계약이 깨졌습니다.")

    split_counts = {
        split: {"0": 0, "1": 0, "rows": 0} for split in ALLOWED_SPLITS
    }
    rows = connection.execute(
        f"""
        SELECT {SPLIT_COLUMN}, {TARGET_COLUMN}, COUNT(*)
        FROM {relation}
        GROUP BY {SPLIT_COLUMN}, {TARGET_COLUMN}
        ORDER BY {SPLIT_COLUMN}, {TARGET_COLUMN}
        """
    ).fetchall()
    for split, target, count in rows:
        value = int(count)
        split_counts[str(split)][str(int(target))] = value
        split_counts[str(split)]["rows"] += value
    return schema, split_counts


def _profile_feature(
    frame: pd.DataFrame, feature: str, data_type: str
) -> dict[str, Any]:
    values = pd.to_numeric(frame[feature], errors="coerce")
    observed = values.dropna()
    observed_count = int(observed.size)
    missing_count = int(values.isna().sum())
    unique_count = int(observed.nunique())
    quantiles = (
        observed.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        if observed_count
        else pd.Series(dtype=float)
    )
    most_frequent_rate = 0.0
    if observed_count:
        most_frequent_rate = _ratio(int(observed.value_counts().iloc[0]), observed_count)

    by_target: dict[str, dict[str, Any]] = {}
    target_means: dict[int, float | None] = {}
    for target in (0, 1):
        target_values = values.loc[frame[TARGET_COLUMN].eq(target)]
        target_observed = target_values.dropna()
        count = int(target_observed.size)
        target_means[target] = _finite(target_observed.mean()) if count else None
        by_target[str(target)] = {
            "rows": int(target_values.size),
            "observed_count": count,
            "missing_rate": _ratio(int(target_values.isna().sum()), int(target_values.size)),
            "mean": target_means[target],
            "median": _finite(target_observed.median()) if count else None,
        }

    standardised_mean_difference: float | None = None
    standard_deviation = _finite(observed.std()) if observed_count > 1 else None
    if (
        standard_deviation is not None
        and standard_deviation > 0
        and target_means[0] is not None
        and target_means[1] is not None
    ):
        standardised_mean_difference = (
            target_means[1] - target_means[0]
        ) / standard_deviation

    minimum = _finite(observed.min()) if observed_count else None
    median = _finite(quantiles.get(0.5))
    p99 = _finite(quantiles.get(0.99))
    right_skew_candidate = bool(
        unique_count > 2
        and minimum is not None
        and minimum >= 0
        and p99 is not None
        and median is not None
        and p99 > max(1.0, median * 10)
    )
    return {
        "source_type": data_type,
        "family": _feature_family(feature),
        "observed_count": observed_count,
        "missing_count": missing_count,
        "missing_rate": _ratio(missing_count, len(frame)),
        "unique_count": unique_count,
        "zero_count": int(observed.eq(0).sum()),
        "zero_rate_observed": _ratio(int(observed.eq(0).sum()), observed_count),
        "most_frequent_rate_observed": most_frequent_rate,
        "mean": _finite(observed.mean()) if observed_count else None,
        "std": standard_deviation,
        "min": minimum,
        "p1": _finite(quantiles.get(0.01)),
        "q1": _finite(quantiles.get(0.25)),
        "median": median,
        "q3": _finite(quantiles.get(0.75)),
        "p99": p99,
        "max": _finite(observed.max()) if observed_count else None,
        "by_target": by_target,
        "target_1_minus_0_standardised_mean": _finite(
            standardised_mean_difference
        ),
        "missing_indicator_candidate": missing_count > 0,
        "nonnegative_log1p_candidate": right_skew_candidate,
    }


def profile_stage3_features(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """V3의 Stage 3 파생 피처를 train 행으로만 분석하고 JSON을 저장한다."""

    source = Path(input_path)
    output = Path(output_path)
    _validate_paths(source, output)
    relation = f"read_parquet({_sql_literal(source)})"

    with duckdb.connect(database=":memory:") as connection:
        schema, split_counts = _validate_contract(connection, relation)
        projected = ", ".join(
            [_sql_identifier(TARGET_COLUMN)]
            + [_sql_identifier(feature) for feature in PROFILE_FEATURES]
        )
        train = connection.execute(
            f"""
            SELECT {projected}
            FROM {relation}
            WHERE {SPLIT_COLUMN} = 'train'
            """
        ).fetchdf()

    if len(train) != split_counts["train"]["rows"] or train.empty:
        raise Stage3FeatureProfileError("train 피처 적재 행 수가 split 계약과 다릅니다.")
    if not train[TARGET_COLUMN].isin([0, 1]).all():
        raise Stage3FeatureProfileError("train TARGET은 0 또는 1이어야 합니다.")

    statistics = {
        feature: _profile_feature(train, feature, schema[feature])
        for feature in PROFILE_FEATURES
    }
    missingness_ranking = sorted(
        (
            {"feature": feature, "missing_rate": stats["missing_rate"]}
            for feature, stats in statistics.items()
        ),
        key=lambda item: (-item["missing_rate"], item["feature"]),
    )
    signal_ranking = sorted(
        (
            {
                "feature": feature,
                "standardised_mean_difference": stats[
                    "target_1_minus_0_standardised_mean"
                ],
            }
            for feature, stats in statistics.items()
            if stats["target_1_minus_0_standardised_mean"] is not None
        ),
        key=lambda item: (
            -abs(float(item["standardised_mean_difference"])),
            item["feature"],
        ),
    )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "environment": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "pandas": pd.__version__,
        },
        "source": {
            "display_path": _safe_display_path(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "analysis_scope": {
            "statistics_split": "train",
            "train_feature_rows_used": len(train),
            "validation_feature_rows_used": 0,
            "test_feature_rows_used": 0,
            "split_counts_metadata_only": split_counts,
            "profiled_feature_count": len(PROFILE_FEATURES),
            "feature_groups": {
                name: len(features) for name, features in FEATURE_GROUPS.items()
            },
        },
        "feature_statistics": statistics,
        "rankings": {
            "missingness_top15": missingness_ranking[:15],
            "absolute_standardised_mean_difference_top15": signal_ranking[:15],
        },
        "processing_decisions": {
            "history_absence": (
                "이력 없음은 HAS_HISTORY=0과 건수=0으로 보존하고, 금액·평균·비율의 "
                "NULL을 임의의 0으로 바꾸지 않는다."
            ),
            "missing_values": (
                "Stage 4 전처리기는 train에서만 대치값을 학습하고 결측 플래그를 함께 비교한다."
            ),
            "skewed_nonnegative_features": (
                "비음수 오른쪽 꼬리 피처는 선형 모델에서 log1p 후보로 비교하고 트리 모델은 원값을 유지한다."
            ),
            "outliers": (
                "행을 삭제하지 않으며 선형 모델의 clipping 경계가 필요하면 train 분위수로만 정한다."
            ),
            "ratios": (
                "업무상 1을 넘을 수 있는 납부비율을 포함하므로 일괄적으로 0~1 범위에 자르지 않는다."
            ),
            "univariate_relationships": (
                "TARGET별 평균 차이는 탐색 신호이며 인과관계나 단독 피처 선택 기준으로 해석하지 않는다."
            ),
        },
        "invariants": {
            "customer_values_excluded_from_output": True,
            "identifier_not_loaded_with_features": True,
            "non_train_feature_rows_used": 0,
            "processing_statistics_fit_on_train_only": True,
        },
    }

    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if str(source.resolve()) in serialized:
        raise Stage3FeatureProfileError("공유용 프로파일에 절대경로가 포함되었습니다.")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".json.tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temp)
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 3 파생 피처를 train 파티션에서만 프로파일링합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = profile_stage3_features(args.input, args.output)
    scope = result["analysis_scope"]
    print(
        "Stage 3 train-only 피처 프로파일 완료: "
        f"{scope['train_feature_rows_used']:,}행, "
        f"{scope['profiled_feature_count']}개 피처"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
