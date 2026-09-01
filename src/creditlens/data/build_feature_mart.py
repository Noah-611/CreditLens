"""Stage 3 고객 단위 V1·V2·V3 분석 마트를 재현 가능하게 구축한다.

Python은 입력 계약, 실행 순서, 원본 불변성과 산출물 검증을 담당한다.
대용량 CSV 집계와 Parquet 생성은 DuckDB SQL로 수행한다. 고객별 데이터와
모델 입력 파일은 ``data/processed``에만 저장하고 공유용 JSON에는 고객 ID나
절대경로를 기록하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
SPLIT_COLUMN = "SPLIT"
ALLOWED_SPLITS = ("train", "validation", "test")
MODEL_EXCLUDED_COLUMNS = ("CODE_GENDER",)
DAYS_EMPLOYED_SENTINEL = 365_243
SCHEMA_VERSION = "1.0"
BUILD_VERSION = "stage3-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_APPLICATION_INPUT = Path("data/raw/application_train.csv")
DEFAULT_BUREAU_INPUT = Path("data/raw/bureau.csv")
DEFAULT_INSTALLMENTS_INPUT = Path("data/raw/installments_payments.csv")
DEFAULT_SPLITS_INPUT = Path("data/interim/customer_splits.csv")
DEFAULT_V1_OUTPUT = Path("data/processed/feature_mart_v1.parquet")
DEFAULT_V2_OUTPUT = Path("data/processed/feature_mart_v2.parquet")
DEFAULT_V3_OUTPUT = Path("data/processed/feature_mart_v3.parquet")
DEFAULT_SUMMARY_OUTPUT = Path("reports/stage3_build_summary.json")
DEFAULT_SQL_DIR = PROJECT_ROOT / "sql/stage3"
DEFAULT_TEMP_DIR = Path("data/interim/duckdb_tmp")
DEFAULT_MEMORY_LIMIT = "3GB"
DEFAULT_THREADS = 2

SQL_FILES = {
    "v1": "01_v1_application.sql",
    "bureau": "02_bureau_features.sql",
    "v2": "03_v2_bureau.sql",
    "installments": "04_installment_features.sql",
    "v3": "05_v3_installments.sql",
}

APPLICATION_REQUIRED_COLUMNS = {
    ID_COLUMN,
    TARGET_COLUMN,
    "FLAG_OWN_CAR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OWN_CAR_AGE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
}
BUREAU_REQUIRED_COLUMNS = {
    ID_COLUMN,
    "SK_ID_BUREAU",
    "CREDIT_ACTIVE",
    "CREDIT_CURRENCY",
    "CREDIT_TYPE",
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE",
    "DAYS_CREDIT_UPDATE",
    "AMT_ANNUITY",
}
INSTALLMENTS_REQUIRED_COLUMNS = {
    "SK_ID_PREV",
    ID_COLUMN,
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
}
SPLIT_REQUIRED_COLUMNS = {ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}

V1_DERIVED_FEATURES = (
    "DAYS_EMPLOYED_SENTINEL",
    "OWN_CAR_AGE_NOT_APPLICABLE",
    "OWN_CAR_AGE_MISSING",
    "APP_CREDIT_INCOME_RATIO",
    "APP_ANNUITY_INCOME_RATIO",
    "APP_CREDIT_ANNUITY_RATIO",
    "APP_CREDIT_GOODS_RATIO",
    "APP_INCOME_PER_FAMILY_MEMBER",
    "APP_AGE_YEARS",
    "APP_EMPLOYED_YEARS",
    "APP_EMPLOYED_AGE_RATIO",
    "APP_EXT_SOURCE_OBSERVED_COUNT",
    "APP_EXT_SOURCE_MEAN",
)
BUREAU_FEATURES = (
    "BUREAU_HAS_HISTORY",
    "BUREAU_RECORD_COUNT",
    "BUREAU_LOAN_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_CLOSED_COUNT",
    "BUREAU_SOLD_COUNT",
    "BUREAU_BAD_DEBT_COUNT",
    "BUREAU_CREDIT_TYPE_COUNT",
    "BUREAU_NON_PRIMARY_CURRENCY_COUNT",
    "BUREAU_OVERDUE_LOAN_COUNT",
    "BUREAU_ACTIVE_RATIO",
    "BUREAU_OVERDUE_LOAN_RATIO",
    "BUREAU_DAYS_CREDIT_MEAN",
    "BUREAU_DAYS_CREDIT_MIN",
    "BUREAU_DAYS_CREDIT_MAX",
    "BUREAU_DAYS_SINCE_RECENT_CREDIT",
    "BUREAU_DAYS_OVERDUE_MEAN",
    "BUREAU_DAYS_OVERDUE_MAX",
    "BUREAU_PROLONG_COUNT_SUM",
    "BUREAU_CREDIT_AMOUNT_OBSERVED_COUNT",
    "BUREAU_CREDIT_AMOUNT_SUM",
    "BUREAU_CREDIT_AMOUNT_MEAN",
    "BUREAU_CREDIT_AMOUNT_MAX",
    "BUREAU_DEBT_OBSERVED_COUNT",
    "BUREAU_DEBT_SUM",
    "BUREAU_DEBT_MEAN",
    "BUREAU_DEBT_MAX",
    "BUREAU_OVERDUE_AMOUNT_SUM",
    "BUREAU_OVERDUE_AMOUNT_MAX",
    "BUREAU_MAX_OVERDUE_OBSERVED_COUNT",
    "BUREAU_MAX_OVERDUE_AMOUNT",
    "BUREAU_ACTIVE_CREDIT_SUM",
    "BUREAU_ACTIVE_DEBT_SUM",
    "BUREAU_DEBT_CREDIT_RATIO",
    "BUREAU_ANNUITY_OBSERVED_COUNT",
    "BUREAU_ANNUITY_SUM",
    "BUREAU_ANNUITY_MEAN",
)
INSTALLMENT_FEATURES = (
    "INST_HAS_HISTORY",
    "INST_SCHEDULE_COUNT",
    "INST_PREV_LOAN_COUNT",
    "INST_PAYMENT_EVENT_COUNT",
    "INST_PAYMENT_DATE_OBSERVED_SCHEDULE_COUNT",
    "INST_PAYMENT_AMOUNT_OBSERVED_SCHEDULE_COUNT",
    "INST_MISSING_PAYMENT_SCHEDULE_COUNT",
    "INST_MISSING_PAYMENT_RATIO",
    "INST_LATE_SCHEDULE_COUNT",
    "INST_LATE_RATIO",
    "INST_DAYS_LATE_MEAN",
    "INST_DAYS_LATE_MAX",
    "INST_DAYS_LATE_SUM",
    "INST_UNDERPAID_SCHEDULE_COUNT",
    "INST_UNDERPAID_RATIO",
    "INST_SCHEDULED_AMOUNT_SUM",
    "INST_PAID_AMOUNT_SUM",
    "INST_PAYMENT_GAP_SUM",
    "INST_PAYMENT_GAP_MAX",
    "INST_PAYMENT_RATIO",
    "INST_DAYS_SINCE_RECENT_DUE",
    "INST_OLDEST_DUE_AGE_DAYS",
    "INST_HISTORY_SPAN_DAYS",
    "INST_LAST_365_SCHEDULE_COUNT",
    "INST_LAST_365_LATE_COUNT",
    "INST_LAST_365_LATE_RATIO",
    "INST_LAST_730_SCHEDULE_COUNT",
    "INST_LAST_730_LATE_COUNT",
    "INST_LAST_730_LATE_RATIO",
)


class FeatureMartError(ValueError):
    """Stage 3 입력·집계·산출물 계약을 만족하지 못할 때 발생한다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return path.name
    if ".." in relative.parts:
        return path.name
    return relative.as_posix()


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _validate_runtime_settings(memory_limit: str, threads: int) -> tuple[str, int]:
    normalised_memory = memory_limit.strip().upper()
    if re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", normalised_memory) is None:
        raise FeatureMartError("memory_limit은 512MB 또는 4GB와 같은 형식이어야 합니다.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise FeatureMartError("threads는 1 이상의 정수여야 합니다.")
    return normalised_memory, threads


def _load_sql(sql_dir: Path) -> dict[str, str]:
    statements: dict[str, str] = {}
    for name, filename in SQL_FILES.items():
        path = sql_dir / filename
        if not path.is_file():
            raise FeatureMartError(f"Stage 3 SQL 파일을 찾을 수 없습니다: {filename}")
        statement = path.read_text(encoding="utf-8").strip().rstrip(";")
        if not statement:
            raise FeatureMartError(f"Stage 3 SQL 파일이 비어 있습니다: {filename}")
        statements[name] = statement
    return statements


def _validate_paths(
    sources: Mapping[str, Path],
    outputs: Mapping[str, Path],
    sql_dir: Path,
) -> None:
    for label, path in sources.items():
        if not path.is_file():
            raise FeatureMartError(f"{label} 입력 파일을 찾을 수 없습니다: {path.name}")
    if not sql_dir.is_dir():
        raise FeatureMartError("Stage 3 SQL 디렉터리를 찾을 수 없습니다.")

    source_paths = {path.resolve() for path in sources.values()}
    output_paths = [path.resolve() for path in outputs.values()]
    if len(set(output_paths)) != len(output_paths):
        raise FeatureMartError("Stage 3 출력 경로는 서로 달라야 합니다.")
    if source_paths.intersection(output_paths):
        raise FeatureMartError("Stage 3 출력은 원본·분할 입력 파일을 덮어쓸 수 없습니다.")
    for label, path in outputs.items():
        expected_suffix = ".json" if label == "summary" else ".parquet"
        if path.suffix.lower() != expected_suffix:
            raise FeatureMartError(f"{label} 출력 확장자는 {expected_suffix}여야 합니다.")


def _create_csv_view(connection: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    connection.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto({_sql_literal(path)}, "
        "header = true, sample_size = 100000)"
    )


def _column_schema(
    connection: duckdb.DuckDBPyConnection, relation: str
) -> list[tuple[str, str]]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def _column_names(connection: duckdb.DuckDBPyConnection, relation: str) -> list[str]:
    return [name for name, _data_type in _column_schema(connection, relation)]


def _require_columns(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    required: set[str],
    label: str,
) -> None:
    available = set(_column_names(connection, relation))
    missing = sorted(required - available)
    if missing:
        raise FeatureMartError(f"{label} 필수 컬럼이 없습니다: {', '.join(missing)}")


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def _process_peak_rss_mb() -> float:
    """현재 프로세스 수명 동안 관측된 최대 RSS를 MiB로 반환한다."""

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        peak /= 1024
    return peak / 1024


def _validate_sources(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    _require_columns(
        connection,
        "application_source",
        APPLICATION_REQUIRED_COLUMNS,
        "application_train",
    )
    _require_columns(connection, "bureau_source", BUREAU_REQUIRED_COLUMNS, "bureau")
    _require_columns(
        connection,
        "installments_source",
        INSTALLMENTS_REQUIRED_COLUMNS,
        "installments_payments",
    )
    _require_columns(connection, "split_source", SPLIT_REQUIRED_COLUMNS, "고객 분할표")

    application = connection.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT {ID_COLUMN}) AS unique_ids,
            COUNT(*) FILTER (WHERE {ID_COLUMN} IS NULL) AS missing_ids,
            COUNT(*) FILTER (WHERE {TARGET_COLUMN} IS NULL OR {TARGET_COLUMN} NOT IN (0, 1))
                AS invalid_targets,
            COUNT(*) FILTER (WHERE DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL})
                AS employed_sentinel_rows
        FROM application_source
        """
    ).fetchone()
    if application[0] == 0:
        raise FeatureMartError("application_train에 데이터 행이 없습니다.")
    if application[0] != application[1] or application[2] or application[3]:
        raise FeatureMartError("application_train의 고객 키 또는 TARGET 계약이 깨졌습니다.")

    split = connection.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT {ID_COLUMN}) AS unique_ids,
            COUNT(*) FILTER (WHERE {ID_COLUMN} IS NULL) AS missing_ids,
            COUNT(*) FILTER (WHERE {TARGET_COLUMN} IS NULL OR {TARGET_COLUMN} NOT IN (0, 1))
                AS invalid_targets,
            COUNT(*) FILTER (
                WHERE {SPLIT_COLUMN} IS NULL
                   OR {SPLIT_COLUMN} NOT IN {ALLOWED_SPLITS}
            )
                AS invalid_splits,
            COUNT(DISTINCT {SPLIT_COLUMN}) AS split_count
        FROM split_source
        """
    ).fetchone()
    if split[0] != split[1] or split[2] or split[3] or split[4] or split[5] != 3:
        raise FeatureMartError("고객 분할표의 키·TARGET·split 계약이 깨졌습니다.")
    if split[0] != application[0]:
        raise FeatureMartError("application_train과 고객 분할표의 행 수가 다릅니다.")

    assignment_mismatches = _scalar(
        connection,
        f"""
        SELECT COUNT(*)
        FROM application_source AS a
        FULL OUTER JOIN split_source AS s USING ({ID_COLUMN})
        WHERE a.{ID_COLUMN} IS NULL
           OR s.{ID_COLUMN} IS NULL
           OR a.{TARGET_COLUMN} <> s.{TARGET_COLUMN}
        """,
    )
    if assignment_mismatches:
        raise FeatureMartError("고객 분할표가 application_train의 고객·TARGET을 보존하지 않습니다.")

    bureau = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(*) FILTER (WHERE SK_ID_CURR IS NULL OR SK_ID_BUREAU IS NULL)
                AS missing_keys,
            COUNT(DISTINCT SK_ID_BUREAU) AS unique_loan_ids,
            COUNT(*) FILTER (WHERE DAYS_CREDIT > 0) AS future_credit_rows,
            COUNT(*) FILTER (WHERE DAYS_CREDIT_UPDATE > 0) AS positive_update_rows
        FROM bureau_source
        """
    ).fetchone()
    if bureau[1] or bureau[0] != bureau[2]:
        raise FeatureMartError("bureau의 고객·대출 키 계약이 깨졌습니다.")
    if bureau[3]:
        raise FeatureMartError("bureau에 현재 신청 이후 생성된 DAYS_CREDIT 행이 있습니다.")

    installments = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(*) FILTER (WHERE SK_ID_CURR IS NULL OR SK_ID_PREV IS NULL)
                AS missing_keys,
            COUNT(*) FILTER (WHERE DAYS_INSTALMENT > 0) AS future_due_rows,
            COUNT(*) FILTER (
                WHERE DAYS_ENTRY_PAYMENT IS NOT NULL AND DAYS_ENTRY_PAYMENT > 0
            ) AS future_payment_rows,
            COUNT(*) FILTER (
                WHERE (DAYS_ENTRY_PAYMENT IS NULL) <> (AMT_PAYMENT IS NULL)
            ) AS mismatched_payment_missingness
        FROM installments_source
        """
    ).fetchone()
    if installments[1]:
        raise FeatureMartError("installments_payments의 고객·대출 키에 결측값이 있습니다.")
    if installments[2] or installments[3]:
        raise FeatureMartError("installments_payments에 현재 신청 이후 납부 행이 있습니다.")
    if installments[4]:
        raise FeatureMartError(
            "installments_payments의 납부일·납부금액 결측 상태가 일치하지 않습니다."
        )

    previous_loan_owner_violations = _scalar(
        connection,
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT SK_ID_PREV
            FROM installments_source
            GROUP BY SK_ID_PREV
            HAVING COUNT(DISTINCT {ID_COLUMN}) > 1
        )
        """,
    )
    if previous_loan_owner_violations:
        raise FeatureMartError("한 SK_ID_PREV가 여러 고객에 연결되어 있습니다.")

    inconsistent_schedules = _scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                SK_ID_CURR,
                SK_ID_PREV,
                NUM_INSTALMENT_VERSION,
                NUM_INSTALMENT_NUMBER
            FROM installments_source
            GROUP BY
                SK_ID_CURR,
                SK_ID_PREV,
                NUM_INSTALMENT_VERSION,
                NUM_INSTALMENT_NUMBER
            HAVING COUNT(DISTINCT DAYS_INSTALMENT) > 1
                OR COUNT(DISTINCT AMT_INSTALMENT) > 1
        )
        """,
    )
    if inconsistent_schedules:
        raise FeatureMartError("같은 할부 회차에 서로 다른 예정일 또는 예정금액이 있습니다.")

    return {
        "application_rows": int(application[0]),
        "application_unique_customers": int(application[1]),
        "days_employed_sentinel_rows": int(application[4]),
        "split_rows": int(split[0]),
        "assignment_mismatches": assignment_mismatches,
        "bureau_rows": int(bureau[0]),
        "bureau_future_credit_rows": int(bureau[3]),
        "bureau_positive_days_credit_update_rows_excluded": int(bureau[4]),
        "installment_rows": int(installments[0]),
        "installment_future_due_rows": int(installments[2]),
        "installment_future_payment_rows": int(installments[3]),
        "installment_mismatched_payment_missingness": int(installments[4]),
        "installment_owner_violations": previous_loan_owner_violations,
        "installment_inconsistent_schedule_groups": inconsistent_schedules,
    }


def _temporary_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    temporary.unlink()
    return temporary


def _copy_parquet(
    connection: duckdb.DuckDBPyConnection, query: str, destination: Path
) -> None:
    connection.execute(
        f"COPY ({query}) TO {_sql_literal(destination)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )


def _lineage_mismatch_count(
    connection: duckdb.DuckDBPyConnection,
    lower_path: Path,
    higher_path: Path,
    columns: Sequence[str],
) -> int:
    projected = ", ".join(_sql_identifier(column) for column in columns)
    return _scalar(
        connection,
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT {projected} FROM read_parquet({_sql_literal(lower_path)})
            EXCEPT ALL
            SELECT {projected} FROM read_parquet({_sql_literal(higher_path)})
        )
        """,
    )


def _split_counts(
    connection: duckdb.DuckDBPyConnection, relation: str
) -> dict[str, dict[str, int]]:
    rows = connection.execute(
        f"""
        SELECT {SPLIT_COLUMN}, {TARGET_COLUMN}, COUNT(*)
        FROM {relation}
        GROUP BY {SPLIT_COLUMN}, {TARGET_COLUMN}
        ORDER BY {SPLIT_COLUMN}, {TARGET_COLUMN}
        """
    ).fetchall()
    counts = {
        split: {"0": 0, "1": 0, "rows": 0} for split in ALLOWED_SPLITS
    }
    for split, target, count in rows:
        name = str(split)
        value = int(count)
        counts[name][str(int(target))] = value
        counts[name]["rows"] += value
    return counts


def _validate_output(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    version: str,
    required_features: Sequence[str],
) -> dict[str, Any]:
    relation = f"read_parquet({_sql_literal(path)})"
    schema = _column_schema(connection, relation)
    columns = [name for name, _data_type in schema]
    types = dict(schema)
    if len(columns) != len(set(columns)):
        raise FeatureMartError(f"{version}에 중복 컬럼명이 있습니다.")
    required_metadata = {ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}
    if not required_metadata.issubset(columns):
        raise FeatureMartError(f"{version}에 고객 ID·TARGET·SPLIT이 모두 필요합니다.")
    missing_features = sorted(set(required_features) - set(columns))
    if missing_features:
        raise FeatureMartError(
            f"{version} 필수 피처가 없습니다: {', '.join(missing_features)}"
        )

    metrics = connection.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT {ID_COLUMN}) AS unique_ids,
            COUNT(*) FILTER (WHERE {ID_COLUMN} IS NULL) AS missing_ids,
            COUNT(*) FILTER (WHERE {TARGET_COLUMN} IS NULL OR {TARGET_COLUMN} NOT IN (0, 1))
                AS invalid_targets,
            COUNT(*) FILTER (
                WHERE {SPLIT_COLUMN} IS NULL
                   OR {SPLIT_COLUMN} NOT IN {ALLOWED_SPLITS}
            )
                AS invalid_splits,
            SUM(CAST({TARGET_COLUMN} AS BIGINT)) AS target_1_rows
        FROM {relation}
        """
    ).fetchone()
    expected_rows = _scalar(connection, "SELECT COUNT(*) FROM split_source")
    if (
        metrics[0] != expected_rows
        or metrics[0] != metrics[1]
        or metrics[2]
        or metrics[3]
        or metrics[4]
    ):
        raise FeatureMartError(f"{version}의 고객 키·TARGET·SPLIT 계약이 깨졌습니다.")

    mismatches = _scalar(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {relation} AS o
        FULL OUTER JOIN split_source AS s USING ({ID_COLUMN})
        WHERE o.{ID_COLUMN} IS NULL
           OR s.{ID_COLUMN} IS NULL
           OR o.{TARGET_COLUMN} <> s.{TARGET_COLUMN}
           OR o.{SPLIT_COLUMN} <> s.{SPLIT_COLUMN}
        """,
    )
    if mismatches:
        raise FeatureMartError(f"{version}가 원본 고객의 TARGET 또는 SPLIT을 변경했습니다.")

    schema_hash = hashlib.sha256(
        "\n".join(f"{name}\t{data_type}" for name, data_type in schema).encode(
            "utf-8"
        )
    ).hexdigest()
    policy_excluded = sorted(set(MODEL_EXCLUDED_COLUMNS).intersection(columns))
    candidate_feature_count = len(columns) - len(required_metadata)
    result: dict[str, Any] = {
        "rows": int(metrics[0]),
        "columns": len(columns),
        "candidate_feature_columns": candidate_feature_count,
        "policy_excluded_columns": policy_excluded,
        "model_feature_columns": candidate_feature_count - len(policy_excluded),
        "target_1_rows": int(metrics[5]),
        "split_counts": _split_counts(connection, relation),
        "schema_sha256": schema_hash,
        "schema_hash_includes_types": True,
        "customer_key_unique": True,
        "target_and_split_preserved": True,
    }
    for flag, key in (
        ("BUREAU_HAS_HISTORY", "customers_with_bureau_history"),
        ("INST_HAS_HISTORY", "customers_with_installment_history"),
    ):
        if flag in columns:
            result[key] = _scalar(
                connection,
                f"SELECT COUNT(*) FROM {relation} WHERE {flag} = 1",
            )

    if "BUREAU_HAS_HISTORY" in columns:
        bureau_metrics = connection.execute(
            f"""
            SELECT
                COALESCE(SUM(BUREAU_RECORD_COUNT), 0) AS rows_aggregated,
                COUNT(*) FILTER (
                    WHERE BUREAU_HAS_HISTORY = 0
                      AND (BUREAU_RECORD_COUNT <> 0 OR BUREAU_ACTIVE_RATIO IS NOT NULL)
                ) AS no_history_violations,
                COUNT(*) FILTER (
                    WHERE BUREAU_HAS_HISTORY = 1 AND BUREAU_RECORD_COUNT = 0
                ) AS history_flag_violations,
                COUNT(*) FILTER (
                    WHERE BUREAU_ACTIVE_COUNT > BUREAU_RECORD_COUNT
                       OR BUREAU_OVERDUE_LOAN_COUNT > BUREAU_RECORD_COUNT
                ) AS count_violations,
                COUNT(*) FILTER (
                    WHERE (BUREAU_ACTIVE_RATIO IS NOT NULL AND (
                            NOT isfinite(BUREAU_ACTIVE_RATIO)
                            OR BUREAU_ACTIVE_RATIO NOT BETWEEN 0 AND 1
                        ))
                       OR (BUREAU_OVERDUE_LOAN_RATIO IS NOT NULL AND (
                            NOT isfinite(BUREAU_OVERDUE_LOAN_RATIO)
                            OR BUREAU_OVERDUE_LOAN_RATIO NOT BETWEEN 0 AND 1
                        ))
                ) AS ratio_violations
            FROM {relation}
            """
        ).fetchone()
        expected_bureau_rows = _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM bureau_source AS b
            INNER JOIN application_ids AS a USING (SK_ID_CURR)
            WHERE b.DAYS_CREDIT_UPDATE <= 0
            """,
        )
        if int(bureau_metrics[0]) != expected_bureau_rows or any(bureau_metrics[1:]):
            raise FeatureMartError(f"{version}의 bureau 집계 계약이 깨졌습니다.")
        result["bureau_rows_aggregated"] = int(bureau_metrics[0])
        result["bureau_feature_contract_violations"] = 0

    if "INST_HAS_HISTORY" in columns:
        expected_double_columns = {
            "INST_SCHEDULED_AMOUNT_SUM",
            "INST_PAID_AMOUNT_SUM",
            "INST_PAYMENT_GAP_SUM",
            "INST_PAYMENT_GAP_MAX",
        }
        invalid_types = sorted(
            column for column in expected_double_columns if types.get(column) != "DOUBLE"
        )
        if invalid_types:
            raise FeatureMartError(
                f"{version} 금액 피처는 DOUBLE이어야 합니다: {', '.join(invalid_types)}"
            )
        installment_metrics = connection.execute(
            f"""
            SELECT
                COALESCE(SUM(INST_PAYMENT_EVENT_COUNT), 0) AS rows_aggregated,
                COALESCE(SUM(INST_SCHEDULE_COUNT), 0) AS schedules_aggregated,
                COUNT(*) FILTER (
                    WHERE INST_HAS_HISTORY = 0
                      AND (INST_SCHEDULE_COUNT <> 0 OR INST_LATE_RATIO IS NOT NULL)
                ) AS no_history_violations,
                COUNT(*) FILTER (
                    WHERE INST_HAS_HISTORY = 1 AND INST_SCHEDULE_COUNT = 0
                ) AS history_flag_violations,
                COUNT(*) FILTER (
                    WHERE INST_PAYMENT_DATE_OBSERVED_SCHEDULE_COUNT
                            > INST_SCHEDULE_COUNT
                       OR INST_PAYMENT_AMOUNT_OBSERVED_SCHEDULE_COUNT
                            > INST_SCHEDULE_COUNT
                       OR INST_PAYMENT_DATE_OBSERVED_SCHEDULE_COUNT
                            + INST_MISSING_PAYMENT_SCHEDULE_COUNT
                            <> INST_SCHEDULE_COUNT
                       OR INST_PAYMENT_AMOUNT_OBSERVED_SCHEDULE_COUNT
                            + INST_MISSING_PAYMENT_SCHEDULE_COUNT
                            <> INST_SCHEDULE_COUNT
                       OR INST_LATE_SCHEDULE_COUNT
                            > INST_PAYMENT_DATE_OBSERVED_SCHEDULE_COUNT
                       OR INST_UNDERPAID_SCHEDULE_COUNT
                            > INST_PAYMENT_AMOUNT_OBSERVED_SCHEDULE_COUNT
                       OR INST_LAST_365_SCHEDULE_COUNT > INST_LAST_730_SCHEDULE_COUNT
                       OR INST_LAST_730_SCHEDULE_COUNT > INST_SCHEDULE_COUNT
                       OR INST_LAST_365_LATE_COUNT > INST_LAST_365_SCHEDULE_COUNT
                       OR INST_LAST_730_LATE_COUNT > INST_LAST_730_SCHEDULE_COUNT
                ) AS count_violations,
                COUNT(*) FILTER (
                    WHERE (INST_MISSING_PAYMENT_RATIO IS NOT NULL AND (
                            NOT isfinite(INST_MISSING_PAYMENT_RATIO)
                            OR INST_MISSING_PAYMENT_RATIO NOT BETWEEN 0 AND 1
                        ))
                       OR (INST_LATE_RATIO IS NOT NULL AND (
                            NOT isfinite(INST_LATE_RATIO)
                            OR INST_LATE_RATIO NOT BETWEEN 0 AND 1
                        ))
                       OR (INST_UNDERPAID_RATIO IS NOT NULL AND (
                            NOT isfinite(INST_UNDERPAID_RATIO)
                            OR INST_UNDERPAID_RATIO NOT BETWEEN 0 AND 1
                        ))
                       OR (INST_LAST_365_LATE_RATIO IS NOT NULL AND (
                            NOT isfinite(INST_LAST_365_LATE_RATIO)
                            OR INST_LAST_365_LATE_RATIO NOT BETWEEN 0 AND 1
                        ))
                       OR (INST_LAST_730_LATE_RATIO IS NOT NULL AND (
                            NOT isfinite(INST_LAST_730_LATE_RATIO)
                            OR INST_LAST_730_LATE_RATIO NOT BETWEEN 0 AND 1
                        ))
                       OR (INST_PAYMENT_RATIO IS NOT NULL AND (
                            NOT isfinite(INST_PAYMENT_RATIO)
                            OR INST_PAYMENT_RATIO < 0
                        ))
                ) AS ratio_violations
            FROM {relation}
            """
        ).fetchone()
        expected_installment_rows = _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM installments_source AS i
            INNER JOIN application_ids AS a USING (SK_ID_CURR)
            """,
        )
        if (
            int(installment_metrics[0]) != expected_installment_rows
            or any(installment_metrics[2:])
        ):
            raise FeatureMartError(f"{version}의 installments 집계 계약이 깨졌습니다.")
        result["installment_payment_events_aggregated"] = int(
            installment_metrics[0]
        )
        result["installment_schedules_aggregated"] = int(installment_metrics[1])
        result["installment_feature_contract_violations"] = 0
    return result


def _source_metadata(path: Path, digest: str) -> dict[str, Any]:
    return {
        "display_path": _safe_display_path(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def build_feature_marts(
    application_path: str | Path = DEFAULT_APPLICATION_INPUT,
    bureau_path: str | Path = DEFAULT_BUREAU_INPUT,
    installments_path: str | Path = DEFAULT_INSTALLMENTS_INPUT,
    splits_path: str | Path = DEFAULT_SPLITS_INPUT,
    *,
    v1_output: str | Path = DEFAULT_V1_OUTPUT,
    v2_output: str | Path = DEFAULT_V2_OUTPUT,
    v3_output: str | Path = DEFAULT_V3_OUTPUT,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    sql_dir: str | Path = DEFAULT_SQL_DIR,
    temp_dir: str | Path = DEFAULT_TEMP_DIR,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
) -> dict[str, Any]:
    """원본을 수정하지 않고 V1·V2·V3 Parquet과 공유용 요약을 생성한다."""

    sources = {
        "application": Path(application_path),
        "bureau": Path(bureau_path),
        "installments": Path(installments_path),
        "splits": Path(splits_path),
    }
    outputs = {
        "v1": Path(v1_output),
        "v2": Path(v2_output),
        "v3": Path(v3_output),
        "summary": Path(summary_output),
    }
    sql_directory = Path(sql_dir)
    scratch = Path(temp_dir)
    memory_limit, threads = _validate_runtime_settings(memory_limit, threads)
    _validate_paths(sources, outputs, sql_directory)
    sql = _load_sql(sql_directory)

    source_stats_before = {
        name: (path.stat().st_size, path.stat().st_mtime_ns)
        for name, path in sources.items()
    }
    source_hashes = {name: _sha256(path) for name, path in sources.items()}
    sql_hashes = {
        SQL_FILES[name]: hashlib.sha256(statement.encode("utf-8")).hexdigest()
        for name, statement in sql.items()
    }

    scratch.mkdir(parents=True, exist_ok=True)
    temporary_outputs = {
        name: _temporary_output(outputs[name]) for name in ("v1", "v2", "v3")
    }
    started = time.perf_counter()
    durations: dict[str, float] = {}
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"SET threads = {threads}")
        connection.execute(f"SET memory_limit = {_sql_literal(memory_limit)}")
        connection.execute(f"SET temp_directory = {_sql_literal(scratch.resolve())}")
        connection.execute("SET preserve_insertion_order = false")

        _create_csv_view(connection, "application_source", sources["application"])
        _create_csv_view(connection, "bureau_source", sources["bureau"])
        _create_csv_view(connection, "installments_source", sources["installments"])
        _create_csv_view(connection, "split_source", sources["splits"])
        connection.execute(
            f"CREATE TEMP VIEW application_ids AS "
            f"SELECT {ID_COLUMN} FROM application_source"
        )

        phase = time.perf_counter()
        quality = _validate_sources(connection)
        durations["source_validation_seconds"] = time.perf_counter() - phase

        phase = time.perf_counter()
        _copy_parquet(connection, sql["v1"], temporary_outputs["v1"])
        connection.execute(
            "CREATE OR REPLACE VIEW v1_source AS "
            f"SELECT * FROM read_parquet({_sql_literal(temporary_outputs['v1'])})"
        )
        durations["v1_build_seconds"] = time.perf_counter() - phase

        phase = time.perf_counter()
        connection.execute(f"CREATE TEMP TABLE bureau_features AS {sql['bureau']}")
        _copy_parquet(connection, sql["v2"], temporary_outputs["v2"])
        connection.execute(
            "CREATE OR REPLACE VIEW v2_source AS "
            f"SELECT * FROM read_parquet({_sql_literal(temporary_outputs['v2'])})"
        )
        durations["v2_build_seconds"] = time.perf_counter() - phase

        phase = time.perf_counter()
        connection.execute(
            f"CREATE TEMP TABLE installment_features AS {sql['installments']}"
        )
        _copy_parquet(connection, sql["v3"], temporary_outputs["v3"])
        durations["v3_build_seconds"] = time.perf_counter() - phase

        phase = time.perf_counter()
        output_metrics = {
            "v1": _validate_output(
                connection,
                temporary_outputs["v1"],
                version="V1",
                required_features=V1_DERIVED_FEATURES,
            ),
            "v2": _validate_output(
                connection,
                temporary_outputs["v2"],
                version="V2",
                required_features=V1_DERIVED_FEATURES + BUREAU_FEATURES,
            ),
            "v3": _validate_output(
                connection,
                temporary_outputs["v3"],
                version="V3",
                required_features=(
                    V1_DERIVED_FEATURES + BUREAU_FEATURES + INSTALLMENT_FEATURES
                ),
            ),
        }
        output_columns = {
            name: _column_names(
                connection,
                f"read_parquet({_sql_literal(temporary_outputs[name])})",
            )
            for name in ("v1", "v2", "v3")
        }
        if (
            output_columns["v2"][: len(output_columns["v1"])]
            != output_columns["v1"]
            or output_columns["v3"][: len(output_columns["v2"])]
            != output_columns["v2"]
        ):
            raise FeatureMartError("V1→V2→V3 컬럼 계보가 보존되지 않았습니다.")
        lineage_mismatches = {
            "v1_to_v2": _lineage_mismatch_count(
                connection,
                temporary_outputs["v1"],
                temporary_outputs["v2"],
                output_columns["v1"],
            ),
            "v2_to_v3": _lineage_mismatch_count(
                connection,
                temporary_outputs["v2"],
                temporary_outputs["v3"],
                output_columns["v2"],
            ),
        }
        if any(lineage_mismatches.values()):
            raise FeatureMartError("V1→V2→V3 공통 컬럼 값이 변경되었습니다.")
        durations["output_validation_seconds"] = time.perf_counter() - phase
    except duckdb.Error as error:
        for temporary in temporary_outputs.values():
            temporary.unlink(missing_ok=True)
        raise FeatureMartError(f"DuckDB Stage 3 구축에 실패했습니다: {error}") from error
    except Exception:
        for temporary in temporary_outputs.values():
            temporary.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    source_stats_after = {
        name: (path.stat().st_size, path.stat().st_mtime_ns)
        for name, path in sources.items()
    }
    if source_stats_before != source_stats_after:
        for temporary in temporary_outputs.values():
            temporary.unlink(missing_ok=True)
        raise FeatureMartError("Stage 3 실행 중 입력 파일이 변경되어 출력을 중단했습니다.")

    summary_temporary: Path | None = None
    try:
        for name in ("v1", "v2", "v3"):
            output_metrics[name]["sha256"] = _sha256(temporary_outputs[name])
            output_metrics[name]["display_path"] = _safe_display_path(outputs[name])

        total_seconds = time.perf_counter() - started
        durations["total_seconds"] = total_seconds
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "build_version": BUILD_VERSION,
            "generated_at_utc": datetime.now(UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "environment": {
                "python": platform.python_version(),
                "duckdb": duckdb.__version__,
                "platform": platform.system(),
            },
            "settings": {
                "memory_limit": memory_limit,
                "threads": threads,
                "parquet_compression": "zstd",
            },
            "resources": {
                "process_peak_rss_mb": round(_process_peak_rss_mb(), 3),
                "measurement_scope": "current_process_lifetime",
            },
            "sources": {
                name: _source_metadata(path, source_hashes[name])
                for name, path in sources.items()
            },
            "sql_sha256": sql_hashes,
            "source_quality": quality,
            "outputs": output_metrics,
            "lineage_mismatch_rows": lineage_mismatches,
            "durations_seconds": {
                name: round(value, 3) for name, value in durations.items()
            },
            "invariants": {
                "inputs_unchanged": True,
                "one_row_per_customer": True,
                "target_preserved": True,
                "split_preserved": True,
                "no_post_application_history_used": True,
                "v1_v2_v3_lineage_preserved": True,
                "model_exclusion_policy_recorded": True,
                "summary_excludes_customer_values": True,
            },
        }

        serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        if any(str(path.resolve()) in serialized for path in sources.values()):
            raise FeatureMartError("공유용 요약에 절대경로가 포함되었습니다.")

        outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_summary_path = tempfile.mkstemp(
            prefix=f".{outputs['summary'].stem}.",
            suffix=".json.tmp",
            dir=outputs["summary"].parent,
        )
        os.close(descriptor)
        summary_temporary = Path(raw_summary_path)
        summary_temporary.write_text(serialized, encoding="utf-8")
        for name in ("v1", "v2", "v3"):
            outputs[name].parent.mkdir(parents=True, exist_ok=True)
            temporary_outputs[name].replace(outputs[name])
        summary_temporary.replace(outputs["summary"])
        return summary
    finally:
        for temporary in temporary_outputs.values():
            temporary.unlink(missing_ok=True)
        if summary_temporary is not None:
            summary_temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="고객 단위 V1·V2·V3 분석 마트를 DuckDB로 구축합니다."
    )
    parser.add_argument("--application", type=Path, default=DEFAULT_APPLICATION_INPUT)
    parser.add_argument("--bureau", type=Path, default=DEFAULT_BUREAU_INPUT)
    parser.add_argument("--installments", type=Path, default=DEFAULT_INSTALLMENTS_INPUT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_INPUT)
    parser.add_argument("--v1-output", type=Path, default=DEFAULT_V1_OUTPUT)
    parser.add_argument("--v2-output", type=Path, default=DEFAULT_V2_OUTPUT)
    parser.add_argument("--v3-output", type=Path, default=DEFAULT_V3_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--sql-dir", type=Path, default=DEFAULT_SQL_DIR)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_feature_marts(
        args.application,
        args.bureau,
        args.installments,
        args.splits,
        v1_output=args.v1_output,
        v2_output=args.v2_output,
        v3_output=args.v3_output,
        summary_output=args.summary_output,
        sql_dir=args.sql_dir,
        temp_dir=args.temp_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(
        "Stage 3 분석 마트 구축 완료: "
        + ", ".join(
            f"{name.upper()}={summary['outputs'][name]['rows']:,}행/"
            f"{summary['outputs'][name]['columns']}열"
            for name in ("v1", "v2", "v3")
        )
    )
    print(f"총 실행시간: {summary['durations_seconds']['total_seconds']:.3f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
