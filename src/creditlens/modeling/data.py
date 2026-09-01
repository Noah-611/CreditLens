"""봉인된 test를 읽지 않는 Stage 4 개발 데이터 로더."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import duckdb
import numpy as np
import pandas as pd

from creditlens.modeling.feature_roles import (
    ID_COLUMN,
    METADATA_COLUMNS,
    POLICY_EXCLUDED_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    FeatureRoleError,
    FeatureRoles,
    MartVersion,
    VERSION_CONTRACTS,
    resolve_feature_roles,
    schema_sha256,
)


DevelopmentSplit = Literal["train", "validation"]
ALLOWED_DEVELOPMENT_SPLITS = ("train", "validation")
DEFAULT_MART_PATHS: dict[MartVersion, Path] = {
    "v1": Path("data/processed/feature_mart_v1.parquet"),
    "v2": Path("data/processed/feature_mart_v2.parquet"),
    "v3": Path("data/processed/feature_mart_v3.parquet"),
}
DEFAULT_BUILD_SUMMARY_PATH = Path("reports/stage3_build_summary.json")


class ModelingDataError(ValueError):
    """Stage 4 모델 데이터 계약 위반."""


class TestSetSealedError(ModelingDataError):
    """Stage 8 이전에 봉인 test를 요청했을 때 발생한다."""


@dataclass(frozen=True)
class ModelSplit:
    name: DevelopmentSplit
    X: pd.DataFrame
    y: pd.Series
    customer_ids: pd.Index


@dataclass(frozen=True)
class DevelopmentDataset:
    version: MartVersion
    train: ModelSplit
    validation: ModelSplit
    roles: FeatureRoles
    parquet_sha256: str | None
    schema_sha256: str
    audit: Mapping[str, Any]


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_development_split(split: str) -> DevelopmentSplit:
    if split == "test":
        raise TestSetSealedError("test 세트는 Stage 8 최종 평가 전까지 봉인됩니다.")
    if split not in ALLOWED_DEVELOPMENT_SPLITS:
        raise ModelingDataError(f"지원하지 않는 개발 split입니다: {split}")
    return split  # type: ignore[return-value]


def _read_schema(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> list[tuple[str, str]]:
    rows = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def _validate_split_metadata(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict[str, Any]:
    """피처나 TARGET을 읽지 않고 고객 키와 split 계약만 확인한다."""

    key_metrics = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT {_quote_identifier(ID_COLUMN)}),
            COUNT(*) FILTER (WHERE {_quote_identifier(ID_COLUMN)} IS NULL)
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()
    if key_metrics[0] != key_metrics[1] or key_metrics[2]:
        raise ModelingDataError("분석 마트 전체에 결측 또는 중복 고객 ID가 있습니다.")

    rows = connection.execute(
        f"SELECT DISTINCT {_quote_identifier(SPLIT_COLUMN)} FROM read_parquet(?)",
        [str(path)],
    ).fetchall()
    observed = {row[0] for row in rows}
    allowed = {"train", "validation", "test"}
    invalid = observed.difference(allowed)
    if None in observed or invalid:
        raise ModelingDataError(f"SPLIT 값이 계약과 다릅니다: {sorted(invalid, key=str)}")
    missing = allowed.difference(observed)
    if missing:
        raise ModelingDataError(f"필수 SPLIT이 없습니다: {sorted(missing)}")

    split_rows = {
        str(split): int(count)
        for split, count in connection.execute(
            f"""
            SELECT {_quote_identifier(SPLIT_COLUMN)}, COUNT(*)
            FROM read_parquet(?)
            GROUP BY {_quote_identifier(SPLIT_COLUMN)}
            """,
            [str(path)],
        ).fetchall()
    }
    return {
        "rows": int(key_metrics[0]),
        "split_rows": split_rows,
    }


def _validate_summary(
    summary_path: Path,
    version: MartVersion,
    path: Path,
    actual_schema_hash: str,
    split_metadata: Mapping[str, Any],
    *,
    verify_file_sha256: bool,
) -> str | None:
    if not summary_path.is_file():
        raise ModelingDataError(f"Stage 3 요약 파일을 찾을 수 없습니다: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        output = summary["outputs"][version]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelingDataError("Stage 3 요약 파일 형식이 올바르지 않습니다.") from error

    contract = VERSION_CONTRACTS[version]
    expected_values = {
        "rows": split_metadata["rows"],
        "columns": contract.total_columns,
        "model_feature_columns": contract.model_features,
        "schema_sha256": actual_schema_hash,
    }
    for key, expected in expected_values.items():
        if output.get(key) != expected:
            raise ModelingDataError(
                f"Stage 3 요약의 {version}.{key}가 현재 마트와 다릅니다."
            )
    try:
        expected_split_rows = {
            split: int(details["rows"])
            for split, details in output["split_counts"].items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ModelingDataError("Stage 3 요약의 split 건수 형식이 올바르지 않습니다.") from error
    if expected_split_rows != split_metadata["split_rows"]:
        raise ModelingDataError(
            f"Stage 3 요약의 {version} split별 행 수가 현재 마트와 다릅니다."
        )
    expected_digest = output.get("sha256")
    if verify_file_sha256:
        if (
            not isinstance(expected_digest, str)
            or _file_sha256(path) != expected_digest
        ):
            raise ModelingDataError(f"{version} Parquet SHA256이 Stage 3 요약과 다릅니다.")
        return expected_digest
    return None


def _resolve_integrity_contract(
    version: MartVersion,
    mart_path: Path,
    summary_path: str | Path | None,
    verify_file_sha256: bool | None,
) -> tuple[Path | None, bool]:
    production_path = mart_path.resolve() == DEFAULT_MART_PATHS[version].resolve()
    selected_summary_path = (
        Path(summary_path)
        if summary_path is not None
        else DEFAULT_BUILD_SUMMARY_PATH if production_path else None
    )
    should_verify_file = (
        production_path if verify_file_sha256 is None else verify_file_sha256
    )
    if should_verify_file and selected_summary_path is None:
        raise ModelingDataError("SHA256 교차검증에는 summary_path가 필요합니다.")
    return selected_summary_path, should_verify_file


def _load_split(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    split: str,
    roles: FeatureRoles,
) -> ModelSplit:
    selected_split = _require_development_split(split)
    projected = (ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN) + roles.model_features
    columns_sql = ", ".join(_quote_identifier(column) for column in projected)
    query = (
        f"SELECT {columns_sql} FROM read_parquet(?) "
        f"WHERE {_quote_identifier(SPLIT_COLUMN)} = ? "
        f"ORDER BY {_quote_identifier(ID_COLUMN)}"
    )
    frame = connection.execute(
        query,
        [str(path), selected_split],
    ).fetchdf()
    if frame.empty:
        raise ModelingDataError(f"{selected_split} split이 비어 있습니다.")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ModelingDataError(f"{selected_split}에 결측 또는 중복 고객 ID가 있습니다.")
    if not frame[TARGET_COLUMN].isin((0, 1)).all() or frame[TARGET_COLUMN].isna().any():
        raise ModelingDataError(f"{selected_split} TARGET은 결측 없는 0/1이어야 합니다.")
    if not frame[SPLIT_COLUMN].eq(selected_split).all():
        raise ModelingDataError(f"{selected_split} 조회 결과에 다른 split이 섞였습니다.")

    X = frame.loc[:, roles.model_features].copy()
    forbidden = set(METADATA_COLUMNS + POLICY_EXCLUDED_COLUMNS)
    forbidden.update(column for column in X if column.startswith("SK_ID_"))
    if forbidden.intersection(X.columns):
        raise ModelingDataError("모델 입력 X에 메타데이터·식별자·정책 제외 컬럼이 있습니다.")
    for column in roles.numeric:
        values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype="float64")
        if np.isinf(values).any():
            raise ModelingDataError(f"{selected_split} 수치 피처에 무한대가 있습니다: {column}")

    y = frame[TARGET_COLUMN].astype("int8").rename(TARGET_COLUMN)
    customer_ids = pd.Index(frame[ID_COLUMN].astype("int64"), name=ID_COLUMN)
    X.index = customer_ids
    y.index = customer_ids
    if tuple(X.columns) != roles.model_features:
        raise ModelingDataError("모델 피처 순서가 역할 계약과 다릅니다.")
    return ModelSplit(selected_split, X, y, customer_ids)


def load_model_split(
    path: str | Path,
    version: MartVersion,
    split: str,
    *,
    summary_path: str | Path | None = None,
    verify_file_sha256: bool | None = None,
) -> ModelSplit:
    """한 개발 split을 읽는다. test 요청은 공개 API에서도 항상 거부한다."""

    selected_split = _require_development_split(split)
    if version not in DEFAULT_MART_PATHS:
        raise ModelingDataError(f"지원하지 않는 분석 마트 버전입니다: {version}")
    mart_path = Path(path)
    if not mart_path.is_file():
        raise ModelingDataError(f"분석 마트를 찾을 수 없습니다: {mart_path}")
    selected_summary_path, should_verify_file = _resolve_integrity_contract(
        version,
        mart_path,
        summary_path,
        verify_file_sha256,
    )
    with duckdb.connect(database=":memory:") as connection:
        split_metadata = _validate_split_metadata(connection, mart_path)
        schema = _read_schema(connection, mart_path)
        try:
            roles = resolve_feature_roles(version, schema)
        except FeatureRoleError as error:
            raise ModelingDataError(str(error)) from error
        if selected_summary_path is not None:
            _validate_summary(
                selected_summary_path,
                version,
                mart_path,
                schema_sha256(schema),
                split_metadata,
                verify_file_sha256=should_verify_file,
            )
        return _load_split(connection, mart_path, selected_split, roles)


def load_development_data(
    version: MartVersion,
    path: str | Path | None = None,
    *,
    summary_path: str | Path | None = None,
    verify_file_sha256: bool | None = None,
) -> DevelopmentDataset:
    """V1/V2/V3 train·validation만 projection하여 공통 계약으로 반환한다."""

    if version not in DEFAULT_MART_PATHS:
        raise ModelingDataError(f"지원하지 않는 분석 마트 버전입니다: {version}")
    mart_path = Path(path) if path is not None else DEFAULT_MART_PATHS[version]
    if not mart_path.is_file():
        raise ModelingDataError(f"분석 마트를 찾을 수 없습니다: {mart_path}")

    selected_summary_path, should_verify_file = _resolve_integrity_contract(
        version,
        mart_path,
        summary_path,
        verify_file_sha256,
    )

    with duckdb.connect(database=":memory:") as connection:
        split_metadata = _validate_split_metadata(connection, mart_path)
        schema = _read_schema(connection, mart_path)
        try:
            roles = resolve_feature_roles(version, schema)
        except FeatureRoleError as error:
            raise ModelingDataError(str(error)) from error
        current_schema_hash = schema_sha256(schema)
        digest = None
        if selected_summary_path is not None:
            digest = _validate_summary(
                selected_summary_path,
                version,
                mart_path,
                current_schema_hash,
                split_metadata,
                verify_file_sha256=should_verify_file,
            )

        train = _load_split(connection, mart_path, "train", roles)
        validation = _load_split(connection, mart_path, "validation", roles)

    if train.customer_ids.intersection(validation.customer_ids).size:
        raise ModelingDataError("train과 validation 고객 ID가 겹칩니다.")
    audit = {
        "feature_rows_loaded": {
            "train": len(train.X),
            "validation": len(validation.X),
            "test": 0,
        },
        "test_feature_rows_used": 0,
        "test_sealed": True,
        "stage3_summary_verified": selected_summary_path is not None,
        "parquet_sha256_verified": should_verify_file,
    }
    return DevelopmentDataset(
        version,
        train,
        validation,
        roles,
        digest,
        current_schema_hash,
        audit,
    )
