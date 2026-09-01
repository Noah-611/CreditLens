"""Stage 4 모델 입력 피처 역할과 V1·V2·V3 스키마 계약."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Sequence


MartVersion = Literal["v1", "v2", "v3"]

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
SPLIT_COLUMN = "SPLIT"
METADATA_COLUMNS = (ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN)
POLICY_EXCLUDED_COLUMNS = ("CODE_GENDER",)

NUMERIC_DUCKDB_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "BOOLEAN",
}
CATEGORICAL_DUCKDB_TYPES = {"VARCHAR"}


class FeatureRoleError(ValueError):
    """분석 마트가 Stage 4 모델 입력 계약을 만족하지 못할 때 발생한다."""


@dataclass(frozen=True)
class VersionContract:
    total_columns: int
    model_features: int
    numeric_features: int
    categorical_features: int


VERSION_CONTRACTS: dict[MartVersion, VersionContract] = {
    "v1": VersionContract(136, 132, 118, 14),
    "v2": VersionContract(173, 169, 155, 14),
    "v3": VersionContract(202, 198, 184, 14),
}


@dataclass(frozen=True)
class FeatureRoles:
    version: MartVersion
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    metadata: tuple[str, ...] = METADATA_COLUMNS
    policy_excluded: tuple[str, ...] = POLICY_EXCLUDED_COLUMNS

    @property
    def model_features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.model_features

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return self.numeric

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return self.categorical


def schema_sha256(schema: Sequence[tuple[str, str]]) -> str:
    """Stage 3과 같은 컬럼명+DuckDB 자료형 순서로 스키마 해시를 만든다."""

    payload = "\n".join(f"{name}\t{data_type}" for name, data_type in schema)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _base_type(data_type: str) -> str:
    return data_type.upper().split("(", 1)[0]


def resolve_feature_roles(
    version: MartVersion,
    schema: Sequence[tuple[str, str]],
) -> FeatureRoles:
    """Parquet 스키마를 검증하고 모델의 수치·범주형 컬럼을 확정한다."""

    if version not in VERSION_CONTRACTS:
        raise FeatureRoleError(f"지원하지 않는 분석 마트 버전입니다: {version}")

    names = [name for name, _ in schema]
    if len(names) != len(set(names)):
        raise FeatureRoleError("분석 마트 스키마에 중복 컬럼명이 있습니다.")
    required = set(METADATA_COLUMNS + POLICY_EXCLUDED_COLUMNS)
    missing = sorted(required.difference(names))
    if missing:
        raise FeatureRoleError(f"필수·정책 컬럼이 없습니다: {missing}")
    if any(name.startswith("SK_ID_") and name != ID_COLUMN for name in names):
        leaked = sorted(
            name for name in names if name.startswith("SK_ID_") and name != ID_COLUMN
        )
        raise FeatureRoleError(f"추가 식별자 컬럼은 모델 마트에 둘 수 없습니다: {leaked}")

    contract = VERSION_CONTRACTS[version]
    if len(schema) != contract.total_columns:
        raise FeatureRoleError(
            f"{version} 전체 컬럼 수가 계약과 다릅니다: "
            f"{len(schema)} != {contract.total_columns}"
        )

    numeric: list[str] = []
    categorical: list[str] = []
    excluded = required
    for name, data_type in schema:
        if name in excluded:
            continue
        base_type = _base_type(data_type)
        if base_type in NUMERIC_DUCKDB_TYPES:
            numeric.append(name)
        elif base_type in CATEGORICAL_DUCKDB_TYPES:
            categorical.append(name)
        else:
            raise FeatureRoleError(
                f"모델 피처 {name}의 자료형을 분류할 수 없습니다: {data_type}"
            )

    actual = (len(numeric) + len(categorical), len(numeric), len(categorical))
    expected = (
        contract.model_features,
        contract.numeric_features,
        contract.categorical_features,
    )
    if actual != expected:
        raise FeatureRoleError(
            f"{version} 피처 역할 수가 계약과 다릅니다: {actual} != {expected}"
        )
    return FeatureRoles(version, tuple(numeric), tuple(categorical))
