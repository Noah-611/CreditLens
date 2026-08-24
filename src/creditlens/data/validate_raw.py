"""Home Credit 원본 CSV를 읽기 전용으로 검증하고 Stage 1 문서를 생성한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


APPLICATION_COLUMNS = tuple(
    "SK_ID_CURR,TARGET,NAME_CONTRACT_TYPE,CODE_GENDER,FLAG_OWN_CAR,"
    "FLAG_OWN_REALTY,CNT_CHILDREN,AMT_INCOME_TOTAL,AMT_CREDIT,AMT_ANNUITY,"
    "AMT_GOODS_PRICE,NAME_TYPE_SUITE,NAME_INCOME_TYPE,NAME_EDUCATION_TYPE,"
    "NAME_FAMILY_STATUS,NAME_HOUSING_TYPE,REGION_POPULATION_RELATIVE,"
    "DAYS_BIRTH,DAYS_EMPLOYED,DAYS_REGISTRATION,DAYS_ID_PUBLISH,OWN_CAR_AGE,"
    "FLAG_MOBIL,FLAG_EMP_PHONE,FLAG_WORK_PHONE,FLAG_CONT_MOBILE,FLAG_PHONE,"
    "FLAG_EMAIL,OCCUPATION_TYPE,CNT_FAM_MEMBERS,REGION_RATING_CLIENT,"
    "REGION_RATING_CLIENT_W_CITY,WEEKDAY_APPR_PROCESS_START,"
    "HOUR_APPR_PROCESS_START,REG_REGION_NOT_LIVE_REGION,REG_REGION_NOT_WORK_REGION,"
    "LIVE_REGION_NOT_WORK_REGION,REG_CITY_NOT_LIVE_CITY,REG_CITY_NOT_WORK_CITY,"
    "LIVE_CITY_NOT_WORK_CITY,ORGANIZATION_TYPE,EXT_SOURCE_1,EXT_SOURCE_2,"
    "EXT_SOURCE_3,APARTMENTS_AVG,BASEMENTAREA_AVG,YEARS_BEGINEXPLUATATION_AVG,"
    "YEARS_BUILD_AVG,COMMONAREA_AVG,ELEVATORS_AVG,ENTRANCES_AVG,FLOORSMAX_AVG,"
    "FLOORSMIN_AVG,LANDAREA_AVG,LIVINGAPARTMENTS_AVG,LIVINGAREA_AVG,"
    "NONLIVINGAPARTMENTS_AVG,NONLIVINGAREA_AVG,APARTMENTS_MODE,BASEMENTAREA_MODE,"
    "YEARS_BEGINEXPLUATATION_MODE,YEARS_BUILD_MODE,COMMONAREA_MODE,ELEVATORS_MODE,"
    "ENTRANCES_MODE,FLOORSMAX_MODE,FLOORSMIN_MODE,LANDAREA_MODE,"
    "LIVINGAPARTMENTS_MODE,LIVINGAREA_MODE,NONLIVINGAPARTMENTS_MODE,"
    "NONLIVINGAREA_MODE,APARTMENTS_MEDI,BASEMENTAREA_MEDI,"
    "YEARS_BEGINEXPLUATATION_MEDI,YEARS_BUILD_MEDI,COMMONAREA_MEDI,ELEVATORS_MEDI,"
    "ENTRANCES_MEDI,FLOORSMAX_MEDI,FLOORSMIN_MEDI,LANDAREA_MEDI,"
    "LIVINGAPARTMENTS_MEDI,LIVINGAREA_MEDI,NONLIVINGAPARTMENTS_MEDI,"
    "NONLIVINGAREA_MEDI,FONDKAPREMONT_MODE,HOUSETYPE_MODE,TOTALAREA_MODE,"
    "WALLSMATERIAL_MODE,EMERGENCYSTATE_MODE,OBS_30_CNT_SOCIAL_CIRCLE,"
    "DEF_30_CNT_SOCIAL_CIRCLE,OBS_60_CNT_SOCIAL_CIRCLE,"
    "DEF_60_CNT_SOCIAL_CIRCLE,DAYS_LAST_PHONE_CHANGE,FLAG_DOCUMENT_2,"
    "FLAG_DOCUMENT_3,FLAG_DOCUMENT_4,FLAG_DOCUMENT_5,FLAG_DOCUMENT_6,"
    "FLAG_DOCUMENT_7,FLAG_DOCUMENT_8,FLAG_DOCUMENT_9,FLAG_DOCUMENT_10,"
    "FLAG_DOCUMENT_11,FLAG_DOCUMENT_12,FLAG_DOCUMENT_13,FLAG_DOCUMENT_14,"
    "FLAG_DOCUMENT_15,FLAG_DOCUMENT_16,FLAG_DOCUMENT_17,FLAG_DOCUMENT_18,"
    "FLAG_DOCUMENT_19,FLAG_DOCUMENT_20,FLAG_DOCUMENT_21,"
    "AMT_REQ_CREDIT_BUREAU_HOUR,AMT_REQ_CREDIT_BUREAU_DAY,"
    "AMT_REQ_CREDIT_BUREAU_WEEK,AMT_REQ_CREDIT_BUREAU_MON,"
    "AMT_REQ_CREDIT_BUREAU_QRT,AMT_REQ_CREDIT_BUREAU_YEAR"
    .split(",")
)

BUREAU_COLUMNS = tuple(
    "SK_ID_CURR,SK_ID_BUREAU,CREDIT_ACTIVE,CREDIT_CURRENCY,DAYS_CREDIT,"
    "CREDIT_DAY_OVERDUE,DAYS_CREDIT_ENDDATE,DAYS_ENDDATE_FACT,"
    "AMT_CREDIT_MAX_OVERDUE,CNT_CREDIT_PROLONG,AMT_CREDIT_SUM,"
    "AMT_CREDIT_SUM_DEBT,AMT_CREDIT_SUM_LIMIT,AMT_CREDIT_SUM_OVERDUE,"
    "CREDIT_TYPE,DAYS_CREDIT_UPDATE,AMT_ANNUITY"
    .split(",")
)

INSTALLMENTS_COLUMNS = tuple(
    "SK_ID_PREV,SK_ID_CURR,NUM_INSTALMENT_VERSION,NUM_INSTALMENT_NUMBER,"
    "DAYS_INSTALMENT,DAYS_ENTRY_PAYMENT,AMT_INSTALMENT,AMT_PAYMENT"
    .split(",")
)

APPLICATION_CATEGORICAL_COLUMNS = frozenset(
    {
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "ORGANIZATION_TYPE",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    }
)
BUREAU_CATEGORICAL_COLUMNS = frozenset(
    {"CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"}
)
CATEGORICAL_COLUMNS_BY_TABLE = {
    "application": APPLICATION_CATEGORICAL_COLUMNS,
    "bureau": BUREAU_CATEGORICAL_COLUMNS,
    "installments": frozenset(),
}


@dataclass(frozen=True)
class TableSpec:
    """고정된 원본 테이블 계약."""

    name: str
    filename: str
    columns: tuple[str, ...]
    primary_key: str | None
    grain: str
    expected_rows: int | None = None
    expected_sha256: str | None = None


TABLE_SPECS = (
    TableSpec(
        name="application",
        filename="application_train.csv",
        columns=APPLICATION_COLUMNS,
        primary_key="SK_ID_CURR",
        grain="현재 대출 신청 1건당 1행",
        expected_rows=307_511,
        expected_sha256="52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f",
    ),
    TableSpec(
        name="bureau",
        filename="bureau.csv",
        columns=BUREAU_COLUMNS,
        primary_key="SK_ID_BUREAU",
        grain="타 금융기관 신용거래 1건당 1행",
        expected_rows=1_716_428,
        expected_sha256="9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7",
    ),
    TableSpec(
        name="installments",
        filename="installments_payments.csv",
        columns=INSTALLMENTS_COLUMNS,
        primary_key=None,
        grain="예정 할부금에 대한 실제 납부행위 1건당 1행(분할납부 가능)",
        expected_rows=13_605_401,
        expected_sha256="428c2e2496e4d6d697ee8270e98497e5213c41be16d882eed1bc95b133726797",
    ),
)

DESCRIPTION_FILENAME = "HomeCredit_columns_description.csv"
DESCRIPTION_EXPECTED_SHA256 = "eef7665398228a80f7367c9258220c5fbe1038f3f54094244f354d54e2d4fb03"
DESCRIPTION_EXPECTED_ROWS = 219
DESCRIPTION_TABLE_ALIASES = {
    "application_train.csv": "application_{train|test}.csv",
    "bureau.csv": "bureau.csv",
    "installments_payments.csv": "installments_payments.csv",
}


class RawDataValidationError(RuntimeError):
    """검증을 실행할 수 없을 정도로 원본 계약이 깨졌을 때 발생한다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _series_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _numeric_columns(table: str, columns: Iterable[str]) -> set[str]:
    """공식 스키마에서 범주형으로 선언하지 않은 컬럼을 수치형으로 반환한다."""

    return set(columns) - set(CATEGORICAL_COLUMNS_BY_TABLE[table])


def _stable_row_hash(chunk: pd.DataFrame, numeric_columns: set[str]) -> np.ndarray:
    """청크별 dtype 추론 차이를 제거한 안정적인 64비트 행 해시를 만든다."""

    normalized: dict[str, pd.Series] = {}
    invalid_tokens: list[tuple[str, pd.Series, pd.Series]] = []
    for column in chunk.columns:
        if column in numeric_columns:
            original = chunk[column]
            numeric = pd.to_numeric(original, errors="coerce")
            normalized[column] = numeric.astype("Float64")
            invalid = original.notna() & numeric.isna()
            if invalid.any():
                invalid_tokens.append((column, original, invalid))
        else:
            normalized[column] = chunk[column].astype("string")
    frame = pd.DataFrame(normalized, columns=chunk.columns)
    row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(
        dtype="uint64", copy=True
    )
    for column, original, invalid in invalid_tokens:
        positions = np.flatnonzero(invalid.to_numpy(dtype=bool, na_value=False))
        tokens = column + "\x1f" + original.iloc[positions].astype("string")
        correction = pd.util.hash_pandas_object(
            tokens, index=False, categorize=True
        ).to_numpy(dtype="uint64", copy=False)
        row_hashes[positions] ^= correction
    return row_hashes


def _update_rule(
    rules: dict[str, dict[str, Any]],
    *,
    code: str,
    column: str,
    description: str,
    violations: int,
    severity: str = "ERROR",
) -> None:
    record = rules.setdefault(
        code,
        {
            "code": code,
            "column": column,
            "description": description,
            "severity": severity,
            "violations": 0,
        },
    )
    record["violations"] += int(violations)


def _evaluate_rules(table: str, chunk: pd.DataFrame, rules: dict[str, dict[str, Any]]) -> None:
    numeric = {
        column: pd.to_numeric(chunk[column], errors="coerce")
        for column in _numeric_columns(table, chunk.columns)
    }
    for column, values in numeric.items():
        if column.startswith("SK_ID_"):
            continue
        _update_rule(
            rules,
            code=f"{column.lower()}_numeric",
            column=column,
            description=f"{column}은 숫자여야 한다.",
            violations=int((chunk[column].notna() & values.isna()).sum()),
        )

    for column in (name for name in chunk.columns if name.startswith("SK_ID_")):
        original = chunk[column]
        numeric_id = numeric[column]
        _update_rule(
            rules,
            code=f"{column.lower()}_not_null",
            column=column,
            description="ID는 결측일 수 없다.",
            violations=int(original.isna().sum()),
        )
        _update_rule(
            rules,
            code=f"{column.lower()}_numeric",
            column=column,
            description="ID는 숫자여야 한다.",
            violations=int((original.notna() & numeric_id.isna()).sum()),
        )
        values = numeric_id.to_numpy(dtype="float64", na_value=np.nan)
        finite = np.isfinite(values)
        _update_rule(
            rules,
            code=f"{column.lower()}_integer",
            column=column,
            description="ID는 정수여야 한다.",
            violations=int(np.count_nonzero(finite & (values != np.floor(values)))),
        )
        _update_rule(
            rules,
            code=f"{column.lower()}_positive",
            column=column,
            description="ID는 양수여야 한다.",
            violations=int(np.count_nonzero(finite & (values <= 0))),
        )

    if table == "application":
        binary_columns = [
            column
            for column in chunk.columns
            if column.startswith("FLAG_")
            and column not in {"FLAG_OWN_CAR", "FLAG_OWN_REALTY"}
        ]
        binary_columns.extend(
            column
            for column in (
                "REG_REGION_NOT_LIVE_REGION",
                "REG_REGION_NOT_WORK_REGION",
                "LIVE_REGION_NOT_WORK_REGION",
                "REG_CITY_NOT_LIVE_CITY",
                "REG_CITY_NOT_WORK_CITY",
                "LIVE_CITY_NOT_WORK_CITY",
            )
            if column in chunk
        )
        for column in binary_columns:
            original = chunk[column]
            values = numeric[column]
            invalid = original.isna() | values.isna() | ~values.isin([0, 1])
            _update_rule(
                rules,
                code=f"{column.lower()}_binary",
                column=column,
                description="이진 플래그는 0 또는 1이어야 한다.",
                violations=int(invalid.sum()),
            )

        for column in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
            original = chunk[column]
            invalid = original.isna() | ~original.isin(["Y", "N"])
            _update_rule(
                rules,
                code=f"{column.lower()}_yn",
                column=column,
                description="소유 여부 플래그는 Y 또는 N이어야 한다.",
                violations=int(invalid.sum()),
            )

        target = chunk["TARGET"]
        target_numeric = numeric["TARGET"]
        _update_rule(
            rules,
            code="target_domain",
            column="TARGET",
            description="TARGET은 0 또는 1이어야 한다.",
            violations=int(
                (target.notna() & (target_numeric.isna() | ~target_numeric.isin([0, 1]))).sum()
            ),
        )
        days_birth = numeric["DAYS_BIRTH"]
        _update_rule(
            rules,
            code="days_birth_negative",
            column="DAYS_BIRTH",
            description="DAYS_BIRTH는 신청일 이전을 나타내는 음수여야 한다.",
            violations=int((days_birth.notna() & (days_birth >= 0)).sum()),
        )
        employed = numeric["DAYS_EMPLOYED"]
        invalid_employed = employed.notna() & (employed > 0) & (employed != 365243)
        _update_rule(
            rules,
            code="days_employed_domain",
            column="DAYS_EMPLOYED",
            description="DAYS_EMPLOYED의 양수는 알려진 365243 sentinel만 허용한다.",
            violations=int(invalid_employed.sum()),
        )
        hour = numeric["HOUR_APPR_PROCESS_START"]
        _update_rule(
            rules,
            code="hour_range",
            column="HOUR_APPR_PROCESS_START",
            description="신청 시각은 0~23 범위여야 한다.",
            violations=int((hour.notna() & ~hour.between(0, 23)).sum()),
        )
        for column in (
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
        ):
            values = numeric[column]
            invalid = values.notna() & (values < 0)
            _update_rule(
                rules,
                code=f"{column.lower()}_non_negative",
                column=column,
                description=f"{column}은 음수일 수 없다.",
                violations=int(invalid.sum()),
            )

    elif table == "bureau":
        overdue = numeric["CREDIT_DAY_OVERDUE"]
        _update_rule(
            rules,
            code="credit_day_overdue_non_negative",
            column="CREDIT_DAY_OVERDUE",
            description="연체일수는 음수일 수 없다.",
            violations=int((overdue.notna() & (overdue < 0)).sum()),
        )
        for column in ("AMT_CREDIT_SUM", "AMT_CREDIT_SUM_OVERDUE"):
            values = numeric[column]
            invalid = values.notna() & (values < 0)
            _update_rule(
                rules,
                code=f"{column.lower()}_non_negative",
                column=column,
                description=f"{column}은 음수일 수 없다.",
                violations=int(invalid.sum()),
            )

    elif table == "installments":
        for column in ("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"):
            values = numeric[column]
            invalid = values.notna() & (values > 0)
            _update_rule(
                rules,
                code=f"{column.lower()}_non_positive",
                column=column,
                description=f"{column}은 현재 신청일 기준 0 이하이어야 한다.",
                violations=int(invalid.sum()),
            )
        for column in ("NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "AMT_INSTALMENT", "AMT_PAYMENT"):
            values = numeric[column]
            invalid = values.notna() & (values < 0)
            _update_rule(
                rules,
                code=f"{column.lower()}_non_negative",
                column=column,
                description=f"{column}은 음수일 수 없다.",
                violations=int(invalid.sum()),
            )


def _update_domain_observations(table: str, chunk: pd.DataFrame, counts: Counter[str]) -> None:
    if table == "application":
        days_employed = pd.to_numeric(chunk["DAYS_EMPLOYED"], errors="coerce")
        counts["days_employed_365243"] += int((days_employed == 365243).sum())
        counts["code_gender_xna"] += int((chunk["CODE_GENDER"] == "XNA").sum())
        counts["family_status_unknown"] += int((chunk["NAME_FAMILY_STATUS"] == "Unknown").sum())
        counts["own_car_age_missing_when_car_owned"] += int(
            ((chunk["FLAG_OWN_CAR"] == "Y") & chunk["OWN_CAR_AGE"].isna()).sum()
        )
    elif table == "bureau":
        debt = pd.to_numeric(chunk["AMT_CREDIT_SUM_DEBT"], errors="coerce")
        limit = pd.to_numeric(chunk["AMT_CREDIT_SUM_LIMIT"], errors="coerce")
        update_days = pd.to_numeric(chunk["DAYS_CREDIT_UPDATE"], errors="coerce")
        counts["negative_credit_sum_debt"] += int((debt < 0).sum())
        counts["negative_credit_sum_limit"] += int((limit < 0).sum())
        counts["positive_days_credit_update"] += int((update_days > 0).sum())
        counts["active_with_enddate_fact"] += int(
            ((chunk["CREDIT_ACTIVE"] == "Active") & chunk["DAYS_ENDDATE_FACT"].notna()).sum()
        )
        counts["closed_without_enddate_fact"] += int(
            ((chunk["CREDIT_ACTIVE"] == "Closed") & chunk["DAYS_ENDDATE_FACT"].isna()).sum()
        )
    elif table == "installments":
        days_entry = pd.to_numeric(chunk["DAYS_ENTRY_PAYMENT"], errors="coerce")
        days_installment = pd.to_numeric(chunk["DAYS_INSTALMENT"], errors="coerce")
        amount_payment = pd.to_numeric(chunk["AMT_PAYMENT"], errors="coerce")
        amount_installment = pd.to_numeric(chunk["AMT_INSTALMENT"], errors="coerce")
        entry_missing = chunk["DAYS_ENTRY_PAYMENT"].isna()
        payment_missing = chunk["AMT_PAYMENT"].isna()
        counts["payment_fields_both_missing"] += int((entry_missing & payment_missing).sum())
        counts["payment_fields_missing_mismatch"] += int((entry_missing ^ payment_missing).sum())
        comparable_days = days_entry.notna() & days_installment.notna()
        counts["payment_late"] += int(
            (comparable_days & (days_entry > days_installment)).sum()
        )
        counts["payment_early"] += int(
            (comparable_days & (days_entry < days_installment)).sum()
        )
        counts["payment_on_time"] += int(
            (comparable_days & (days_entry == days_installment)).sum()
        )
        comparable_amount = amount_payment.notna() & amount_installment.notna()
        counts["payment_below_scheduled"] += int(
            (comparable_amount & (amount_payment < amount_installment)).sum()
        )
        counts["payment_above_scheduled"] += int(
            (comparable_amount & (amount_payment > amount_installment)).sum()
        )
        counts["payment_equal_scheduled"] += int(
            (comparable_amount & (amount_payment == amount_installment)).sum()
        )


def _scan_table(
    raw_dir: Path,
    spec: TableSpec,
    *,
    chunk_size: int,
    reference_customer_ids: set[int] | None = None,
) -> tuple[dict[str, Any], set[int]]:
    path = raw_dir / spec.filename
    if not path.is_file():
        raise RawDataValidationError(f"필수 원본 파일이 없습니다: {spec.filename}")

    try:
        reader = pd.read_csv(path, chunksize=chunk_size)
        first_chunk = next(reader)
    except StopIteration as exc:
        raise RawDataValidationError(f"원본 파일이 비어 있습니다: {spec.filename}") from exc
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise RawDataValidationError(f"CSV를 읽을 수 없습니다: {spec.filename}: {exc}") from exc

    columns = tuple(str(column) for column in first_chunk.columns)
    if columns != spec.columns:
        missing = sorted(set(spec.columns) - set(columns))
        unexpected = sorted(set(columns) - set(spec.columns))
        raise RawDataValidationError(
            f"스키마 불일치: {spec.filename}; missing={missing}; unexpected={unexpected}; "
            f"column_order_changed={set(columns) == set(spec.columns)}"
        )

    numeric_columns = _numeric_columns(spec.name, columns)
    row_count = 0
    missing_counts = Counter({column: 0 for column in columns})
    blank_counts = Counter({column: 0 for column in columns})
    infinite_counts = Counter({column: 0 for column in columns})
    dtype_sets: dict[str, set[str]] = defaultdict(set)
    minima: dict[str, int | float | None] = {column: None for column in columns}
    maxima: dict[str, int | float | None] = {column: None for column in columns}
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rules: dict[str, dict[str, Any]] = {}
    domain_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    customer_ids: set[int] = set()
    matched_reference_rows = 0
    primary_key_parts: list[np.ndarray] = []
    previous_to_customer: dict[int, int] = {}
    dependency_violation_keys: set[int] = set()

    def consume(chunk: pd.DataFrame, hash_output: Any) -> None:
        nonlocal row_count, matched_reference_rows
        row_count += len(chunk)
        _stable_row_hash(chunk, numeric_columns).tofile(hash_output)

        for column in columns:
            series = chunk[column]
            dtype_sets[column].add(str(series.dtype))
            missing_counts[column] += int(series.isna().sum())
            if column in numeric_columns:
                values = _series_numeric(series)
                finite_values = values[np.isfinite(values)]
                infinite_counts[column] += int(np.isinf(values).sum())
                if finite_values.size:
                    chunk_min = _json_number(finite_values.min())
                    chunk_max = _json_number(finite_values.max())
                    if minima[column] is None or (chunk_min is not None and chunk_min < minima[column]):
                        minima[column] = chunk_min
                    if maxima[column] is None or (chunk_max is not None and chunk_max > maxima[column]):
                        maxima[column] = chunk_max
            else:
                strings = series.dropna().astype(str)
                blank_counts[column] += int(strings.str.strip().eq("").sum())
                category_counts[column].update(strings.value_counts(dropna=False).to_dict())

        customer_numeric = pd.to_numeric(chunk["SK_ID_CURR"], errors="coerce")
        customer_values = customer_numeric.to_numpy(dtype="float64", na_value=np.nan)
        valid_customer = (
            np.isfinite(customer_values)
            & (customer_values == np.floor(customer_values))
            & (customer_values > 0)
        )
        customer_ids.update(int(value) for value in np.unique(customer_values[valid_customer]))
        if reference_customer_ids is not None:
            matched_reference_rows += int(customer_numeric.isin(reference_customer_ids).sum())

        if spec.primary_key:
            primary_key = pd.to_numeric(chunk[spec.primary_key], errors="coerce").dropna()
            primary_key_parts.append(primary_key.to_numpy(copy=True))

        if spec.name == "application":
            target_numeric = pd.to_numeric(chunk["TARGET"], errors="coerce")
            for value, count in target_numeric.dropna().value_counts().items():
                numeric_value = float(value)
                key = (
                    str(int(numeric_value))
                    if np.isfinite(numeric_value) and numeric_value.is_integer()
                    else str(numeric_value)
                )
                target_counts[key] += int(count)

        if spec.name == "installments":
            pairs = chunk[["SK_ID_PREV", "SK_ID_CURR"]].apply(
                pd.to_numeric, errors="coerce"
            )
            pairs = pairs[
                np.isfinite(pairs["SK_ID_PREV"])
                & np.isfinite(pairs["SK_ID_CURR"])
                & (pairs["SK_ID_PREV"] == np.floor(pairs["SK_ID_PREV"]))
                & (pairs["SK_ID_CURR"] == np.floor(pairs["SK_ID_CURR"]))
                & (pairs["SK_ID_PREV"] > 0)
                & (pairs["SK_ID_CURR"] > 0)
            ].drop_duplicates()
            pairs = pairs.itertuples(index=False, name=None)
            for previous_id, customer_id in pairs:
                previous = int(previous_id)
                customer = int(customer_id)
                known_customer = previous_to_customer.setdefault(previous, customer)
                if known_customer != customer:
                    dependency_violation_keys.add(previous)

        _evaluate_rules(spec.name, chunk, rules)
        _update_domain_observations(spec.name, chunk, domain_counts)

    try:
        with tempfile.TemporaryDirectory(prefix="creditlens-row-hash-") as temp_dir:
            hash_path = Path(temp_dir) / "rows.uint64"
            with hash_path.open("wb") as hash_output:
                consume(first_chunk, hash_output)
                for chunk in reader:
                    consume(chunk, hash_output)

            if row_count == 0:
                raise RawDataValidationError(f"데이터 행이 없습니다: {spec.filename}")

            all_hashes = np.memmap(
                hash_path, dtype="uint64", mode="r+", shape=(row_count,)
            )
            all_hashes.sort()
            duplicate_row_hash_count = int(
                np.count_nonzero(all_hashes[1:] == all_hashes[:-1])
            )
            del all_hashes
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise RawDataValidationError(
            f"CSV 또는 임시 행 해시 처리 중 오류가 발생했습니다: {spec.filename}: {exc}"
        ) from exc

    primary_key_null_count = 0
    primary_key_duplicate_count = 0
    primary_key_unique_count: int | None = None
    if spec.primary_key:
        primary_key_null_count = int(missing_counts[spec.primary_key])
        all_primary_keys = np.concatenate(primary_key_parts) if primary_key_parts else np.array([])
        primary_key_unique_count = int(np.unique(all_primary_keys).size)
        primary_key_duplicate_count = int(len(all_primary_keys) - primary_key_unique_count)

    column_stats: dict[str, dict[str, Any]] = {}
    for column in columns:
        categories = category_counts.get(column, Counter())
        category_items = sorted(categories.items(), key=lambda item: (-item[1], item[0]))
        column_stats[column] = {
            "dtypes": sorted(dtype_sets[column]),
            "missing_count": int(missing_counts[column]),
            "missing_rate": _safe_ratio(int(missing_counts[column]), row_count),
            "blank_string_count": int(blank_counts[column]),
            "infinite_count": int(infinite_counts[column]),
            "min": minima[column],
            "max": maxima[column],
            "unique_category_count": len(categories) if categories else None,
            "top_categories": [
                {"value": value, "count": int(count)} for value, count in category_items[:20]
            ],
        }

    sha256 = _sha256(path)
    result: dict[str, Any] = {
        "name": spec.name,
        "filename": spec.filename,
        "display_path": f"data/raw/{spec.filename}",
        "grain": spec.grain,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256,
        "expected_sha256": spec.expected_sha256,
        "row_count": row_count,
        "expected_row_count": spec.expected_rows,
        "column_count": len(columns),
        "columns": list(columns),
        "column_stats": column_stats,
        "primary_key": spec.primary_key,
        "primary_key_null_count": primary_key_null_count,
        "primary_key_unique_count": primary_key_unique_count,
        "primary_key_duplicate_count": primary_key_duplicate_count,
        "duplicate_row_hash_count": duplicate_row_hash_count,
        "customer_unique_count": len(customer_ids),
        "matched_application_row_count": matched_reference_rows if reference_customer_ids is not None else None,
        "matched_application_customer_count": (
            len(customer_ids & reference_customer_ids) if reference_customer_ids is not None else None
        ),
        "rules": sorted(rules.values(), key=lambda item: (item["column"], item["code"])),
        "domain_observations": dict(sorted(domain_counts.items())),
        "target_counts": dict(sorted(target_counts.items())) if target_counts else None,
        "dependency": (
            {
                "rule": "SK_ID_PREV는 하나의 SK_ID_CURR에만 연결되어야 한다.",
                "violation_key_count": len(dependency_violation_keys),
            }
            if spec.name == "installments"
            else None
        ),
    }
    return result, customer_ids


def _finding(
    severity: str,
    code: str,
    message: str,
    *,
    table: str | None = None,
    rule_code: str | None = None,
) -> dict[str, str | None]:
    finding = {"severity": severity, "code": code, "table": table, "message": message}
    if rule_code is not None:
        finding["rule_code"] = rule_code
    return finding


def _build_findings(
    tables: Mapping[str, Mapping[str, Any]],
    relationships: Mapping[str, Any],
    description_manifest: Mapping[str, Any],
) -> list[dict[str, str | None]]:
    findings: list[dict[str, str | None]] = []
    for table_name, table in tables.items():
        if table["expected_row_count"] is not None and table["row_count"] != table["expected_row_count"]:
            findings.append(
                _finding(
                    "ERROR",
                    "row_count_mismatch",
                    f"고정 manifest 행 수 {table['expected_row_count']:,}와 실제 {table['row_count']:,}가 다릅니다.",
                    table=table_name,
                )
            )
        if table["expected_sha256"] and table["sha256"] != table["expected_sha256"]:
            findings.append(
                _finding(
                    "ERROR",
                    "checksum_mismatch",
                    "고정 manifest SHA-256과 실제 파일 체크섬이 다릅니다.",
                    table=table_name,
                )
            )
        if table["primary_key"]:
            if table["primary_key_null_count"]:
                findings.append(
                    _finding(
                        "ERROR",
                        "primary_key_null",
                        f"기본키 {table['primary_key']}에 결측 {table['primary_key_null_count']:,}건이 있습니다.",
                        table=table_name,
                    )
                )
            if table["primary_key_duplicate_count"]:
                findings.append(
                    _finding(
                        "ERROR",
                        "primary_key_duplicate",
                        f"기본키 {table['primary_key']}에 중복 {table['primary_key_duplicate_count']:,}건이 있습니다.",
                        table=table_name,
                    )
                )
        if table["duplicate_row_hash_count"]:
            findings.append(
                _finding(
                    "WARNING",
                    "duplicate_row_hash",
                    f"동일 행 해시가 {table['duplicate_row_hash_count']:,}건 반복됩니다. 실제 중복 여부를 Stage 2에서 확인합니다.",
                    table=table_name,
                )
            )
        for column, stats in table["column_stats"].items():
            if len(stats["dtypes"]) > 1:
                findings.append(
                    _finding(
                        "WARNING",
                        "dtype_drift",
                        f"{column}의 청크별 추론 dtype이 {', '.join(stats['dtypes'])}로 달라집니다.",
                        table=table_name,
                    )
                )
            if stats["infinite_count"]:
                findings.append(
                    _finding(
                        "ERROR",
                        "infinite_numeric_value",
                        f"{column}에 무한값 {stats['infinite_count']:,}건이 있습니다.",
                        table=table_name,
                    )
                )
            if stats["missing_rate"] >= 0.5:
                findings.append(
                    _finding(
                        "WARNING",
                        "high_missing_rate",
                        f"{column} 결측률이 {stats['missing_rate']:.2%}입니다.",
                        table=table_name,
                    )
                )
        for rule in table["rules"]:
            if rule["violations"]:
                findings.append(
                    _finding(
                        rule["severity"],
                        "rule_violation",
                        f"{rule['description']} 위반 {rule['violations']:,}건",
                        table=table_name,
                        rule_code=rule["code"],
                    )
                )

    application = tables["application"]
    target_counts = application["target_counts"] or {}
    target_missing = application["column_stats"]["TARGET"]["missing_count"]
    invalid_target = next(
        (
            int(rule["violations"])
            for rule in application["rules"]
            if rule["code"] == "target_domain"
        ),
        0,
    )
    if target_missing or invalid_target:
        findings.append(
            _finding(
                "ERROR",
                "target_invalid",
                f"TARGET 결측 {target_missing:,}건, 허용 범위 밖 {invalid_target:,}건입니다.",
                table="application",
            )
        )
    else:
        positive = int(target_counts.get(1, target_counts.get("1", 0)))
        findings.append(
            _finding(
                "INFO",
                "target_imbalance",
                f"TARGET=1은 {positive:,}건({_safe_ratio(positive, application['row_count']):.2%})으로 불균형 데이터입니다.",
                table="application",
            )
        )

    dependency = tables["installments"]["dependency"]
    if dependency and dependency["violation_key_count"]:
        findings.append(
            _finding(
                "ERROR",
                "previous_customer_dependency",
                f"SK_ID_PREV가 여러 고객에 연결된 키가 {dependency['violation_key_count']:,}개입니다.",
                table="installments",
            )
        )

    for child_name in ("bureau", "installments"):
        relation = relationships[child_name]
        findings.append(
            _finding(
                "INFO",
                "support_table_scope",
                f"train 밖 고객 {relation['outside_application_customer_count']:,}명은 application_test 고객이 포함된 원본 구조로 해석합니다.",
                table=child_name,
            )
        )

    if (
        description_manifest["expected_row_count"] is not None
        and description_manifest["row_count"]
        != description_manifest["expected_row_count"]
    ):
        findings.append(
            _finding(
                "ERROR",
                "description_row_count_mismatch",
                "공식 컬럼 설명 파일의 고정 manifest 행 수와 실제 행 수가 다릅니다.",
            )
        )
    if (
        description_manifest["expected_sha256"]
        and description_manifest["sha256"] != description_manifest["expected_sha256"]
    ):
        findings.append(
            _finding(
                "ERROR",
                "description_checksum_mismatch",
                "공식 컬럼 설명 파일의 고정 manifest SHA-256과 실제 체크섬이 다릅니다.",
            )
        )
    return findings


def _read_description_csv(
    path: Path,
) -> tuple[list[dict[str | None, str | None]], tuple[str, ...]]:
    """공식 컬럼 설명 CSV를 지원 인코딩으로 읽고 최소 스키마를 확인한다."""

    if not path.is_file():
        raise RawDataValidationError(f"필수 원본 파일이 없습니다: {DESCRIPTION_FILENAME}")
    content: str | None = None
    for encoding in ("utf-8-sig", "windows-1252"):
        try:
            content = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise RawDataValidationError(
                f"컬럼 설명 파일을 읽을 수 없습니다: {DESCRIPTION_FILENAME}: {exc}"
            ) from exc
    if content is None:
        raise RawDataValidationError(
            f"컬럼 설명 파일 인코딩을 해석할 수 없습니다: {DESCRIPTION_FILENAME}"
        )

    try:
        reader = csv.DictReader(content.splitlines())
        fieldnames = tuple(name for name in (reader.fieldnames or []) if name is not None)
        required_columns = {"Table", "Row", "Description", "Special"}
        if not required_columns.issubset(fieldnames):
            missing = sorted(required_columns - set(fieldnames))
            raise RawDataValidationError(
                f"컬럼 설명 파일 스키마 불일치: {DESCRIPTION_FILENAME}; missing={missing}"
            )
        rows = list(reader)
    except csv.Error as exc:
        raise RawDataValidationError(
            f"컬럼 설명 CSV를 읽을 수 없습니다: {DESCRIPTION_FILENAME}: {exc}"
        ) from exc
    return rows, fieldnames


def validate_raw_data(
    raw_dir: Path,
    *,
    chunk_size: int = 200_000,
    specs: Sequence[TableSpec] = TABLE_SPECS,
) -> dict[str, Any]:
    """세 원본 테이블을 청크 단위로 검증하고 직렬화 가능한 결과를 반환한다."""

    if chunk_size <= 0:
        raise ValueError("chunk_size는 양수여야 합니다.")
    raw_dir = Path(raw_dir)
    specs_by_name = {spec.name: spec for spec in specs}
    required_names = {"application", "bureau", "installments"}
    if set(specs_by_name) != required_names:
        raise ValueError(f"specs에는 {sorted(required_names)}가 정확히 한 번씩 있어야 합니다.")

    for name in ("application", "bureau", "installments"):
        filename = specs_by_name[name].filename
        if not (raw_dir / filename).is_file():
            raise RawDataValidationError(f"필수 원본 파일이 없습니다: {filename}")

    description_path = raw_dir / DESCRIPTION_FILENAME
    description_rows, description_columns = _read_description_csv(description_path)
    pin_description_manifest = all(
        spec.expected_rows is not None and spec.expected_sha256 is not None
        for spec in specs
    )
    description_manifest = {
        "filename": DESCRIPTION_FILENAME,
        "display_path": f"data/raw/{DESCRIPTION_FILENAME}",
        "file_size_bytes": description_path.stat().st_size,
        "sha256": _sha256(description_path),
        "expected_sha256": (
            DESCRIPTION_EXPECTED_SHA256 if pin_description_manifest else None
        ),
        "row_count": len(description_rows),
        "expected_row_count": DESCRIPTION_EXPECTED_ROWS if pin_description_manifest else None,
        "column_count": len(description_columns),
    }

    application, application_ids = _scan_table(
        raw_dir, specs_by_name["application"], chunk_size=chunk_size
    )
    bureau, bureau_ids = _scan_table(
        raw_dir,
        specs_by_name["bureau"],
        chunk_size=chunk_size,
        reference_customer_ids=application_ids,
    )
    installments, installment_ids = _scan_table(
        raw_dir,
        specs_by_name["installments"],
        chunk_size=chunk_size,
        reference_customer_ids=application_ids,
    )
    tables = {
        "application": application,
        "bureau": bureau,
        "installments": installments,
    }

    relationships: dict[str, Any] = {}
    for name, customer_ids in (("bureau", bureau_ids), ("installments", installment_ids)):
        table = tables[name]
        matched_customers = customer_ids & application_ids
        relationships[name] = {
            "application_customer_count": len(application_ids),
            "support_customer_count": len(customer_ids),
            "matched_application_customer_count": len(matched_customers),
            "application_customer_coverage_rate": _safe_ratio(len(matched_customers), len(application_ids)),
            "outside_application_customer_count": len(customer_ids - application_ids),
            "matched_application_row_count": table["matched_application_row_count"],
            "matched_application_row_rate": _safe_ratio(
                int(table["matched_application_row_count"]), int(table["row_count"])
            ),
        }

    both = application_ids & bureau_ids & installment_ids
    bureau_only = (application_ids & bureau_ids) - installment_ids
    installments_only = (application_ids & installment_ids) - bureau_ids
    neither = application_ids - bureau_ids - installment_ids
    history_overlap = {
        "both_history_customer_count": len(both),
        "bureau_only_customer_count": len(bureau_only),
        "installments_only_customer_count": len(installments_only),
        "neither_history_customer_count": len(neither),
        "both_history_rate": _safe_ratio(len(both), len(application_ids)),
        "neither_history_rate": _safe_ratio(len(neither), len(application_ids)),
    }

    findings = _build_findings(tables, relationships, description_manifest)
    severity_counts = Counter(str(item["severity"]) for item in findings)
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "chunk_size": chunk_size,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "description_manifest": description_manifest,
        "tables": tables,
        "relationships": relationships,
        "history_overlap": history_overlap,
        "findings": findings,
        "summary": {
            "status": "PASS" if severity_counts["ERROR"] == 0 else "FAIL",
            "error_count": severity_counts["ERROR"],
            "warning_count": severity_counts["WARNING"],
            "info_count": severity_counts["INFO"],
        },
    }
    return result


def _load_descriptions(raw_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = raw_dir / DESCRIPTION_FILENAME
    rows, _ = _read_description_csv(path)
    descriptions: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        table = (row.get("Table") or "").strip()
        column = (row.get("Row") or "").strip()
        if table and column:
            descriptions[(table, column)] = {
                "description": (row.get("Description") or "").strip(),
                "special": (row.get("Special") or "").strip(),
            }
    bureau_alias = descriptions.get(("bureau.csv", "SK_BUREAU_ID"))
    if bureau_alias and ("bureau.csv", "SK_ID_BUREAU") not in descriptions:
        descriptions[("bureau.csv", "SK_ID_BUREAU")] = {
            "description": bureau_alias["description"],
            "special": (
                f"{bureau_alias['special']}; 공식 설명 파일에는 SK_BUREAU_ID로 표기"
                if bureau_alias["special"]
                else "공식 설명 파일에는 SK_BUREAU_ID로 표기"
            ),
        }
    return descriptions


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{value} B"


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.6g}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _semantic_type(column: str, stats: Mapping[str, Any]) -> str:
    if column == "TARGET":
        return "타깃"
    if column.startswith("SK_ID_"):
        return "식별자"
    if column.startswith("AMT_REQ_CREDIT_BUREAU_"):
        return "조회 횟수"
    if column.startswith("AMT_"):
        return "금액"
    if column.startswith("DAYS_"):
        return "상대 일수"
    if column.startswith("FLAG_") or column.startswith(("REG_", "LIVE_")):
        return "플래그"
    if column.startswith("CNT_"):
        return "개수"
    if column.startswith("NUM_"):
        return "번호/수치"
    if column.startswith("EXT_SOURCE_"):
        return "외부 신용점수"
    dtypes = stats.get("dtypes", [])
    return "범주" if any(dtype in {"object", "str", "string"} for dtype in dtypes) else "수치"


def _stage2_note(table: str, column: str, stats: Mapping[str, Any]) -> str:
    if column == "TARGET":
        return "정답 레이블이며 입력 피처에서 제외"
    if column.startswith("SK_ID_"):
        return "조인·추적용 ID이며 모델 피처에서 제외"
    if column == "DAYS_EMPLOYED":
        return "365243 sentinel을 결측/별도 플래그로 검토"
    if table == "installments" and column in {"DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"}:
        return "동시 결측은 미납/미기록 의미를 확인하고 단순 0 대치 금지"
    if stats["missing_rate"] >= 0.5:
        return "고결측 변수: 의미와 활용 여부를 Stage 2에서 결정"
    if column.startswith("DAYS_"):
        return "신청일 기준 상대 일수와 부호 의미 유지"
    return "Stage 2에서 분포·결측 처리 기준 확정"


def _observed_values(stats: Mapping[str, Any]) -> str:
    if stats["min"] is not None or stats["max"] is not None:
        return f"{_format_value(stats['min'])} ~ {_format_value(stats['max'])}"
    categories = stats.get("top_categories") or []
    if not categories:
        return "-"
    values = [f"{item['value']}({item['count']:,})" for item in categories[:6]]
    suffix = " 외" if (stats.get("unique_category_count") or 0) > 6 else ""
    return ", ".join(values) + suffix


def render_data_dictionary(result: Mapping[str, Any], raw_dir: Path) -> str:
    """공식 설명과 관측 통계를 결합한 Markdown 데이터 사전을 만든다."""

    descriptions = _load_descriptions(Path(raw_dir))
    lines = [
        "# CreditLens 데이터 사전",
        "",
        "> Stage 1 원본 데이터 검증 결과와 Kaggle 공식 컬럼 설명을 결합한 초안입니다.",
        "",
        "## 작성 기준",
        "",
        f"- 생성 시각(UTC): `{result['generated_at_utc']}`",
        f"- 설명 원본: `data/raw/{DESCRIPTION_FILENAME}`",
        "- 원본 공식 설명은 의미 왜곡을 막기 위해 영문 그대로 보존했습니다.",
        "- 관측 자료형·결측률·범위는 현재 로컬 원본을 전체 스캔한 실제 값입니다.",
        "- 최종 전처리 방식은 Stage 2에서 확정합니다.",
        "",
        "## 테이블 관계",
        "",
        "- `application_train.csv`: 현재 신청 1건당 1행, `SK_ID_CURR` 유일",
        "- `bureau.csv`: 외부 신용거래 1건당 1행, `SK_ID_BUREAU` 유일, 고객당 여러 행",
        "- `installments_payments.csv`: 납부행위 1건당 1행, 분할납부 때문에 보장된 단일 행 기본키 없음",
        "- 보조 테이블에는 `application_test.csv` 고객도 포함되므로 train 외 고객은 원본 오류가 아닙니다.",
        "",
    ]
    for table_name in ("application", "bureau", "installments"):
        table = result["tables"][table_name]
        alias = DESCRIPTION_TABLE_ALIASES[table["filename"]]
        lines.extend(
            [
                f"## `{table['filename']}`",
                "",
                f"- 행 단위: {table['grain']}",
                f"- 크기: {table['row_count']:,}행 × {table['column_count']:,}열",
                "",
                "| 컬럼 | 의미 유형 | 관측 dtype | 결측 | 관측 범위/주요 값 | 공식 설명(영문) | Stage 2 메모 |",
                "|---|---|---|---:|---|---|---|",
            ]
        )
        for column in table["columns"]:
            stats = table["column_stats"][column]
            official = descriptions.get((alias, column), {})
            official_text = official.get("description") or "공식 설명 없음"
            if official.get("special"):
                official_text = f"{official_text} / Special: {official['special']}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{column}`",
                        _semantic_type(column, stats),
                        _escape_markdown(" / ".join(stats["dtypes"])),
                        f"{stats['missing_count']:,} ({stats['missing_rate']:.2%})",
                        _escape_markdown(_observed_values(stats)),
                        _escape_markdown(official_text),
                        _escape_markdown(_stage2_note(table_name, column, stats)),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_validation_report(result: Mapping[str, Any]) -> str:
    """검증 결과를 사람이 읽기 쉬운 한국어 Markdown 보고서로 만든다."""

    summary = result["summary"]
    lines = [
        "# CreditLens 원본 데이터 검증 보고서",
        "",
        f"> 최종 상태: **{summary['status']}** — 오류 {summary['error_count']}건, 경고 {summary['warning_count']}건, 정보 {summary['info_count']}건",
        "",
        "## 실행 정보",
        "",
        f"- 생성 시각(UTC): `{result['generated_at_utc']}`",
        f"- 청크 크기: `{result['chunk_size']:,}`행",
        f"- Python: `{result['environment']['python']}`",
        f"- Pandas: `{result['environment']['pandas']}`",
        f"- NumPy: `{result['environment']['numpy']}`",
        "- 검증 방식: 원본을 수정하지 않는 전체 청크 스캔",
        "",
        "## 원본 파일 Manifest",
        "",
        "| 파일 | 행 | 열 | 크기 | SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for table_name in ("application", "bureau", "installments"):
        table = result["tables"][table_name]
        lines.append(
            f"| `{table['display_path']}` | {table['row_count']:,} | {table['column_count']:,} | "
            f"{_format_bytes(table['file_size_bytes'])} | `{table['sha256']}` |"
        )
    description = result["description_manifest"]
    lines.append(
        f"| `{description['display_path']}` | {description['row_count']:,} | "
        f"{description['column_count']:,} | {_format_bytes(description['file_size_bytes'])} | "
        f"`{description['sha256']}` |"
    )

    lines.extend(
        [
            "",
            "## 스키마·키·중복 검증",
            "",
            "| 테이블 | 행 단위 | 기본키 | 키 결측 | 키 중복 | 동일 행 해시 반복 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for table_name in ("application", "bureau", "installments"):
        table = result["tables"][table_name]
        primary_key = f"`{table['primary_key']}`" if table["primary_key"] else "보장된 단일 키 없음"
        key_null = f"{table['primary_key_null_count']:,}" if table["primary_key"] else "-"
        key_duplicate = (
            f"{table['primary_key_duplicate_count']:,}" if table["primary_key"] else "-"
        )
        lines.append(
            f"| {table_name} | {_escape_markdown(table['grain'])} | {primary_key} | "
            f"{key_null} | {key_duplicate} | {table['duplicate_row_hash_count']:,} |"
        )
    dependency = result["tables"]["installments"]["dependency"]
    lines.extend(
        [
            "",
            f"- `SK_ID_PREV → SK_ID_CURR` 함수 종속 위반: **{dependency['violation_key_count']:,}개 키**",
            "- 동일 행 검사는 64비트 행 해시 기반 후보 탐지이며, 반복이 발견되면 Stage 2에서 실제 행을 재확인합니다.",
            "",
            "## 테이블 관계와 이력 커버리지",
            "",
            "| 보조 테이블 | 전체 고객 | train 매칭 고객 | train 고객 커버리지 | train 밖 고객 | train 매칭 행 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("bureau", "installments"):
        relation = result["relationships"][name]
        lines.append(
            f"| {name} | {relation['support_customer_count']:,} | "
            f"{relation['matched_application_customer_count']:,} | "
            f"{relation['application_customer_coverage_rate']:.2%} | "
            f"{relation['outside_application_customer_count']:,} | "
            f"{relation['matched_application_row_count']:,} ({relation['matched_application_row_rate']:.2%}) |"
        )
    overlap = result["history_overlap"]
    lines.extend(
        [
            "",
            "train 고객 이력 조합:",
            "",
            f"- 두 이력 모두 존재: {overlap['both_history_customer_count']:,}명 ({overlap['both_history_rate']:.2%})",
            f"- bureau만 존재: {overlap['bureau_only_customer_count']:,}명",
            f"- installments만 존재: {overlap['installments_only_customer_count']:,}명",
            f"- 두 이력 모두 없음: {overlap['neither_history_customer_count']:,}명 ({overlap['neither_history_rate']:.2%})",
            "",
            "보조 테이블에는 test 고객도 포함됩니다. Stage 3에서는 각각 고객 단위로 먼저 집계하고 `application_train`에 LEFT JOIN합니다.",
            "",
            "## TARGET 분포",
            "",
            "| TARGET | 의미 | 건수 | 비율 |",
            "|---:|---|---:|---:|",
        ]
    )
    application = result["tables"]["application"]
    target_counts = application["target_counts"] or {}
    for value, meaning in ((0, "정상 상환"), (1, "상환곤란")):
        count = int(target_counts.get(value, target_counts.get(str(value), 0)))
        lines.append(f"| {value} | {meaning} | {count:,} | {_safe_ratio(count, application['row_count']):.4%} |")

    lines.extend(["", "## 결측률 상위 컬럼", ""])
    for table_name in ("application", "bureau", "installments"):
        table = result["tables"][table_name]
        missing = sorted(
            table["column_stats"].items(), key=lambda item: item[1]["missing_rate"], reverse=True
        )[:10]
        lines.extend(
            [
                f"### {table_name}",
                "",
                "| 컬럼 | 결측 건수 | 결측률 |",
                "|---|---:|---:|",
            ]
        )
        for column, stats in missing:
            lines.append(f"| `{column}` | {stats['missing_count']:,} | {stats['missing_rate']:.2%} |")
        lines.append("")

    lines.extend(["## 주요 도메인 관찰", ""])
    domain_labels = {
        "days_employed_365243": "DAYS_EMPLOYED=365243 sentinel",
        "code_gender_xna": "CODE_GENDER=XNA",
        "family_status_unknown": "가족상태 Unknown",
        "own_car_age_missing_when_car_owned": "차량 보유 고객의 차량연식 결측",
        "negative_credit_sum_debt": "음수 외부 대출잔액",
        "negative_credit_sum_limit": "음수 신용한도",
        "positive_days_credit_update": "양수 DAYS_CREDIT_UPDATE",
        "active_with_enddate_fact": "Active이지만 실제 종료일 존재",
        "closed_without_enddate_fact": "Closed이지만 실제 종료일 결측",
        "payment_fields_both_missing": "실제 납부일·납부금액 동시 결측",
        "payment_fields_missing_mismatch": "실제 납부일·납부금액 결측 불일치",
        "payment_late": "예정일보다 늦은 납부",
        "payment_early": "예정일보다 이른 납부",
        "payment_on_time": "예정일 당일 납부",
        "payment_below_scheduled": "예정금액보다 적은 납부",
        "payment_above_scheduled": "예정금액보다 많은 납부",
        "payment_equal_scheduled": "예정금액과 같은 납부",
    }
    for table_name in ("application", "bureau", "installments"):
        lines.extend([f"### {table_name}", "", "| 관찰 항목 | 건수 |", "|---|---:|"])
        for code, count in result["tables"][table_name]["domain_observations"].items():
            lines.append(f"| {domain_labels.get(code, code)} | {count:,} |")
        lines.append("")

    lines.extend(["## 발견사항", "", "| 심각도 | 테이블 | 코드 | 내용 |", "|---|---|---|---|"])
    for item in result["findings"]:
        lines.append(
            f"| {item['severity']} | {item['table'] or '-'} | `{item['code']}` | {_escape_markdown(item['message'])} |"
        )
    lines.extend(
        [
            "",
            "## Stage 2 전달사항",
            "",
            "- `DAYS_EMPLOYED=365243`은 오류가 아니라 알려진 sentinel이므로 결측 처리와 별도 플래그를 검토합니다.",
            "- 음수 외부 대출잔액·신용한도는 업무 의미를 확인하기 전 삭제하지 않습니다.",
            "- 실제 납부일·금액 동시 결측은 미납 또는 미기록 가능성이 있어 단순 0 대치하지 않습니다.",
            "- 극단값은 삭제 근거가 아니며 분포와 업무 의미를 함께 검토합니다.",
            "- 데이터 분할과 누수 검토는 Stage 2에서 시작합니다.",
            "",
            "## 재실행 명령",
            "",
            "```bash",
            "PYTHONPATH=src .venv/bin/python -m creditlens.data.validate_raw --raw-dir data/raw",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: Mapping[str, Any],
    *,
    raw_dir: Path,
    json_path: Path,
    report_path: Path,
    dictionary_path: Path,
) -> None:
    """집계 결과와 두 Markdown 문서를 기록한다."""

    for path in (json_path, report_path, dictionary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_validation_report(result), encoding="utf-8")
    dictionary_path.write_text(render_data_dictionary(result, raw_dir), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--json", type=Path, default=Path("reports/data_validation.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/Data_Validation_Report.md"))
    parser.add_argument("--dictionary", type=Path, default=Path("docs/Data_Dictionary.md"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        result = validate_raw_data(args.raw_dir, chunk_size=args.chunk_size)
    except (RawDataValidationError, ValueError) as exc:
        print(f"검증 실행 실패: {exc}")
        return 2
    write_outputs(
        result,
        raw_dir=args.raw_dir,
        json_path=args.json,
        report_path=args.report,
        dictionary_path=args.dictionary,
    )
    summary = result["summary"]
    print(
        f"Stage 1 원본 검증 {summary['status']}: "
        f"ERROR={summary['error_count']}, WARNING={summary['warning_count']}, INFO={summary['info_count']}"
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
