"""Stage 7의 JSON 호환 입력 계약. 값을 추측하거나 피처를 새로 학습하지 않는다."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from creditlens.modeling.preprocessing import (
    MISSING_CATEGORY, RARE_CATEGORY, PreprocessingContractError,
    SemanticValueTransformer,
)

SCHEMA_VERSION = "creditlens-inference-v1"
FEATURE_VERSION = "v3"
MAX_BATCH_SIZE = 1000
FLOAT32_MAX = float(np.finfo(np.float32).max)
CAR_FLAGS = {"OWN_CAR_AGE_NOT_APPLICABLE", "OWN_CAR_AGE_MISSING"}
UNIT_INTERVAL = {
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "APP_EXT_SOURCE_MEAN",
    "REGION_POPULATION_RELATIVE", "BUREAU_ACTIVE_RATIO", "BUREAU_OVERDUE_LOAN_RATIO",
    "INST_MISSING_PAYMENT_RATIO", "INST_LATE_RATIO", "INST_UNDERPAID_RATIO",
    "INST_LAST_365_LATE_RATIO", "INST_LAST_730_LATE_RATIO",
}


class InputContractError(ValueError):
    """입력값 자체를 메시지에 노출하지 않는 호출자 입력 오류."""

    def __init__(self, code: str, *, row: int | None = None, field: str | None = None):
        self.code, self.row, self.field = code, row, field
        super().__init__(f"{code} (row={row}, field={field})")

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "row": self.row, "field": self.field}


def _history_count(name: str) -> bool:
    return name.startswith(("BUREAU_", "INST_")) and (
        name.endswith("_COUNT") or name == "BUREAU_PROLONG_COUNT_SUM"
    )


def field_spec(name: str, numeric: bool) -> dict[str, Any]:
    """Stage 3 산식으로 확정할 수 있는 범위만 강제한다. 경험적 최솟값은 쓰지 않는다."""
    spec: dict[str, Any] = {
        "name": name, "type": "number" if numeric else "string",
        "required": True, "nullable": True,
    }
    if not numeric:
        return {**spec, "min_length": 1, "max_length": 128,
                "reserved_values": [MISSING_CATEGORY, RARE_CATEGORY]}
    spec.update(minimum=-FLOAT32_MAX, maximum=FLOAT32_MAX)
    binary = (name.startswith("FLAG_") or name in CAR_FLAGS
              or name.endswith("_HAS_HISTORY") or name == "DAYS_EMPLOYED_SENTINEL"
              or name.startswith(("REG_REGION_", "LIVE_REGION_", "REG_CITY_", "LIVE_CITY_"))
              or name == "EMERGENCYSTATE_MODE")
    if binary:
        spec.update(type="integer", minimum=0, maximum=1)
    if name in CAR_FLAGS or name.endswith("_HAS_HISTORY") or _history_count(name):
        spec["nullable"] = False
    if _history_count(name) or name in {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}:
        spec.update(type="integer", minimum=0)
    if name in UNIT_INTERVAL:
        spec.update(minimum=0, maximum=1)
    if name == "APP_EXT_SOURCE_OBSERVED_COUNT":
        spec.update(type="integer", minimum=0, maximum=3, nullable=False)
    if name == "HOUR_APPR_PROCESS_START":
        spec.update(type="integer", minimum=0, maximum=23)
    if name in {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                "OWN_CAR_AGE", "APP_AGE_YEARS", "APP_EMPLOYED_YEARS"}:
        spec["minimum"] = 0
    # BUREAU_DEBT_*는 음수 원본을 보존한다. INST_PAYMENT_RATIO는 1을 넘을 수 있다.
    return spec


class InputContract:
    def __init__(self, manifest: dict[str, Any]):
        roles = manifest["input_contract"]
        self.features = tuple(roles["features"])
        self.numeric = tuple(roles["numeric"])
        self.categorical = tuple(roles["categorical"])
        if (manifest["data_version"]["version"] != FEATURE_VERSION
                or len(self.features) != 198 or len(set(self.features)) != 198
                or len(self.numeric) != 184 or len(self.categorical) != 14
                or set(self.numeric) & set(self.categorical)
                or set(self.features) != set(self.numeric) | set(self.categorical)
                or set(self.features) & {"SK_ID_CURR", "TARGET", "SPLIT", "CODE_GENDER"}):
            raise ValueError("지원하지 않는 동결 피처 계약")
        self.release = manifest["release"]
        self.feature_schema_sha256 = manifest["data_version"]["schema_sha256"]
        self.specs = [field_spec(name, name in self.numeric) for name in self.features]

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
            "model_release": self.release,
            "feature_schema_sha256": self.feature_schema_sha256,
            "max_batch_size": MAX_BATCH_SIZE,
            "unknown_category_policy": "warn_and_use_frozen_encoder",
            "fields": self.specs,
        }

    def validate(self, payload: Any) -> tuple[list[str], pd.DataFrame]:
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "feature_version", "records",
        }:
            raise InputContractError("REQUEST_FIELDS")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise InputContractError("SCHEMA_VERSION")
        if payload["feature_version"] != FEATURE_VERSION:
            raise InputContractError("FEATURE_VERSION")
        records = payload["records"]
        if not isinstance(records, list) or not 1 <= len(records) <= MAX_BATCH_SIZE:
            raise InputContractError("BATCH_SIZE")
        ids, values = [], []
        seen_ids: set[str] = set()
        for row, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {"record_id", "features"}:
                raise InputContractError("RECORD_FIELDS", row=row)
            record_id = record["record_id"]
            if (not isinstance(record_id, str) or not 1 <= len(record_id) <= 128
                    or not record_id.strip() or any(ord(c) < 32 for c in record_id)):
                raise InputContractError("RECORD_ID", row=row)
            if record_id in seen_ids:
                raise InputContractError("DUPLICATE_RECORD_ID", row=row)
            seen_ids.add(record_id)
            features = record["features"]
            if not isinstance(features, dict) or set(features) != set(self.features):
                raise InputContractError("FEATURE_FIELDS", row=row)
            for spec in self.specs:
                name, value = spec["name"], features[spec["name"]]
                if value is None:
                    if not spec["nullable"]:
                        raise InputContractError("NULL_NOT_ALLOWED", row=row, field=name)
                    continue
                if spec["type"] == "string":
                    if (not isinstance(value, str) or not value.strip()
                            or not 1 <= len(value) <= 128
                            or value in {MISSING_CATEGORY, RARE_CATEGORY}):
                        raise InputContractError("CATEGORY_VALUE", row=row, field=name)
                else:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise InputContractError("NUMERIC_TYPE", row=row, field=name)
                    try:
                        finite = math.isfinite(value)
                    except OverflowError:
                        finite = False
                    if not finite or not spec["minimum"] <= value <= spec["maximum"]:
                        raise InputContractError("NUMERIC_RANGE", row=row, field=name)
                    if spec["type"] == "integer" and value != math.floor(value):
                        raise InputContractError("INTEGER_REQUIRED", row=row, field=name)
            self._validate_relations(features, row)
            ids.append(record_id)
            values.append(features)
        # 호출자의 딕셔너리를 변경하지 않고 학습 시점의 순서·결측 표현으로 맞춘다.
        frame = pd.DataFrame(values, columns=self.features)
        for name in self.numeric:
            frame[name] = pd.Series([v[name] for v in values], dtype="float64")
        for name in self.categorical:
            frame[name] = pd.Series([v[name] if v[name] is not None else np.nan
                                     for v in values], dtype=object)
        return ids, frame

    @staticmethod
    def _validate_relations(f: dict[str, Any], row: int) -> None:
        try:
            car_columns = ("OWN_CAR_AGE", "OWN_CAR_AGE_NOT_APPLICABLE", "OWN_CAR_AGE_MISSING")
            SemanticValueTransformer()._validated_car_values(
                pd.DataFrame([{name: f[name] for name in car_columns}])
            )
        except PreprocessingContractError:
            raise InputContractError("CAR_AGE_FLAGS", row=row) from None
        if ((f["OWN_CAR_AGE_NOT_APPLICABLE"] == 1 and f["FLAG_OWN_CAR"] != "N")
                or (f["OWN_CAR_AGE_MISSING"] == 1 and f["FLAG_OWN_CAR"] != "Y")):
            raise InputContractError("CAR_OWNERSHIP_FLAGS", row=row)
        if f["DAYS_EMPLOYED"] == 365243 or (
            f["DAYS_EMPLOYED_SENTINEL"] == 1 and f["DAYS_EMPLOYED"] is not None
        ):
            raise InputContractError("EMPLOYMENT_SENTINEL", row=row)
        observed = [f[n] for n in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
                    if f[n] is not None]
        mean = sum(observed) / len(observed) if observed else None
        actual = f["APP_EXT_SOURCE_MEAN"]
        if (f["APP_EXT_SOURCE_OBSERVED_COUNT"] != len(observed)
                or (mean is None) != (actual is None)
                or (mean is not None and not math.isclose(mean, actual, rel_tol=1e-6, abs_tol=1e-8))):
            raise InputContractError("EXTERNAL_SCORE_SUMMARY", row=row)
        for prefix, count in (("BUREAU_", "BUREAU_RECORD_COUNT"),
                              ("INST_", "INST_SCHEDULE_COUNT")):
            has_history = f[prefix + "HAS_HISTORY"]
            if has_history != int(f[count] > 0):
                raise InputContractError("HISTORY_FLAG_COUNT", row=row, field=count)
            if not has_history:
                for name, value in f.items():
                    if name.startswith(prefix) and not name.endswith("_HAS_HISTORY"):
                        valid = value == 0 if _history_count(name) else value is None
                        if not valid:
                            raise InputContractError("ABSENT_HISTORY_VALUES", row=row, field=name)
