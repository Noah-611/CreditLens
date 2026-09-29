"""실제 고객과 관계없는 수작업 합성 예시. 학습/validation/test 파일은 읽지 않는다."""

from __future__ import annotations

from creditlens.inference.contract import InputContract, SCHEMA_VERSION, FEATURE_VERSION


def synthetic_request(contract: InputContract) -> dict:
    features = {s["name"]: None if s["nullable"] else 0 for s in contract.specs}
    features.update({
        "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2,
        "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 540000,
        "AMT_ANNUITY": 27000, "AMT_GOODS_PRICE": 500000,
        "DAYS_BIRTH": -14610, "DAYS_EMPLOYED": -3652.5,
        "DAYS_EMPLOYED_SENTINEL": 0, "FLAG_OWN_CAR": "N",
        "OWN_CAR_AGE_NOT_APPLICABLE": 1, "OWN_CAR_AGE_MISSING": 0,
        "APP_CREDIT_INCOME_RATIO": 3, "APP_ANNUITY_INCOME_RATIO": 0.15,
        "APP_CREDIT_ANNUITY_RATIO": 20, "APP_CREDIT_GOODS_RATIO": 500000 / 540000,
        "APP_INCOME_PER_FAMILY_MEMBER": 90000, "APP_AGE_YEARS": 40,
        "APP_EMPLOYED_YEARS": 10, "APP_EMPLOYED_AGE_RATIO": 0.25,
        "EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.55,
        "APP_EXT_SOURCE_OBSERVED_COUNT": 3, "APP_EXT_SOURCE_MEAN": 0.55,
        "NAME_CONTRACT_TYPE": "Cash loans", "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working", "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
    })
    return {
        "schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
        "records": [{"record_id": "synthetic-001", "features": features}],
    }
