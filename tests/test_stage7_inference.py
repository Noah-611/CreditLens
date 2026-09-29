"""합성 입력으로 계약과 추론을 검증한다. train/validation/test를 읽지 않는다."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from creditlens.inference.contract import InputContract, InputContractError, MAX_BATCH_SIZE
from creditlens.inference.demo import synthetic_request
from creditlens.inference.predictor import PredictionService
from creditlens.modeling.calibration import IdentityCalibrator
from creditlens.modeling.frozen_model import DEFAULT_BUNDLE, FrozenModelError, verify_bundle


@pytest.fixture
def manifest():
    snapshot = json.loads(Path("reports/stage7_input_contract.json").read_text())
    return {
        "release": snapshot["model_release"],
        "data_version": {"version": snapshot["feature_version"],
                         "schema_sha256": snapshot["feature_schema_sha256"]},
        "input_contract": {
            "features": [s["name"] for s in snapshot["fields"]],
            "numeric": [s["name"] for s in snapshot["fields"] if s["type"] != "string"],
            "categorical": [s["name"] for s in snapshot["fields"] if s["type"] == "string"],
        },
    }


@pytest.fixture
def contract(manifest):
    return InputContract(manifest)


@pytest.fixture
def payload(contract):
    return synthetic_request(contract)


class StubPipeline:
    """モデルなしCIでも検証するための未学習スタブ。fitは一切持たない。"""

    def __init__(self, contract, score=0.2):
        self.score, self.calls, self.frames = score, 0, []
        self.classes_ = np.array([0, 1])
        example = synthetic_request(contract)["records"][0]["features"]
        roles = SimpleNamespace(feature_columns=contract.features,
                                numeric_columns=contract.numeric,
                                categorical_columns=contract.categorical)
        rare = SimpleNamespace(seen_categories_=tuple(
            frozenset([example[n]]) for n in contract.categorical
        ))
        columns = SimpleNamespace(named_transformers_={
            "categorical": SimpleNamespace(named_steps={"rare": rare}),
        })
        self.named_steps = {"preprocessor": SimpleNamespace(named_steps={
            "contract": roles, "columns": columns,
        })}

    def predict_proba(self, frame):
        self.calls += 1
        self.frames.append(frame.copy(deep=True))
        return np.tile([1 - self.score, self.score], (len(frame), 1))


@pytest.fixture
def service(manifest, contract):
    policy = json.loads(Path("reports/stage6_final_results.json").read_text())["policy"]
    return PredictionService(manifest, StubPipeline(contract), IdentityCalibrator(), policy)


def test_snapshot_covers_all_features(contract):
    assert contract.describe() == json.loads(Path("reports/stage7_input_contract.json").read_text())
    assert (len(contract.features), len(contract.numeric), len(contract.categorical)) == (198, 184, 14)
    assert not set(contract.features) & {"TARGET", "SPLIT", "SK_ID_CURR", "CODE_GENDER"}


@pytest.mark.parametrize("invalid", [None, [], "not-an-object", 42])
def test_non_object_requests_are_contract_errors(contract, invalid):
    with pytest.raises(InputContractError, match="REQUEST_FIELDS"):
        contract.validate(invalid)


def test_explicit_nulls_use_frozen_missing_representation(contract, payload):
    f = payload["records"][0]["features"]
    f.update(AMT_INCOME_TOTAL=None, EXT_SOURCE_1=None, EXT_SOURCE_2=None,
             EXT_SOURCE_3=None, APP_EXT_SOURCE_OBSERVED_COUNT=0, APP_EXT_SOURCE_MEAN=None)
    _, frame = contract.validate(payload)
    assert pd.isna(frame.iloc[0]["AMT_INCOME_TOTAL"])
    assert pd.isna(frame.iloc[0]["APP_EXT_SOURCE_MEAN"])


def test_valid_prediction_is_json_safe_and_does_not_mutate_request(service, payload):
    before = copy.deepcopy(payload)
    result = service.predict(payload)
    assert payload == before
    prediction = result["predictions"][0]
    assert prediction["record_id"] == "synthetic-001"
    assert prediction["repayment_difficulty_probability"] == 0.2
    assert prediction["risk_band"] == "high"
    assert prediction["priority_review"] is True
    assert result["policy"]["automatic_credit_decision"] is False
    assert prediction["warnings"][0] == {"code": "MISSING_VALUES", "count": 138}
    json.dumps(result, allow_nan=False)


def test_key_order_batch_order_and_single_batch_equivalence(service, payload):
    one = service.predict(payload)["predictions"][0]
    second = copy.deepcopy(payload["records"][0])
    second["record_id"] = "synthetic-002"
    second["features"] = dict(reversed(list(second["features"].items())))
    payload["records"].insert(0, second)
    result = service.predict(payload)["predictions"]
    assert [r["record_id"] for r in result] == ["synthetic-002", "synthetic-001"]
    assert result[1] == one
    assert result[0] == {**one, "record_id": "synthetic-002"}
    frame = service._pipeline.frames[-1]
    assert tuple(frame.columns) == service.contract.features
    assert frame["OWN_CAR_AGE"].isna().all()
    assert frame["OCCUPATION_TYPE"].isna().all()


@pytest.mark.parametrize("change,code", [
    (lambda p: p.update(schema_version="v0"), "SCHEMA_VERSION"),
    (lambda p: p.update(feature_version="v1"), "FEATURE_VERSION"),
    (lambda p: p.update(model_path="untrusted.joblib"), "REQUEST_FIELDS"),
    (lambda p: p.update(records=[]), "BATCH_SIZE"),
    (lambda p: p.update(records=p["records"] * (MAX_BATCH_SIZE + 1)), "BATCH_SIZE"),
    (lambda p: p["records"].append(copy.deepcopy(p["records"][0])), "DUPLICATE_RECORD_ID"),
    (lambda p: p["records"][0].update(record_id=""), "RECORD_ID"),
    (lambda p: p["records"][0]["features"].pop("AMT_CREDIT"), "FEATURE_FIELDS"),
    (lambda p: p["records"][0]["features"].update(TARGET=1), "FEATURE_FIELDS"),
    (lambda p: p["records"][0]["features"].update(CODE_GENDER="M"), "FEATURE_FIELDS"),
])
def test_request_errors_before_any_prediction(service, payload, change, code):
    change(payload)
    with pytest.raises(InputContractError) as error:
        service.predict(payload)
    assert error.value.code == code
    assert service._pipeline.calls == 0


@pytest.mark.parametrize("field,value,code", [
    ("AMT_CREDIT", "540000", "NUMERIC_TYPE"),
    ("AMT_CREDIT", True, "NUMERIC_TYPE"),
    ("AMT_CREDIT", float("nan"), "NUMERIC_RANGE"),
    ("AMT_CREDIT", float("inf"), "NUMERIC_RANGE"),
    ("AMT_CREDIT", 1e100, "NUMERIC_RANGE"),
    ("AMT_CREDIT", 10 ** 1000, "NUMERIC_RANGE"),
    ("AMT_CREDIT", -1, "NUMERIC_RANGE"),
    ("EXT_SOURCE_1", 1.1, "NUMERIC_RANGE"),
    ("HOUR_APPR_PROCESS_START", 24, "NUMERIC_RANGE"),
    ("CNT_CHILDREN", 0.5, "INTEGER_REQUIRED"),
    ("OWN_CAR_AGE_MISSING", None, "NULL_NOT_ALLOWED"),
    ("BUREAU_RECORD_COUNT", None, "NULL_NOT_ALLOWED"),
    ("NAME_INCOME_TYPE", 42, "CATEGORY_VALUE"),
    ("NAME_INCOME_TYPE", "", "CATEGORY_VALUE"),
    ("NAME_INCOME_TYPE", "__MISSING__", "CATEGORY_VALUE"),
    ("NAME_INCOME_TYPE", "__RARE__", "CATEGORY_VALUE"),
    ("NAME_INCOME_TYPE", ["Working"], "CATEGORY_VALUE"),
    ("OWN_CAR_AGE_MISSING", 1, "CAR_AGE_FLAGS"),
    ("FLAG_OWN_CAR", "Y", "CAR_OWNERSHIP_FLAGS"),
    ("DAYS_EMPLOYED", 365243, "EMPLOYMENT_SENTINEL"),
    ("DAYS_EMPLOYED_SENTINEL", 1, "EMPLOYMENT_SENTINEL"),
    ("APP_EXT_SOURCE_MEAN", 0.9, "EXTERNAL_SCORE_SUMMARY"),
    ("APP_EXT_SOURCE_OBSERVED_COUNT", 2, "EXTERNAL_SCORE_SUMMARY"),
    ("BUREAU_RECORD_COUNT", 1, "HISTORY_FLAG_COUNT"),
    ("INST_HAS_HISTORY", 1, "HISTORY_FLAG_COUNT"),
    ("INST_LATE_RATIO", 0, "ABSENT_HISTORY_VALUES"),
])
def test_feature_errors(service, payload, field, value, code):
    payload["records"][0]["features"][field] = value
    with pytest.raises(InputContractError) as error:
        service.predict(payload)
    assert error.value.code == code
    assert error.value.row == 0
    assert service._pipeline.calls == 0


def test_unknown_category_warns_and_preserves_frozen_behavior(service, payload):
    payload["records"][0]["features"]["OCCUPATION_TYPE"] = "new-category-for-test"
    result = service.predict(payload)["predictions"][0]
    assert {"code": "UNKNOWN_CATEGORIES", "fields": ["OCCUPATION_TYPE"]} in result["warnings"]
    assert service._pipeline.frames[0].iloc[0]["OCCUPATION_TYPE"] == "new-category-for-test"
    assert "new-category-for-test" not in json.dumps(result)


def test_partial_batch_is_not_scored_and_values_are_not_in_errors(service, payload):
    bad = copy.deepcopy(payload["records"][0])
    bad["record_id"] = "synthetic-002"
    bad["features"]["AMT_CREDIT"] = "secret-value"
    payload["records"].append(bad)
    with pytest.raises(InputContractError) as error:
        service.predict(payload)
    assert error.value.row == 1
    assert "secret-value" not in str(error.value)
    assert error.value.as_dict() == {"code": "NUMERIC_TYPE", "row": 1, "field": "AMT_CREDIT"}
    assert service._pipeline.calls == 0


def test_valid_negative_debt_and_overpayment_are_not_clipped(contract, payload):
    f = payload["records"][0]["features"]
    f.update(BUREAU_HAS_HISTORY=1, BUREAU_RECORD_COUNT=1, BUREAU_DEBT_SUM=-50,
             INST_HAS_HISTORY=1, INST_SCHEDULE_COUNT=1, INST_PAYMENT_RATIO=1.2)
    _, frame = contract.validate(payload)
    assert frame.iloc[0]["BUREAU_DEBT_SUM"] == -50
    assert frame.iloc[0]["INST_PAYMENT_RATIO"] == 1.2


@pytest.mark.parametrize("boundary", ["medium_cutoff", "high_cutoff"])
def test_full_precision_boundaries(service, payload, boundary):
    cutoff = service._policy[boundary]
    service._pipeline.score = cutoff
    at = service.predict(payload)["predictions"][0]
    service._pipeline.score = np.nextafter(cutoff, 0)
    below = service.predict(payload)["predictions"][0]
    assert at["risk_band"] == ("medium" if boundary == "medium_cutoff" else "high")
    assert below["risk_band"] == ("low" if boundary == "medium_cutoff" else "medium")


def test_verification_failure_does_not_load_model(monkeypatch):
    from creditlens.inference import predictor
    def fail():
        raise FrozenModelError("changed bundle")
    monkeypatch.setattr(predictor, "verify_bundle", fail)
    monkeypatch.setattr(predictor, "load_frozen_model", lambda: pytest.fail("검증 전 로드"))
    with pytest.raises(FrozenModelError):
        PredictionService.from_frozen()


def test_reject_changed_roles(manifest):
    manifest["input_contract"]["features"][0] = "TARGET"
    with pytest.raises(ValueError, match="피처 계약"):
        InputContract(manifest)


def test_real_frozen_model_matches_direct_prediction_without_data(monkeypatch):
    if not (DEFAULT_BUNDLE / "manifest.json").exists():
        pytest.skip("Git에서 제외한 로컬 동결 모델이 없는 환경")
    from creditlens.modeling import data
    def forbidden(*args, **kwargs):
        pytest.fail("추론 테스트는 데이터셋을 읽거나 fit하면 안 됩니다.")
    monkeypatch.setattr(data, "load_model_split", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    service = PredictionService.from_frozen()
    monkeypatch.setattr(service._pipeline, "fit", forbidden)
    monkeypatch.setattr(service._pipeline.named_steps["preprocessor"], "fit", forbidden)
    monkeypatch.setattr(service._pipeline.named_steps["preprocessor"], "fit_transform", forbidden)
    payload = synthetic_request(service.contract)
    _, frame = service.contract.validate(payload)
    expected = service._calibrator.predict(service._pipeline.predict_proba(frame)[:, 1])
    result = service.predict(payload)
    assert result["predictions"][0]["repayment_difficulty_probability"] == expected[0]
    assert result["predictions"][0]["risk_band"] == "low"
    original = result["predictions"][0]
    second = copy.deepcopy(payload["records"][0])
    second["record_id"] = "synthetic-002"
    second["features"]["OCCUPATION_TYPE"] = "new-synthetic-category"
    second["features"].update(FLAG_OWN_CAR="Y", OWN_CAR_AGE_MISSING=1,
                              OWN_CAR_AGE_NOT_APPLICABLE=0)
    payload["records"].append(second)
    batch = service.predict(payload)["predictions"]
    assert batch[0] == original
    single = service.predict({**payload, "records": [second]})["predictions"][0]
    assert batch[1] == single
    assert {"code": "UNKNOWN_CATEGORIES", "fields": ["OCCUPATION_TYPE"]} in single["warnings"]
    snapshot = json.loads(Path("reports/stage7_input_contract.json").read_text())
    assert service.contract.describe() == snapshot
    # 예측 후에도 모델·정책·전처리 소스·환경의 동결 해시는 그대로다.
    assert verify_bundle()["status"] == "frozen"
