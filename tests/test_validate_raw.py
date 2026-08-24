"""Stage 1 원본 데이터 검증기의 회귀 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from creditlens.data.validate_raw import (
    APPLICATION_COLUMNS,
    BUREAU_COLUMNS,
    INSTALLMENTS_COLUMNS,
    RawDataValidationError,
    TableSpec,
    validate_raw_data,
    write_outputs,
)


def _application_rows(
    customer_ids: list[int], targets: list[int] | None = None
) -> pd.DataFrame:
    targets = targets or [index % 2 for index in range(len(customer_ids))]
    rows: list[dict[str, object]] = []
    for customer_id, target in zip(customer_ids, targets, strict=True):
        row: dict[str, object] = dict.fromkeys(APPLICATION_COLUMNS, 0)
        row.update(
            {
                "SK_ID_CURR": customer_id,
                "TARGET": target,
                "DAYS_BIRTH": -12_000,
                "DAYS_EMPLOYED": -1_000,
                "HOUR_APPR_PROCESS_START": 12,
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "N",
                "CODE_GENDER": "F",
                "NAME_FAMILY_STATUS": "Married",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=APPLICATION_COLUMNS)


def _bureau_rows(customer_ids: list[int], bureau_ids: list[int] | None = None) -> pd.DataFrame:
    bureau_ids = bureau_ids or list(range(200_001, 200_001 + len(customer_ids)))
    rows: list[dict[str, object]] = []
    for customer_id, bureau_id in zip(customer_ids, bureau_ids, strict=True):
        row: dict[str, object] = dict.fromkeys(BUREAU_COLUMNS, 0)
        row.update(
            {
                "SK_ID_CURR": customer_id,
                "SK_ID_BUREAU": bureau_id,
                "CREDIT_ACTIVE": "Closed",
                "DAYS_CREDIT": -100,
                "DAYS_ENDDATE_FACT": -10,
                "DAYS_CREDIT_UPDATE": -5,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=BUREAU_COLUMNS)


def _installment_rows(
    customer_ids: list[int], previous_ids: list[int] | None = None
) -> pd.DataFrame:
    previous_ids = previous_ids or list(range(300_001, 300_001 + len(customer_ids)))
    rows: list[dict[str, object]] = []
    for customer_id, previous_id in zip(customer_ids, previous_ids, strict=True):
        row: dict[str, object] = dict.fromkeys(INSTALLMENTS_COLUMNS, 0)
        row.update(
            {
                "SK_ID_PREV": previous_id,
                "SK_ID_CURR": customer_id,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 1,
                "DAYS_INSTALMENT": -20,
                "DAYS_ENTRY_PAYMENT": -20,
                "AMT_INSTALMENT": 100.0,
                "AMT_PAYMENT": 100.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=INSTALLMENTS_COLUMNS)


def _write_tables(
    raw_dir: Path,
    *,
    application: pd.DataFrame | None = None,
    bureau: pd.DataFrame | None = None,
    installments: pd.DataFrame | None = None,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (application if application is not None else _application_rows([100_001, 100_002])).to_csv(
        raw_dir / "application_train.csv", index=False
    )
    (bureau if bureau is not None else _bureau_rows([100_001, 100_002])).to_csv(
        raw_dir / "bureau.csv", index=False
    )
    (
        installments
        if installments is not None
        else _installment_rows([100_001, 100_002])
    ).to_csv(raw_dir / "installments_payments.csv", index=False)
    description = (
        ",Table,Row,Description,Special\n"
        "1,application_{train|test}.csv,SK_ID_CURR,Client identifier,\n"
        "2,bureau.csv,SK_BUREAU_ID,Bureau identifier,hashed\n"
        "3,installments_payments.csv,SK_ID_PREV,Previous loan identifier,\n"
    )
    (raw_dir / "HomeCredit_columns_description.csv").write_text(
        description, encoding="utf-8"
    )


def _test_specs() -> tuple[TableSpec, ...]:
    """운영 manifest의 행 수·체크섬과 분리된 작은 CSV 계약을 만든다."""

    return (
        TableSpec(
            name="application",
            filename="application_train.csv",
            columns=APPLICATION_COLUMNS,
            primary_key="SK_ID_CURR",
            grain="테스트 신청 1건당 1행",
        ),
        TableSpec(
            name="bureau",
            filename="bureau.csv",
            columns=BUREAU_COLUMNS,
            primary_key="SK_ID_BUREAU",
            grain="테스트 외부 신용거래 1건당 1행",
        ),
        TableSpec(
            name="installments",
            filename="installments_payments.csv",
            columns=INSTALLMENTS_COLUMNS,
            primary_key=None,
            grain="테스트 납부행위 1건당 1행",
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finding_codes(result: dict[str, object]) -> set[str]:
    findings = result["findings"]
    assert isinstance(findings, list)
    return {str(item["code"]) for item in findings}


def test_valid_data_allows_support_customers_outside_application(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_tables(
        raw_dir,
        bureau=_bureau_rows([100_001, 999_001]),
        installments=_installment_rows([100_002, 999_002]),
    )

    result = validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())

    assert result["summary"]["status"] == "PASS"
    assert result["relationships"]["bureau"]["outside_application_customer_count"] == 1
    assert result["relationships"]["installments"]["outside_application_customer_count"] == 1
    assert "support_table_scope" in _finding_codes(result)


def test_primary_key_duplicate_is_detected_across_chunk_boundary(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    application = _application_rows([100_001, 100_002, 100_002], [0, 0, 1])
    _write_tables(raw_dir, application=application)

    result = validate_raw_data(raw_dir, chunk_size=2, specs=_test_specs())

    table = result["tables"]["application"]
    assert table["primary_key_duplicate_count"] == 1
    assert result["summary"]["status"] == "FAIL"
    assert "primary_key_duplicate" in _finding_codes(result)


def test_invalid_target_infinity_and_domain_rules_are_errors(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    application = _application_rows([100_001, 100_002], [0, 1])
    application["AMT_CREDIT"] = application["AMT_CREDIT"].astype(float)
    application.loc[1, "AMT_CREDIT"] = float("inf")
    application.loc[1, "DAYS_BIRTH"] = 1
    application["TARGET"] = application["TARGET"].astype("object")
    application.loc[1, "TARGET"] = "invalid"
    application["FLAG_MOBIL"] = application["FLAG_MOBIL"].astype("object")
    application.loc[1, "FLAG_MOBIL"] = "invalid"
    application["AMT_GOODS_PRICE"] = application["AMT_GOODS_PRICE"].astype("object")
    application.loc[1, "AMT_GOODS_PRICE"] = "bad"
    _write_tables(raw_dir, application=application)

    result = validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())

    application_result = result["tables"]["application"]
    rules = {rule["code"]: rule["violations"] for rule in application_result["rules"]}
    assert application_result["column_stats"]["AMT_CREDIT"]["infinite_count"] == 1
    assert rules["target_domain"] == 1
    assert rules["days_birth_negative"] == 1
    assert rules["flag_mobil_binary"] == 1
    assert rules["amt_goods_price_numeric"] == 1
    assert {"infinite_numeric_value", "rule_violation", "target_invalid"} <= _finding_codes(
        result
    )
    assert result["summary"]["status"] == "FAIL"


def test_child_table_identifier_nulls_are_errors(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    bureau = _bureau_rows([100_001, 100_002])
    bureau["SK_ID_CURR"] = bureau["SK_ID_CURR"].astype("object")
    bureau.loc[1, "SK_ID_CURR"] = None
    installments = _installment_rows([100_001, 100_002])
    installments["SK_ID_PREV"] = installments["SK_ID_PREV"].astype("object")
    installments.loc[1, "SK_ID_PREV"] = None
    _write_tables(raw_dir, bureau=bureau, installments=installments)

    result = validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())

    bureau_rules = {
        rule["code"]: rule["violations"] for rule in result["tables"]["bureau"]["rules"]
    }
    installment_rules = {
        rule["code"]: rule["violations"]
        for rule in result["tables"]["installments"]["rules"]
    }
    assert bureau_rules["sk_id_curr_not_null"] == 1
    assert installment_rules["sk_id_prev_not_null"] == 1
    assert result["summary"]["status"] == "FAIL"


def test_identical_row_hash_survives_chunk_dtype_drift(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    application = _application_rows(
        [100_001, 100_002, 100_001, 100_003], [0, 0, 0, 1]
    )
    application["OWN_CAR_AGE"] = application["OWN_CAR_AGE"].astype("object")
    application.loc[:, "OWN_CAR_AGE"] = [1, "bad", 1, None]
    _write_tables(raw_dir, application=application)

    result = validate_raw_data(raw_dir, chunk_size=2, specs=_test_specs())

    table = result["tables"]["application"]
    assert table["duplicate_row_hash_count"] == 1
    assert len(table["column_stats"]["OWN_CAR_AGE"]["dtypes"]) == 2
    assert "dtype_drift" in _finding_codes(result)


def test_installment_previous_id_dependency_is_checked_across_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    installments = _installment_rows([100_001, 100_002], [300_001, 300_001])
    _write_tables(raw_dir, installments=installments)

    result = validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())

    dependency = result["tables"]["installments"]["dependency"]
    assert dependency["violation_key_count"] == 1
    assert "previous_customer_dependency" in _finding_codes(result)
    assert result["summary"]["status"] == "FAIL"


def test_identical_row_hash_repetition_is_reported(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    installments = _installment_rows([100_001])
    installments = pd.concat([installments, installments], ignore_index=True)
    _write_tables(raw_dir, installments=installments)

    result = validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())

    assert result["tables"]["installments"]["duplicate_row_hash_count"] == 1
    assert "duplicate_row_hash" in _finding_codes(result)
    assert result["summary"]["warning_count"] >= 1


def test_missing_required_file_raises_contract_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _application_rows([100_001]).to_csv(raw_dir / "application_train.csv", index=False)

    with pytest.raises(RawDataValidationError, match="필수 원본 파일.*bureau.csv"):
        validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())


def test_missing_description_file_raises_contract_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_tables(raw_dir)
    (raw_dir / "HomeCredit_columns_description.csv").unlink()

    with pytest.raises(
        RawDataValidationError,
        match="필수 원본 파일.*HomeCredit_columns_description.csv",
    ):
        validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())


def test_schema_mismatch_raises_contract_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    application = _application_rows([100_001]).drop(columns="TARGET")
    _write_tables(raw_dir, application=application)

    with pytest.raises(RawDataValidationError, match=r"스키마 불일치.*TARGET"):
        validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())


def test_outputs_hide_local_paths_and_credentials_and_preserve_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = "test-only-credential-value"
    raw_dir = tmp_path / credential / "raw"
    _write_tables(raw_dir)
    description = (
        ",Table,Row,Description,Special\n"
        "1,application_{train|test}.csv,SK_ID_CURR,Client identifier\u2026,\n"
        "2,bureau.csv,SK_BUREAU_ID,Bureau identifier,hashed\n"
    )
    (raw_dir / "HomeCredit_columns_description.csv").write_bytes(
        description.encode("windows-1252")
    )
    source_paths = sorted(raw_dir.glob("*.csv"))
    checksums_before = {path.name: _sha256(path) for path in source_paths}
    monkeypatch.setenv("KAGGLE_KEY", credential)

    result = validate_raw_data(raw_dir, chunk_size=1, specs=_test_specs())
    output_dir = tmp_path / "outputs"
    json_path = output_dir / "validation.json"
    report_path = output_dir / "Validation.md"
    dictionary_path = output_dir / "Dictionary.md"
    write_outputs(
        result,
        raw_dir=raw_dir,
        json_path=json_path,
        report_path=report_path,
        dictionary_path=dictionary_path,
    )

    output_texts = [
        json_path.read_text(encoding="utf-8"),
        report_path.read_text(encoding="utf-8"),
        dictionary_path.read_text(encoding="utf-8"),
    ]
    for output_text in output_texts:
        assert str(tmp_path.resolve()) not in output_text
        assert credential not in output_text
    assert "Client identifier" in output_texts[2]
    assert "Bureau identifier" in output_texts[2]
    assert "공식 설명 파일에는 SK_BUREAU_ID로 표기" in output_texts[2]
    assert "`AMT_REQ_CREDIT_BUREAU_HOUR` | 조회 횟수" in output_texts[2]
    assert "`data/raw/HomeCredit_columns_description.csv`" in output_texts[1]
    assert "| installments |" in output_texts[1]
    assert "보장된 단일 키 없음 | - | - |" in output_texts[1]
    assert json.loads(output_texts[0])["tables"]["application"]["display_path"] == (
        "data/raw/application_train.csv"
    )

    checksums_after = {path.name: _sha256(path) for path in source_paths}
    assert checksums_after == checksums_before
    for table_name, filename in (
        ("application", "application_train.csv"),
        ("bureau", "bureau.csv"),
        ("installments", "installments_payments.csv"),
    ):
        assert result["tables"][table_name]["sha256"] == checksums_before[filename]
