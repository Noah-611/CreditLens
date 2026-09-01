"""Stage 3 V1·V2·V3 고객 분석 마트 구축 회귀 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from creditlens.data.build_feature_mart import (
    DEFAULT_APPLICATION_INPUT,
    DEFAULT_BUREAU_INPUT,
    DEFAULT_INSTALLMENTS_INPUT,
    DEFAULT_SPLITS_INPUT,
    DEFAULT_V1_OUTPUT,
    DEFAULT_V2_OUTPUT,
    DEFAULT_V3_OUTPUT,
    FeatureMartError,
    build_feature_marts,
)


def _application_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003, 100004],
            "TARGET": [1, 0, 0, 1],
            "NAME_CONTRACT_TYPE": ["Cash loans"] * 4,
            "CODE_GENDER": ["F", "M", "F", "M"],
            "FLAG_OWN_CAR": ["N", "Y", "Y", "N"],
            "AMT_INCOME_TOTAL": [100.0, 200.0, 0.0, 400.0],
            "AMT_CREDIT": [200.0, 100.0, 50.0, 400.0],
            "AMT_ANNUITY": [20.0, 10.0, 0.0, 40.0],
            "AMT_GOODS_PRICE": [160.0, 100.0, 0.0, 320.0],
            "CNT_FAM_MEMBERS": [2.0, 1.0, 0.0, 4.0],
            "DAYS_BIRTH": [-36525, -14610, -10958, -18262],
            "DAYS_EMPLOYED": [-3650, 365243, -1000, -2000],
            "OWN_CAR_AGE": [np.nan, np.nan, 5.0, np.nan],
            "EXT_SOURCE_1": [0.1, np.nan, 0.3, 0.4],
            "EXT_SOURCE_2": [0.2, np.nan, 0.6, 0.5],
            "EXT_SOURCE_3": [np.nan, np.nan, 0.9, 0.6],
        }
    )


def _split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003, 100004],
            "TARGET": [1, 0, 0, 1],
            "SPLIT": ["train", "validation", "test", "train"],
        }
    )


def _bureau_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100001, 100001, 100002, 999999],
            "SK_ID_BUREAU": [5000001, 5000002, 5000004, 5000003, 5999999],
            "CREDIT_ACTIVE": ["Active", "Closed", "Sold", "Active", "Closed"],
            "CREDIT_CURRENCY": [
                "currency 1",
                "currency 1",
                "currency 1",
                "currency 2",
                "currency 1",
            ],
            "CREDIT_TYPE": [
                "Credit card",
                "Consumer credit",
                "Microloan",
                "Mortgage",
                "Car loan",
            ],
            "DAYS_CREDIT": [-100, -500, -300, -200, -50],
            "CREDIT_DAY_OVERDUE": [0, 5, 99, 0, 0],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 30.0, 99.0, np.nan, 0.0],
            "CNT_CREDIT_PROLONG": [0, 1, 9, 0, 0],
            "AMT_CREDIT_SUM": [1000.0, 500.0, 777.0, 999.0, 1.0],
            "AMT_CREDIT_SUM_DEBT": [200.0, np.nan, 777.0, 999.0, 0.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 50.0, 99.0, 0.0, 0.0],
            "DAYS_CREDIT_UPDATE": [-1, -2, 1, -10, -1],
            "AMT_ANNUITY": [100.0, np.nan, 77.0, 10.0, 1.0],
        }
    )


def _installments_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_PREV": [2000001, 2000001, 2000001, 2000002, 2000003, 2999999],
            "SK_ID_CURR": [100001, 100001, 100001, 100002, 100003, 999999],
            "NUM_INSTALMENT_VERSION": [1.0] * 6,
            "NUM_INSTALMENT_NUMBER": [1, 1, 2, 1, 1, 1],
            "DAYS_INSTALMENT": [-100.0, -100.0, -50.0, -20.0, -30.0, -10.0],
            "DAYS_ENTRY_PAYMENT": [-100.0, -90.0, np.nan, -10.0, -40.0, -10.0],
            "AMT_INSTALMENT": [100.0, 100.0, 100.0, 100.0, 100.0, 1.0],
            "AMT_PAYMENT": [40.0, 60.0, np.nan, 0.0, 120.0, 1.0],
        }
    )


def _write_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "application": root / "raw" / "application_train.csv",
        "bureau": root / "raw" / "bureau.csv",
        "installments": root / "raw" / "installments_payments.csv",
        "splits": root / "interim" / "customer_splits.csv",
    }
    frames = {
        "application": _application_frame(),
        "bureau": _bureau_frame(),
        "installments": _installments_frame(),
        "splits": _split_frame(),
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(path, index=False)
    return paths


def _outputs(root: Path) -> dict[str, Path]:
    return {
        "v1": root / "processed" / "v1.parquet",
        "v2": root / "processed" / "v2.parquet",
        "v3": root / "processed" / "v3.parquet",
        "summary": root / "reports" / "summary.json",
        "temp": root / "interim" / "duckdb_tmp",
    }


def _build(root: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, object]]:
    sources = _write_inputs(root)
    outputs = _outputs(root)
    summary = build_feature_marts(
        sources["application"],
        sources["bureau"],
        sources["installments"],
        sources["splits"],
        v1_output=outputs["v1"],
        v2_output=outputs["v2"],
        v3_output=outputs["v3"],
        summary_output=outputs["summary"],
        temp_dir=outputs["temp"],
        memory_limit="512MB",
        threads=1,
    )
    return sources, outputs, summary


def _read(path: Path) -> pd.DataFrame:
    with duckdb.connect(database=":memory:") as connection:
        return connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY SK_ID_CURR", [str(path)]
        ).fetchdf()


def test_end_to_end_marts_preserve_customers_and_reconcile_features(
    tmp_path: Path,
) -> None:
    sources = _write_inputs(tmp_path)
    source_bytes = {name: path.read_bytes() for name, path in sources.items()}
    outputs = _outputs(tmp_path)

    summary = build_feature_marts(
        sources["application"],
        sources["bureau"],
        sources["installments"],
        sources["splits"],
        v1_output=outputs["v1"],
        v2_output=outputs["v2"],
        v3_output=outputs["v3"],
        summary_output=outputs["summary"],
        temp_dir=outputs["temp"],
        memory_limit="512MB",
        threads=1,
    )

    assert {name: path.read_bytes() for name, path in sources.items()} == source_bytes
    v1, v2, v3 = (_read(outputs[name]) for name in ("v1", "v2", "v3"))
    assert [len(frame) for frame in (v1, v2, v3)] == [4, 4, 4]
    assert all(not frame["SK_ID_CURR"].duplicated().any() for frame in (v1, v2, v3))
    pd.testing.assert_frame_equal(v1, v2[v1.columns])
    pd.testing.assert_frame_equal(v2, v3[v2.columns])

    applicant = v1.set_index("SK_ID_CURR")
    assert applicant.loc[100001, "APP_CREDIT_INCOME_RATIO"] == pytest.approx(2.0)
    assert applicant.loc[100001, "APP_INCOME_PER_FAMILY_MEMBER"] == pytest.approx(50.0)
    assert applicant.loc[100001, "APP_EXT_SOURCE_MEAN"] == pytest.approx(0.15)
    assert applicant.loc[100002, "DAYS_EMPLOYED_SENTINEL"] == 1
    assert pd.isna(applicant.loc[100002, "DAYS_EMPLOYED"])
    assert applicant.loc[100001, "OWN_CAR_AGE_NOT_APPLICABLE"] == 1
    assert applicant.loc[100002, "OWN_CAR_AGE_MISSING"] == 1

    bureau = v2.set_index("SK_ID_CURR")
    assert bureau.loc[100001, "BUREAU_RECORD_COUNT"] == 2
    assert bureau.loc[100001, "BUREAU_ACTIVE_COUNT"] == 1
    assert bureau.loc[100001, "BUREAU_OVERDUE_LOAN_COUNT"] == 1
    assert bureau.loc[100001, "BUREAU_CREDIT_AMOUNT_SUM"] == pytest.approx(1500.0)
    assert bureau.loc[100001, "BUREAU_DEBT_SUM"] == pytest.approx(200.0)
    assert bureau.loc[100001, "BUREAU_DEBT_CREDIT_RATIO"] == pytest.approx(0.2)
    assert bureau.loc[100002, "BUREAU_NON_PRIMARY_CURRENCY_COUNT"] == 1
    assert bureau.loc[100002, "BUREAU_CREDIT_AMOUNT_OBSERVED_COUNT"] == 0
    assert pd.isna(bureau.loc[100002, "BUREAU_CREDIT_AMOUNT_SUM"])
    assert bureau.loc[100003, "BUREAU_HAS_HISTORY"] == 0
    assert bureau.loc[100003, "BUREAU_RECORD_COUNT"] == 0
    assert pd.isna(bureau.loc[100003, "BUREAU_ACTIVE_RATIO"])

    installments = v3.set_index("SK_ID_CURR")
    assert installments.loc[100001, "INST_SCHEDULE_COUNT"] == 2
    assert installments.loc[100001, "INST_PAYMENT_EVENT_COUNT"] == 3
    assert (
        installments.loc[100001, "INST_PAYMENT_AMOUNT_OBSERVED_SCHEDULE_COUNT"]
        == 1
    )
    assert installments.loc[100001, "INST_MISSING_PAYMENT_SCHEDULE_COUNT"] == 1
    assert installments.loc[100001, "INST_LATE_SCHEDULE_COUNT"] == 1
    assert installments.loc[100001, "INST_DAYS_LATE_MAX"] == pytest.approx(10.0)
    assert installments.loc[100001, "INST_PAID_AMOUNT_SUM"] == pytest.approx(100.0)
    assert installments.loc[100001, "INST_PAYMENT_RATIO"] == pytest.approx(1.0)
    assert installments.loc[100002, "INST_UNDERPAID_SCHEDULE_COUNT"] == 1
    assert installments.loc[100002, "INST_PAID_AMOUNT_SUM"] == pytest.approx(0.0)
    assert installments.loc[100002, "INST_PAYMENT_RATIO"] == pytest.approx(0.0)
    assert installments.loc[100003, "INST_PAYMENT_RATIO"] == pytest.approx(1.2)
    assert installments.loc[100004, "INST_HAS_HISTORY"] == 0
    assert installments.loc[100004, "INST_SCHEDULE_COUNT"] == 0
    assert pd.isna(installments.loc[100004, "INST_PAYMENT_RATIO"])

    pandas_schedules = _installments_frame().loc[
        _installments_frame()["SK_ID_CURR"] == 100001
    ].groupby(
        [
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
        ],
        dropna=False,
    )
    assert installments.loc[100001, "INST_SCHEDULE_COUNT"] == pandas_schedules.ngroups
    assert summary["source_quality"][
        "bureau_positive_days_credit_update_rows_excluded"
    ] == 1
    assert summary["outputs"]["v2"]["bureau_rows_aggregated"] == 3
    assert summary["outputs"]["v3"]["installment_payment_events_aggregated"] == 5
    assert summary["outputs"]["v3"]["installment_schedules_aggregated"] == 4
    assert summary["outputs"]["v1"]["policy_excluded_columns"] == ["CODE_GENDER"]
    assert summary["outputs"]["v1"]["model_feature_columns"] == (
        summary["outputs"]["v1"]["candidate_feature_columns"] - 1
    )
    assert summary["lineage_mismatch_rows"] == {"v1_to_v2": 0, "v2_to_v3": 0}
    assert all(summary["invariants"].values())
    assert json.loads(outputs["summary"].read_text(encoding="utf-8")) == summary


def test_invalid_split_fails_before_replacing_existing_outputs(tmp_path: Path) -> None:
    sources, outputs, _summary = _build(tmp_path)
    original_outputs = {
        name: outputs[name].read_bytes() for name in ("v1", "v2", "v3", "summary")
    }
    broken = _split_frame()
    broken.loc[0, "TARGET"] = 0
    broken.to_csv(sources["splits"], index=False)

    with pytest.raises(FeatureMartError, match="고객·TARGET"):
        build_feature_marts(
            sources["application"],
            sources["bureau"],
            sources["installments"],
            sources["splits"],
            v1_output=outputs["v1"],
            v2_output=outputs["v2"],
            v3_output=outputs["v3"],
            summary_output=outputs["summary"],
            temp_dir=outputs["temp"],
            memory_limit="512MB",
            threads=1,
        )

    assert {
        name: outputs[name].read_bytes() for name in ("v1", "v2", "v3", "summary")
    } == original_outputs


@pytest.mark.parametrize("column", ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
def test_future_installment_dates_are_rejected(tmp_path: Path, column: str) -> None:
    sources = _write_inputs(tmp_path)
    installments = _installments_frame()
    installments.loc[0, column] = 1
    installments.to_csv(sources["installments"], index=False)
    outputs = _outputs(tmp_path)

    with pytest.raises(FeatureMartError, match="현재 신청 이후"):
        build_feature_marts(
            sources["application"],
            sources["bureau"],
            sources["installments"],
            sources["splits"],
            v1_output=outputs["v1"],
            v2_output=outputs["v2"],
            v3_output=outputs["v3"],
            summary_output=outputs["summary"],
            temp_dir=outputs["temp"],
            memory_limit="512MB",
            threads=1,
        )

    assert not any(outputs[name].exists() for name in ("v1", "v2", "v3"))


def test_inconsistent_installment_schedule_is_rejected(tmp_path: Path) -> None:
    sources = _write_inputs(tmp_path)
    installments = _installments_frame()
    installments.loc[1, "AMT_INSTALMENT"] = 101.0
    installments.to_csv(sources["installments"], index=False)
    outputs = _outputs(tmp_path)

    with pytest.raises(FeatureMartError, match="서로 다른 예정일 또는 예정금액"):
        build_feature_marts(
            sources["application"],
            sources["bureau"],
            sources["installments"],
            sources["splits"],
            v1_output=outputs["v1"],
            v2_output=outputs["v2"],
            v3_output=outputs["v3"],
            summary_output=outputs["summary"],
            temp_dir=outputs["temp"],
            memory_limit="512MB",
            threads=1,
        )


def test_mismatched_payment_missingness_is_rejected(tmp_path: Path) -> None:
    sources = _write_inputs(tmp_path)
    installments = _installments_frame()
    installments.loc[0, "DAYS_ENTRY_PAYMENT"] = np.nan
    installments.to_csv(sources["installments"], index=False)
    outputs = _outputs(tmp_path)

    with pytest.raises(FeatureMartError, match="납부일·납부금액 결측 상태"):
        build_feature_marts(
            sources["application"],
            sources["bureau"],
            sources["installments"],
            sources["splits"],
            v1_output=outputs["v1"],
            v2_output=outputs["v2"],
            v3_output=outputs["v3"],
            summary_output=outputs["summary"],
            temp_dir=outputs["temp"],
            memory_limit="512MB",
            threads=1,
        )


def test_summary_has_no_customer_values_or_absolute_paths(tmp_path: Path) -> None:
    sources, outputs, summary = _build(tmp_path / "private-CfDJ8-area")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert str(tmp_path) not in serialized
    assert "private-CfDJ8-area" not in serialized
    assert "customer_ids" not in serialized
    assert summary["invariants"]["summary_excludes_customer_values"] is True
    assert all(path.is_file() for path in sources.values())
    assert outputs["summary"].is_file()


def test_output_cannot_overwrite_an_input(tmp_path: Path) -> None:
    sources = _write_inputs(tmp_path)
    outputs = _outputs(tmp_path)

    with pytest.raises(FeatureMartError, match="덮어쓸 수 없습니다"):
        build_feature_marts(
            sources["application"],
            sources["bureau"],
            sources["installments"],
            sources["splits"],
            v1_output=sources["application"],
            v2_output=outputs["v2"],
            v3_output=outputs["v3"],
            summary_output=outputs["summary"],
            temp_dir=outputs["temp"],
        )


def test_default_paths_match_stage3_contract() -> None:
    assert DEFAULT_APPLICATION_INPUT == Path("data/raw/application_train.csv")
    assert DEFAULT_BUREAU_INPUT == Path("data/raw/bureau.csv")
    assert DEFAULT_INSTALLMENTS_INPUT == Path("data/raw/installments_payments.csv")
    assert DEFAULT_SPLITS_INPUT == Path("data/interim/customer_splits.csv")
    assert DEFAULT_V1_OUTPUT == Path("data/processed/feature_mart_v1.parquet")
    assert DEFAULT_V2_OUTPUT == Path("data/processed/feature_mart_v2.parquet")
    assert DEFAULT_V3_OUTPUT == Path("data/processed/feature_mart_v3.parquet")
