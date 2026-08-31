"""Stage 2 train-only EDA의 데이터 누수 및 출력 안전성 회귀 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from creditlens.analysis.stage2_eda import (
    FIGURE_FILENAMES,
    Stage2EDAError,
    run_stage2_eda,
)


def _application() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [700_001, 700_002, 700_003, 800_001, 900_001],
            "TARGET": [0, 1, 0, 1, 0],
            "AMT_INCOME_TOTAL": [10.0, 20.0, 30.0, 99_999_991.0, 99_999_992.0],
            "AMT_CREDIT": [100.0, 200.0, 300.0, 88_888_881.0, 88_888_882.0],
            "DAYS_BIRTH": [-10_000, -12_000, -14_000, -1, -2],
            "DAYS_EMPLOYED": [-500, 365_243, -1_000, 777_777, 888_888],
            "EXT_SOURCE_2": [0.1, None, None, 0.99, 0.98],
            "CODE_GENDER": ["F", "XNA", "M", "XNA", "XNA"],
            "NAME_FAMILY_STATUS": ["Married", "Unknown", "Single", "Unknown", "Unknown"],
            "FLAG_OWN_CAR": ["N", "Y", "Y", "Y", "N"],
            "OWN_CAR_AGE": [None, None, 5.0, 888.0, 999.0],
            "FLAG_MOBIL": [1, 1, 1, 0, 0],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Cash loans",
                "/private/train-category",
                "/home/nontrain/validation-secret",
                "DO_NOT_LEAK_TEST",
            ],
        }
    )


def _assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [700_001, 700_002, 700_003, 800_001, 900_001],
            "TARGET": [0, 1, 0, 1, 0],
            "SPLIT": ["train", "train", "train", "validation", "test"],
        }
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    application_path = tmp_path / "application_train.csv"
    assignment_path = tmp_path / "customer_splits.csv"
    _application().to_csv(application_path, index=False)
    _assignments().to_csv(assignment_path, index=False)
    return application_path, assignment_path


def _run(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    application_path, assignment_path = _write_inputs(tmp_path)
    json_path = tmp_path / "reports" / "stage2_eda.json"
    report_path = tmp_path / "docs" / "Stage2_EDA_Report.md"
    figures_dir = tmp_path / "reports" / "figures"
    result = run_stage2_eda(
        application_path,
        assignment_path,
        json_path,
        report_path,
        figures_dir,
        chunk_size=2,
        correlation_sample_size=2,
        max_correlation_columns=5,
    )
    return result, json_path, report_path, figures_dir


def test_statistics_use_train_features_only(tmp_path: Path) -> None:
    result, _, _, _ = _run(tmp_path)

    scope = result["analysis_scope"]
    assert scope["split"] == "train"
    assert scope["train_rows"] == 3
    assert scope["non_train_feature_rows_used"] == 0
    income = result["numeric_statistics"]["columns"]["AMT_INCOME_TOTAL"]
    assert income["mean"] == pytest.approx(20.0)
    assert income["min"] == pytest.approx(10.0)
    assert income["max"] == pytest.approx(30.0)
    assert income["p99"] == pytest.approx(29.8)
    assert result["target_distribution"]["0"]["count"] == 2
    assert result["target_distribution"]["1"]["count"] == 1
    income_target = result["target_relationships"]["numeric"]["columns"][
        "AMT_INCOME_TOTAL"
    ]
    assert income_target["by_target"]["0"]["count"] == 2
    assert income_target["by_target"]["0"]["median"] == pytest.approx(20.0)
    assert income_target["by_target"]["1"]["count"] == 1
    sentinel = result["sentinel_statistics"]["DAYS_EMPLOYED_365243"]
    assert sentinel["count"] == 1
    assert sentinel["rate"] == pytest.approx(1 / 3)
    high_correlation = result["correlation"]
    assert len(high_correlation["absolute_pairs_at_least_0_95"]) == high_correlation[
        "absolute_correlation_at_least_0_95_pair_count"
    ]
    assert all(
        item["absolute_correlation"] >= 0.95
        for item in high_correlation["absolute_pairs_at_least_0_95"]
    )
    quality = result["data_quality_observations"]
    assert quality["known_category_values"]["CODE_GENDER_XNA"]["count"] == 1
    assert quality["known_category_values"]["NAME_FAMILY_STATUS_Unknown"]["count"] == 1
    assert quality["own_car_age_missingness"]["missing_count_among_car_owners"] == 1
    assert any(
        item["column"] == "EXT_SOURCE_2"
        for item in quality["high_missing_columns_at_least_40_percent"]
    )
    assert any(
        item["column"] == "FLAG_MOBIL"
        for item in quality["near_constant_columns_at_least_99_percent"]
    )


def test_non_train_feature_changes_do_not_change_aggregates(tmp_path: Path) -> None:
    result_before, _, _, _ = _run(tmp_path)
    application_path = tmp_path / "application_train.csv"
    application = pd.read_csv(application_path)
    application.loc[application["SK_ID_CURR"].eq(800_001), "AMT_INCOME_TOTAL"] = -123_456_789
    application.loc[application["SK_ID_CURR"].eq(900_001), "NAME_CONTRACT_TYPE"] = "CHANGED_TEST_SECRET"
    application.to_csv(application_path, index=False)

    result_after = run_stage2_eda(
        application_path,
        tmp_path / "customer_splits.csv",
        tmp_path / "reports-2" / "stage2_eda.json",
        tmp_path / "docs-2" / "Stage2_EDA_Report.md",
        tmp_path / "figures-2",
        chunk_size=2,
        correlation_sample_size=2,
        max_correlation_columns=5,
    )

    assert result_after["source"]["application_sha256"] != result_before["source"][
        "application_sha256"
    ]
    for section in (
        "analysis_scope",
        "target_distribution",
        "column_summary",
        "missingness",
        "numeric_statistics",
        "categorical_statistics",
        "correlation",
        "target_relationships",
        "sentinel_statistics",
        "data_quality_observations",
    ):
        assert result_after[section] == result_before[section]


def test_outputs_are_aggregate_safe_and_figures_are_created(tmp_path: Path) -> None:
    result, json_path, report_path, figures_dir = _run(tmp_path)

    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored == result
    combined_text = json_path.read_text(encoding="utf-8") + report_path.read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "700001",
        "700002",
        "700003",
        "800001",
        "900001",
        str(tmp_path),
        "/private/train-category",
        "/home/nontrain/validation-secret",
        "DO_NOT_LEAK_TEST",
    ):
        assert forbidden not in combined_text
    assert "[경로 값 숨김]" in combined_text
    assert "학습 데이터 EDA 보고서" in report_path.read_text(encoding="utf-8")
    for filename in FIGURE_FILENAMES:
        figure = figures_dir / filename
        assert figure.is_file()
        assert figure.stat().st_size > 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.assign(EXTRA="unsafe"), "세 개만"),
        (
            lambda frame: frame.assign(
                SPLIT=frame["SPLIT"].replace({"validation": "holdout"})
            ),
            "허용되지 않은",
        ),
        (
            lambda frame: frame.assign(
                TARGET=frame["TARGET"].mask(frame["SK_ID_CURR"].eq(700_001), 1)
            ),
            "TARGET 값이 일치",
        ),
    ],
)
def test_invalid_assignment_contract_fails_before_outputs(
    tmp_path: Path, change, message: str
) -> None:
    application_path, assignment_path = _write_inputs(tmp_path)
    changed = change(pd.read_csv(assignment_path))
    changed.to_csv(assignment_path, index=False)

    with pytest.raises(Stage2EDAError, match=message):
        run_stage2_eda(
            application_path,
            assignment_path,
            tmp_path / "result.json",
            tmp_path / "report.md",
            tmp_path / "figures",
        )
    assert not (tmp_path / "result.json").exists()


def test_identifier_columns_are_excluded_even_if_extra_id_is_present(tmp_path: Path) -> None:
    application_path, assignment_path = _write_inputs(tmp_path)
    application = pd.read_csv(application_path)
    application["SK_ID_SECONDARY"] = [710_001, 710_002, 710_003, 810_001, 910_001]
    application.to_csv(application_path, index=False)

    result = run_stage2_eda(
        application_path,
        assignment_path,
        tmp_path / "result.json",
        tmp_path / "report.md",
        tmp_path / "figures",
        chunk_size=2,
    )

    assert "SK_ID_CURR" not in result["numeric_statistics"]["columns"]
    assert "SK_ID_SECONDARY" not in result["numeric_statistics"]["columns"]


def test_output_path_collisions_fail_without_changing_inputs(tmp_path: Path) -> None:
    scenarios = (
        "json_application",
        "report_assignment",
        "json_report",
        "figure_application",
        "figure_assignment",
        "json_figure",
        "report_figure",
    )

    for scenario in scenarios:
        case_dir = tmp_path / scenario
        case_dir.mkdir()
        application_path, assignment_path = _write_inputs(case_dir)
        json_path = case_dir / "result.json"
        report_path = case_dir / "report.md"
        figures_dir = case_dir / "figures"

        if scenario == "json_application":
            json_path = application_path
        elif scenario == "report_assignment":
            report_path = assignment_path
        elif scenario == "json_report":
            report_path = json_path
        elif scenario == "figure_application":
            figures_dir.mkdir()
            moved = figures_dir / FIGURE_FILENAMES[0]
            application_path.replace(moved)
            application_path = moved
        elif scenario == "figure_assignment":
            figures_dir.mkdir()
            moved = figures_dir / FIGURE_FILENAMES[1]
            assignment_path.replace(moved)
            assignment_path = moved
        elif scenario == "json_figure":
            json_path = figures_dir / FIGURE_FILENAMES[0]
        elif scenario == "report_figure":
            report_path = figures_dir / FIGURE_FILENAMES[2]

        application_before = application_path.read_bytes()
        assignment_before = assignment_path.read_bytes()
        with pytest.raises(Stage2EDAError, match="경로 충돌"):
            run_stage2_eda(
                application_path,
                assignment_path,
                json_path,
                report_path,
                figures_dir,
                chunk_size=2,
            )

        assert application_path.read_bytes() == application_before, scenario
        assert assignment_path.read_bytes() == assignment_before, scenario
