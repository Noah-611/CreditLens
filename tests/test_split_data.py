"""Stage 2 고객 층화 분할 모듈의 회귀 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from creditlens.data.split_data import (
    DEFAULT_ASSIGNMENTS_OUTPUT,
    DEFAULT_INPUT,
    DEFAULT_SUMMARY_OUTPUT,
    DataSplitError,
    assignment_sha256,
    create_customer_splits,
    load_application_targets,
    main,
    split_application_data,
    validate_split_assignments,
)


def _customers(total: int = 200, positives: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(100_000, 100_000 + total),
            "TARGET": [1] * positives + [0] * (total - positives),
            "IGNORED_FEATURE": ["not-used"] * total,
        }
    )


def _write_application(
    path: Path, *, total: int = 200, positives: int = 40
) -> pd.DataFrame:
    frame = _customers(total=total, positives=positives)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _two_column(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["SK_ID_CURR", "TARGET"]].copy()


def test_end_to_end_split_is_stratified_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "private-area" / "application_train.csv"
    assignment_path = tmp_path / "interim" / "customer_splits.csv"
    summary_path = tmp_path / "reports" / "stage2_split_summary.json"
    original = _write_application(source).to_csv(index=False).encode()

    assignments, summary = split_application_data(
        source,
        assignments_output=assignment_path,
        summary_output=summary_path,
    )

    assert source.read_bytes() == original
    assert list(assignments.columns) == ["SK_ID_CURR", "TARGET", "SPLIT"]
    assert len(assignments) == 200
    assert not assignments["SK_ID_CURR"].duplicated().any()
    assert assignments.groupby("SPLIT").size().to_dict() == {
        "test": 30,
        "train": 140,
        "validation": 30,
    }
    assert (
        assignments.groupby(["SPLIT", "TARGET"]).size().unstack().to_dict()
        == {
            0: {"test": 24, "train": 112, "validation": 24},
            1: {"test": 6, "train": 28, "validation": 6},
        }
    )
    assert all(summary["invariants"].values())
    assert summary["assignment_sha256"] == assignment_sha256(assignments)
    assert hashlib.sha256(assignment_path.read_bytes()).hexdigest() == summary[
        "assignment_sha256"
    ]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_same_seed_is_reproducible_and_independent_of_input_order() -> None:
    customers = _two_column(_customers(total=300, positives=51))
    shuffled = customers.sample(frac=1, random_state=99)

    first = create_customer_splits(customers, seed=42)
    second = create_customer_splits(shuffled, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert assignment_sha256(first) == assignment_sha256(second)


def test_different_seed_changes_membership_but_not_counts() -> None:
    customers = _two_column(_customers(total=300, positives=51))

    first = create_customer_splits(customers, seed=42)
    second = create_customer_splits(customers, seed=43)

    assert not first["SPLIT"].equals(second["SPLIT"])
    pd.testing.assert_series_equal(
        first.groupby(["SPLIT", "TARGET"]).size(),
        second.groupby(["SPLIT", "TARGET"]).size(),
    )
    assert assignment_sha256(first) != assignment_sha256(second)


def test_target_rate_stays_close_for_imbalanced_data() -> None:
    customers = _two_column(_customers(total=1_000, positives=83))

    assignments = create_customer_splits(customers)

    overall = assignments["TARGET"].mean()
    rates = assignments.groupby("SPLIT")["TARGET"].mean()
    assert (rates - overall).abs().max() <= 1 / 150
    assert assignments.groupby("SPLIT").size().to_dict() == {
        "test": 150,
        "train": 700,
        "validation": 150,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.assign(
                SK_ID_CURR=[100_000, 100_000] + list(range(100_002, 100_200))
            ),
            "중복",
        ),
        (
            lambda frame: frame.assign(
                TARGET=frame["TARGET"].mask(frame.index == 0)
            ),
            "TARGET 결측값",
        ),
        (
            lambda frame: frame.assign(
                TARGET=frame["TARGET"].mask(frame.index == 0, 2)
            ),
            "0과 1만",
        ),
        (
            lambda frame: frame.assign(
                TARGET=frame["TARGET"].astype("object").mask(frame.index == 0, "bad")
            ),
            "0과 1만",
        ),
    ],
)
def test_source_contract_rejects_duplicate_missing_and_nonbinary_target(
    tmp_path: Path, mutate: object, message: str
) -> None:
    source = tmp_path / "application_train.csv"
    frame = _customers()
    broken = mutate(frame)  # type: ignore[operator]
    broken.to_csv(source, index=False)

    with pytest.raises(DataSplitError, match=message):
        load_application_targets(source)


def test_missing_customer_id_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "application_train.csv"
    frame = _customers()
    frame.loc[0, "SK_ID_CURR"] = None
    frame.to_csv(source, index=False)

    with pytest.raises(DataSplitError, match="SK_ID_CURR 결측값"):
        load_application_targets(source)


def test_assignment_validator_rejects_overlap_and_lost_customer() -> None:
    customers = _two_column(_customers())
    assignments = create_customer_splits(customers)
    duplicated = assignments.copy()
    duplicated.loc[1, "SK_ID_CURR"] = duplicated.loc[0, "SK_ID_CURR"]

    with pytest.raises(DataSplitError, match="중복 고객"):
        validate_split_assignments(customers, duplicated)

    with pytest.raises(DataSplitError, match="고객 전체"):
        validate_split_assignments(customers, assignments.iloc[:-1].copy())


def test_assignment_validator_rejects_target_change_and_lost_stratification() -> None:
    customers = _two_column(_customers())
    assignments = create_customer_splits(customers)
    changed_target = assignments.copy()
    changed_target.loc[0, "TARGET"] = 1 - changed_target.loc[0, "TARGET"]

    with pytest.raises(DataSplitError, match="TARGET 값"):
        validate_split_assignments(customers, changed_target)

    changed_split = assignments.copy()
    train_index = changed_split.index[changed_split["SPLIT"] == "train"][0]
    changed_split.loc[train_index, "SPLIT"] = "validation"
    with pytest.raises(DataSplitError, match="층화 건수"):
        validate_split_assignments(customers, changed_split)


def test_summary_has_required_metadata_without_ids_or_absolute_paths(
    tmp_path: Path,
) -> None:
    secret_parent = tmp_path / "token-CfDJ8-secret"
    source = secret_parent / "application_train.csv"
    _write_application(source)

    _assignments, summary = split_application_data(
        source, assignments_output=None, summary_output=None
    )
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["schema_version"] == "1.0"
    assert summary["generated_at_utc"].endswith("Z")
    assert set(summary["environment"]) == {"python", "numpy", "pandas"}
    assert summary["source"]["display_path"] == "application_train.csv"
    assert summary["strategy"]["seed"] == 42
    assert summary["strategy"]["ratios"] == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }
    assert "token-CfDJ8-secret" not in serialized
    assert str(tmp_path) not in serialized
    assert "100000" not in serialized
    assert "SK_ID_CURR" not in serialized


def test_output_paths_cannot_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "application_train.csv"
    _write_application(source)
    original = source.read_bytes()

    with pytest.raises(DataSplitError, match="원본 application CSV"):
        split_application_data(
            source,
            assignments_output=source,
            summary_output=tmp_path / "summary.json",
        )
    assert source.read_bytes() == original


def test_cli_writes_requested_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "application_train.csv"
    assignment_path = tmp_path / "customer_splits.csv"
    summary_path = tmp_path / "summary.json"
    _write_application(source)

    result = main(
        [
            "--input",
            str(source),
            "--assignments-output",
            str(assignment_path),
            "--summary-output",
            str(summary_path),
            "--seed",
            "42",
        ]
    )

    assert result == 0
    assert assignment_path.is_file()
    assert summary_path.is_file()
    assert "Stage 2 분할 완료" in capsys.readouterr().out


def test_cli_defaults_match_stage2_local_paths() -> None:
    assert DEFAULT_INPUT == Path("data/raw/application_train.csv")
    assert DEFAULT_ASSIGNMENTS_OUTPUT == Path("data/interim/customer_splits.csv")
    assert DEFAULT_SUMMARY_OUTPUT == Path("reports/stage2_split_summary.json")
