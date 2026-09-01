"""Stage 3 파생 피처 프로파일의 train-only 경계 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from creditlens.analysis.stage3_feature_profile import (
    PROFILE_FEATURES,
    Stage3FeatureProfileError,
    profile_stage3_features,
)


def _mart_frame() -> pd.DataFrame:
    data: dict[str, list[float | int | str | None]] = {
        "SK_ID_CURR": [700001, 700002, 700003, 800001, 900001],
        "TARGET": [0, 1, 0, 1, 0],
        "SPLIT": ["train", "train", "train", "validation", "test"],
    }
    data.update({feature: [0.0] * 5 for feature in PROFILE_FEATURES})
    data["APP_CREDIT_INCOME_RATIO"] = [1.0, 2.0, 3.0, 99_999_991.0, 99_999_992.0]
    data["BUREAU_ACTIVE_RATIO"] = [0.1, None, 0.3, 0.99, 0.98]
    data["BUREAU_RECORD_COUNT"] = [1, 2, 20, 999_991, 999_992]
    data["INST_PAYMENT_RATIO"] = [0.8, 1.0, 1.2, 999.0, 888.0]
    return pd.DataFrame(data)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=":memory:") as connection:
        connection.register("mart", frame)
        connection.execute(
            "COPY mart TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)]
        )


def _run(tmp_path: Path, frame: pd.DataFrame | None = None):
    input_path = tmp_path / "processed" / "feature_mart_v3.parquet"
    output_path = tmp_path / "reports" / "stage3_feature_profile.json"
    _write_parquet(input_path, _mart_frame() if frame is None else frame)
    result = profile_stage3_features(input_path, output_path)
    return result, input_path, output_path


def test_profile_uses_train_features_only(tmp_path: Path) -> None:
    result, _input_path, output_path = _run(tmp_path)

    scope = result["analysis_scope"]
    assert scope["statistics_split"] == "train"
    assert scope["train_feature_rows_used"] == 3
    assert scope["validation_feature_rows_used"] == 0
    assert scope["test_feature_rows_used"] == 0
    assert scope["profiled_feature_count"] == len(PROFILE_FEATURES)
    credit_ratio = result["feature_statistics"]["APP_CREDIT_INCOME_RATIO"]
    assert credit_ratio["mean"] == pytest.approx(2.0)
    assert credit_ratio["min"] == pytest.approx(1.0)
    assert credit_ratio["max"] == pytest.approx(3.0)
    bureau_ratio = result["feature_statistics"]["BUREAU_ACTIVE_RATIO"]
    assert bureau_ratio["missing_count"] == 1
    assert bureau_ratio["missing_rate"] == pytest.approx(1 / 3)
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


def test_non_train_feature_changes_do_not_change_profile_statistics(
    tmp_path: Path,
) -> None:
    before, _input_path, _output_path = _run(tmp_path / "before")
    changed = _mart_frame()
    changed.loc[changed["SPLIT"].eq("validation"), "APP_CREDIT_INCOME_RATIO"] = -1e12
    changed.loc[changed["SPLIT"].eq("test"), "BUREAU_ACTIVE_RATIO"] = -1e9
    after, _input_path, _output_path = _run(tmp_path / "after", changed)

    assert before["source"]["sha256"] != after["source"]["sha256"]
    for section in (
        "analysis_scope",
        "feature_statistics",
        "rankings",
        "processing_decisions",
        "invariants",
    ):
        assert before[section] == after[section]


def test_profile_output_excludes_customer_values_and_absolute_paths(
    tmp_path: Path,
) -> None:
    result, _input_path, output_path = _run(tmp_path / "private-profile-area")
    serialized = output_path.read_text(encoding="utf-8")

    for forbidden in ("700001", "700002", "700003", "800001", "900001", str(tmp_path)):
        assert forbidden not in serialized
    assert result["invariants"]["identifier_not_loaded_with_features"] is True
    assert result["invariants"]["non_train_feature_rows_used"] == 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.drop(columns=["INST_PAYMENT_RATIO"]),
            "필수 컬럼",
        ),
        (
            lambda frame: frame.assign(
                SK_ID_CURR=[700001, 700001, 700003, 800001, 900001]
            ),
            "고객 키",
        ),
    ],
)
def test_invalid_mart_contract_fails_without_output(
    tmp_path: Path, mutator, message: str
) -> None:
    input_path = tmp_path / "feature_mart_v3.parquet"
    output_path = tmp_path / "profile.json"
    _write_parquet(input_path, mutator(_mart_frame()))

    with pytest.raises(Stage3FeatureProfileError, match=message):
        profile_stage3_features(input_path, output_path)
    assert not output_path.exists()
