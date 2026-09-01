"""Stage 4 모델 데이터 로더의 봉인·피처 역할 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import creditlens.modeling.data as modeling_data
from creditlens.modeling.data import (
    ModelingDataError,
    TestSetSealedError as SealedTestError,
    _load_split,
    load_development_data,
    load_model_split,
)
from creditlens.modeling.feature_roles import (
    VERSION_CONTRACTS,
    resolve_feature_roles,
    schema_sha256,
)


def _mart_frame(version: str, *, test_value: float = 999.0) -> pd.DataFrame:
    contract = VERSION_CONTRACTS[version]  # type: ignore[index]
    rows = 4
    frame: dict[str, object] = {
        "SK_ID_CURR": [1, 2, 3, 4],
        "TARGET": pd.Series([0, 1, 0, 1], dtype="int8"),
        "SPLIT": ["train", "train", "validation", "test"],
        "CODE_GENDER": ["F", "M", "F", "M"],
    }
    for index in range(contract.numeric_features):
        values = [float(index), float(index + 1), float(index + 2), test_value]
        frame[f"NUM_{index:03d}"] = values
    for index in range(contract.categorical_features):
        frame[f"CAT_{index:03d}"] = ["a", "b", "a", f"test-{test_value}"]
    result = pd.DataFrame(frame)
    assert len(result.columns) == contract.total_columns
    assert len(result) == rows
    return result


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=":memory:") as connection:
        connection.register("mart", frame)
        connection.execute("COPY mart TO ? (FORMAT PARQUET)", [str(path)])


def _write_summary(path: Path, mart_path: Path, version: str) -> None:
    with duckdb.connect(database=":memory:") as connection:
        schema_rows = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(mart_path)]
        ).fetchall()
        split_rows = connection.execute(
            'SELECT "SPLIT", COUNT(*) FROM read_parquet(?) GROUP BY "SPLIT"',
            [str(mart_path)],
        ).fetchall()
        row_count = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(mart_path)]
        ).fetchone()[0]
    digest = hashlib.sha256(mart_path.read_bytes()).hexdigest()
    contract = VERSION_CONTRACTS[version]  # type: ignore[index]
    payload = {
        "outputs": {
            version: {
                "rows": row_count,
                "columns": contract.total_columns,
                "model_feature_columns": contract.model_features,
                "schema_sha256": schema_sha256(
                    [(str(row[0]), str(row[1])) for row in schema_rows]
                ),
                "split_counts": {
                    str(split): {"rows": int(count)} for split, count in split_rows
                },
                "sha256": digest,
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("version", "total", "model", "numeric", "categorical"),
    [
        ("v1", 136, 132, 118, 14),
        ("v2", 173, 169, 155, 14),
        ("v3", 202, 198, 184, 14),
    ],
)
def test_all_versions_enforce_roles_and_exclusions(
    tmp_path: Path,
    version: str,
    total: int,
    model: int,
    numeric: int,
    categorical: int,
) -> None:
    path = tmp_path / f"feature_mart_{version}.parquet"
    _write_parquet(path, _mart_frame(version))

    dataset = load_development_data(version, path)  # type: ignore[arg-type]

    assert total == VERSION_CONTRACTS[version].total_columns  # type: ignore[index]
    assert dataset.train.X.shape == (2, model)
    assert dataset.validation.X.shape == (1, model)
    assert len(dataset.roles.numeric) == numeric
    assert len(dataset.roles.categorical) == categorical
    assert not {"SK_ID_CURR", "TARGET", "SPLIT", "CODE_GENDER"}.intersection(
        dataset.train.X.columns
    )
    assert not any(column.startswith("SK_ID_") for column in dataset.train.X)
    assert not hasattr(dataset, "test")
    assert dataset.audit["feature_rows_loaded"]["test"] == 0
    assert dataset.audit["test_feature_rows_used"] == 0


def test_changing_test_features_cannot_change_loaded_development_data(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    _write_parquet(first_path, _mart_frame("v1", test_value=111.0))
    _write_parquet(second_path, _mart_frame("v1", test_value=999999.0))

    first = load_development_data("v1", first_path)
    second = load_development_data("v1", second_path)

    pd.testing.assert_frame_equal(first.train.X, second.train.X)
    pd.testing.assert_series_equal(first.train.y, second.train.y)
    pd.testing.assert_frame_equal(first.validation.X, second.validation.X)
    pd.testing.assert_series_equal(first.validation.y, second.validation.y)


def test_public_and_internal_split_loaders_reject_test(tmp_path: Path) -> None:
    path = tmp_path / "mart.parquet"
    _write_parquet(path, _mart_frame("v1"))

    with pytest.raises(SealedTestError):
        load_model_split(path, "v1", "test")

    with duckdb.connect(database=":memory:") as connection:
        schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
        roles = resolve_feature_roles("v1", [(row[0], row[1]) for row in schema])
        with pytest.raises(SealedTestError):
            _load_split(connection, path, "test", roles)


def test_invalid_schema_and_data_contracts_fail(tmp_path: Path) -> None:
    invalid_schema = tmp_path / "invalid_schema.parquet"
    frame = _mart_frame("v1").drop(columns="NUM_000")
    _write_parquet(invalid_schema, frame)
    with pytest.raises(ModelingDataError, match="전체 컬럼 수"):
        load_development_data("v1", invalid_schema)

    invalid_data = tmp_path / "invalid_data.parquet"
    frame = _mart_frame("v1")
    frame.loc[1, "SK_ID_CURR"] = 1
    _write_parquet(invalid_data, frame)
    with pytest.raises(ModelingDataError, match="중복 고객 ID"):
        load_development_data("v1", invalid_data)


def test_invalid_target_split_and_infinite_numeric_fail(tmp_path: Path) -> None:
    target_path = tmp_path / "target.parquet"
    target = _mart_frame("v1")
    target.loc[0, "TARGET"] = 2
    _write_parquet(target_path, target)
    with pytest.raises(ModelingDataError, match="TARGET"):
        load_development_data("v1", target_path)

    split_path = tmp_path / "split.parquet"
    split = _mart_frame("v1")
    split.loc[0, "SPLIT"] = "unknown"
    _write_parquet(split_path, split)
    with pytest.raises(ModelingDataError, match="SPLIT"):
        load_development_data("v1", split_path)

    infinite_path = tmp_path / "infinite.parquet"
    infinite = _mart_frame("v1")
    infinite.loc[0, "NUM_000"] = float("inf")
    _write_parquet(infinite_path, infinite)
    with pytest.raises(ModelingDataError, match="무한대"):
        load_development_data("v1", infinite_path)


def test_summary_rejects_column_rename_and_row_loss(tmp_path: Path) -> None:
    original_path = tmp_path / "original.parquet"
    original = _mart_frame("v1")
    _write_parquet(original_path, original)
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, original_path, "v1")

    verified = load_development_data(
        "v1",
        original_path,
        summary_path=summary_path,
        verify_file_sha256=True,
    )
    assert verified.audit["stage3_summary_verified"] is True
    assert verified.audit["parquet_sha256_verified"] is True
    assert verified.parquet_sha256 is not None

    renamed_path = tmp_path / "renamed.parquet"
    renamed = original.rename(columns={"NUM_000": "REPLACED_NUM"})
    _write_parquet(renamed_path, renamed)
    with pytest.raises(ModelingDataError, match="schema_sha256"):
        load_development_data("v1", renamed_path, summary_path=summary_path)

    shortened_path = tmp_path / "shortened.parquet"
    _write_parquet(shortened_path, original.drop(index=0))
    with pytest.raises(ModelingDataError, match=r"\.rows"):
        load_development_data("v1", shortened_path, summary_path=summary_path)


def test_default_production_path_enforces_summary_and_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mart_path = tmp_path / "feature_mart_v1.parquet"
    _write_parquet(mart_path, _mart_frame("v1"))
    summary_path = tmp_path / "stage3_build_summary.json"
    _write_summary(summary_path, mart_path, "v1")
    monkeypatch.setitem(modeling_data.DEFAULT_MART_PATHS, "v1", mart_path)
    monkeypatch.setattr(modeling_data, "DEFAULT_BUILD_SUMMARY_PATH", summary_path)

    dataset = load_development_data("v1")

    assert dataset.audit["stage3_summary_verified"] is True
    assert dataset.audit["parquet_sha256_verified"] is True
    assert dataset.parquet_sha256 is not None
    split = load_model_split(mart_path, "v1", "train")
    assert len(split.X) == 2

    _write_parquet(mart_path, _mart_frame("v1", test_value=123456.0))
    with pytest.raises(ModelingDataError, match="SHA256"):
        load_development_data("v1")
    with pytest.raises(ModelingDataError, match="SHA256"):
        load_model_split(mart_path, "v1", "train")
