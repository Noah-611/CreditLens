"""Stage 4 train-fit 전처리와 누수 방지 계약 테스트."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from creditlens.modeling.preprocessing import (
    MISSING_CATEGORY,
    RARE_CATEGORY,
    PreprocessingContractError,
    make_preprocessor,
    transformed_feature_names,
)


@dataclass(frozen=True)
class _Roles:
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


def _roles() -> _Roles:
    numeric = (
        "AMT_INCOME_TOTAL",
        "OWN_CAR_AGE",
        "OWN_CAR_AGE_NOT_APPLICABLE",
        "OWN_CAR_AGE_MISSING",
        "BUREAU_HAS_HISTORY",
        "BUREAU_DEBT_SUM",
        "ALL_NULL_NUMERIC",
    )
    categorical = ("NAME_CONTRACT_TYPE",)
    return _Roles(numeric + categorical, numeric, categorical)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [1.0, np.nan, 3.0, 5.0],
            "OWN_CAR_AGE": [np.nan, 4.0, 8.0, 12.0],
            "OWN_CAR_AGE_NOT_APPLICABLE": [1, 0, 0, 0],
            "OWN_CAR_AGE_MISSING": [0, 0, 0, 0],
            "BUREAU_HAS_HISTORY": [0, 1, 1, 1],
            "BUREAU_DEBT_SUM": [np.nan, 0.0, 100.0, 200.0],
            "ALL_NULL_NUMERIC": [np.nan, np.nan, np.nan, np.nan],
            "NAME_CONTRACT_TYPE": ["Cash", "Unknown", "XNA", None],
        }
    )


def _validation(category: str = "NEW_LEVEL") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [np.nan, 1e12],
            "OWN_CAR_AGE": [np.nan, np.nan],
            "OWN_CAR_AGE_NOT_APPLICABLE": [1, 0],
            "OWN_CAR_AGE_MISSING": [0, 1],
            "BUREAU_HAS_HISTORY": [0, 1],
            "BUREAU_DEBT_SUM": [0.0, np.nan],
            "ALL_NULL_NUMERIC": [np.nan, 999.0],
            "NAME_CONTRACT_TYPE": [category, "Cash"],
        }
    )


def _index(names: tuple[str, ...], suffix: str) -> int:
    matches = [index for index, name in enumerate(names) if name.endswith(suffix)]
    assert len(matches) == 1, (suffix, matches)
    return matches[0]


def test_tree_preprocessor_preserves_zero_missing_and_structural_absence() -> None:
    preprocessor = make_preprocessor(_roles(), model_family="tree")
    matrix = preprocessor.fit_transform(_train())
    names = transformed_feature_names(preprocessor)
    dense = matrix.toarray()

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.dtype == np.float32
    debt = _index(names, "numeric__BUREAU_DEBT_SUM")
    debt_missing = _index(names, "missingindicator_BUREAU_DEBT_SUM")
    assert dense[0, debt] == pytest.approx(100.0)
    assert dense[0, debt_missing] == 1
    assert dense[1, debt] == 0
    assert dense[1, debt_missing] == 0

    car_age = _index(names, "numeric__OWN_CAR_AGE")
    assert dense[0, car_age] == 0
    assert preprocessor.named_steps["semantic_values"].car_age_fill_value_ == 8
    validation = preprocessor.transform(_validation()).toarray()
    assert validation[0, car_age] == 0
    assert validation[1, car_age] == 8
    all_null = _index(names, "numeric__ALL_NULL_NUMERIC")
    all_null_missing = _index(names, "missingindicator_ALL_NULL_NUMERIC")
    assert np.all(dense[:, all_null] == 0)
    assert np.all(dense[:, all_null_missing] == 1)


def test_categorical_missing_observed_labels_and_unknown_are_distinct() -> None:
    preprocessor = make_preprocessor(_roles(), model_family="tree")
    train_matrix = preprocessor.fit_transform(_train()).toarray()
    names = transformed_feature_names(preprocessor)

    missing = _index(names, f"NAME_CONTRACT_TYPE_{MISSING_CATEGORY}")
    unknown_label = _index(names, "NAME_CONTRACT_TYPE_Unknown")
    xna = _index(names, "NAME_CONTRACT_TYPE_XNA")
    assert train_matrix[3, missing] == 1
    assert train_matrix[1, unknown_label] == 1
    assert train_matrix[2, xna] == 1

    validation = preprocessor.transform(_validation()).toarray()
    categorical_indices = [
        index for index, name in enumerate(names) if name.startswith("categorical__")
    ]
    assert validation[0, categorical_indices].sum() == 0
    assert validation.shape[1] == train_matrix.shape[1]


def test_train_statistics_do_not_change_when_validation_changes() -> None:
    preprocessor = make_preprocessor(_roles(), model_family="linear")
    train_matrix = preprocessor.fit_transform(_train()).copy()
    numeric = preprocessor.named_steps["columns"].named_transformers_["numeric"]
    numeric_values = dict(numeric.transformer_list)["values"]
    statistics_before = numeric_values.named_steps["impute"].statistics_.copy()
    feature_names_before = transformed_feature_names(preprocessor)

    first = preprocessor.transform(_validation("FIRST_SECRET"))
    second_frame = _validation("SECOND_SECRET")
    second_frame.loc[:, "AMT_INCOME_TOTAL"] = [-1e15, 1e15]
    second = preprocessor.transform(second_frame)

    np.testing.assert_array_equal(
        numeric_values.named_steps["impute"].statistics_, statistics_before
    )
    assert transformed_feature_names(preprocessor) == feature_names_before
    assert first.shape == second.shape
    assert (preprocessor.transform(_train()) != train_matrix).nnz == 0


def test_linear_scaling_keeps_missing_indicators_binary() -> None:
    preprocessor = make_preprocessor(_roles(), model_family="linear")
    matrix = preprocessor.fit_transform(_train()).toarray()
    names = transformed_feature_names(preprocessor)

    debt_missing = _index(names, "missingindicator_BUREAU_DEBT_SUM")
    income_missing = _index(names, "missingindicator_AMT_INCOME_TOTAL")
    assert set(matrix[:, debt_missing]) == {0.0, 1.0}
    assert set(matrix[:, income_missing]) == {0.0, 1.0}


def test_linear_scaling_controls_rare_extreme_numeric_values() -> None:
    train = _train()
    train.loc[:, "BUREAU_DEBT_SUM"] = [0.0, 0.0, 0.0, 100_000_000.0]
    preprocessor = make_preprocessor(_roles(), model_family="linear")

    matrix = preprocessor.fit_transform(train).toarray()
    names = transformed_feature_names(preprocessor)
    debt = _index(names, "numeric__BUREAU_DEBT_SUM")

    assert np.max(np.abs(matrix[:, debt])) < 3.0


def test_known_rare_categories_are_grouped_but_unseen_is_ignored() -> None:
    roles = _roles()
    train = pd.concat([_train(), _train().iloc[[0]]], ignore_index=True)
    train.loc[:, "NAME_CONTRACT_TYPE"] = ["A", "A", "A", "B", "C"]
    preprocessor = make_preprocessor(
        roles, model_family="tree", rare_min_frequency=0.3
    )
    transformed = preprocessor.fit_transform(train).toarray()
    names = transformed_feature_names(preprocessor)
    rare = _index(names, f"NAME_CONTRACT_TYPE_{RARE_CATEGORY}")
    assert np.all(transformed[[3, 4], rare] == 1)

    validation = _validation("UNSEEN")
    validation_matrix = preprocessor.transform(validation).toarray()
    categorical_indices = [
        index for index, name in enumerate(names) if name.startswith("categorical__")
    ]
    assert validation_matrix[0, categorical_indices].sum() == 0


def test_output_is_finite_sparse_float32_and_reproducible() -> None:
    first = make_preprocessor(_roles(), model_family="linear")
    second = make_preprocessor(_roles(), model_family="linear")
    first_matrix = first.fit_transform(_train())
    second_matrix = second.fit_transform(_train())

    assert sparse.isspmatrix_csr(first_matrix)
    assert first_matrix.dtype == np.float32
    assert np.isfinite(first_matrix.data).all()
    assert transformed_feature_names(first) == transformed_feature_names(second)
    assert (first_matrix != second_matrix).nnz == 0


def test_invalid_numeric_or_column_contract_is_rejected() -> None:
    preprocessor = make_preprocessor(_roles(), model_family="tree")
    invalid = _train()
    invalid.loc[0, "AMT_INCOME_TOTAL"] = np.inf
    with pytest.raises(PreprocessingContractError, match="무한값"):
        preprocessor.fit(invalid)

    wrong_columns = _train().drop(columns=["BUREAU_DEBT_SUM"])
    with pytest.raises(PreprocessingContractError, match="컬럼과 순서"):
        preprocessor.fit(wrong_columns)


def test_invalid_model_family_is_rejected() -> None:
    with pytest.raises(PreprocessingContractError, match="linear 또는 tree"):
        make_preprocessor(_roles(), model_family="neural")  # type: ignore[arg-type]


@pytest.mark.parametrize("reserved", [MISSING_CATEGORY, RARE_CATEGORY])
def test_reserved_category_tokens_are_rejected_during_transform(
    reserved: str,
) -> None:
    preprocessor = make_preprocessor(_roles(), model_family="tree")
    preprocessor.fit(_train())
    validation = _validation()
    validation.loc[0, "NAME_CONTRACT_TYPE"] = reserved

    with pytest.raises(PreprocessingContractError, match="예약 토큰"):
        preprocessor.transform(validation)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("OWN_CAR_AGE_NOT_APPLICABLE", 2, "결측 없는 0/1"),
        ("OWN_CAR_AGE_MISSING", np.nan, "결측 없는 0/1"),
        ("OWN_CAR_AGE_MISSING", 1, "모순"),
    ],
)
def test_invalid_car_age_semantics_are_rejected(
    column: str,
    value: float,
    message: str,
) -> None:
    invalid = _train()
    invalid.loc[1, column] = value

    with pytest.raises(PreprocessingContractError, match=message):
        make_preprocessor(_roles(), model_family="tree").fit(invalid)
