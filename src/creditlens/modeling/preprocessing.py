"""Stage 4 모델용 train-fit 전처리 파이프라인.

모든 학습 통계는 이 파이프라인의 ``fit`` 입력에서만 계산된다. 호출자는
반드시 train 데이터로만 fit하고 validation에는 transform만 적용해야 한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


MISSING_CATEGORY = "__MISSING__"
RARE_CATEGORY = "__RARE__"
DEFAULT_RARE_MIN_FREQUENCY = 0.001
ModelFamily = Literal["linear", "tree"]


class FeatureRolesLike(Protocol):
    """전처리기가 요구하는 피처 역할의 최소 인터페이스."""

    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


class PreprocessingContractError(ValueError):
    """모델 입력 또는 전처리 설정이 계약을 위반할 때 발생한다."""


def _feature_names(
    input_features: Sequence[str] | None,
    fallback: Sequence[str] | None,
) -> np.ndarray:
    if input_features is not None:
        return np.asarray(input_features, dtype=object)
    if fallback is None:
        raise PreprocessingContractError("입력 피처 이름을 확인할 수 없습니다.")
    return np.asarray(fallback, dtype=object)


class FeatureContractValidator(BaseEstimator, TransformerMixin):
    """컬럼 순서·역할과 수치형 유한값을 fit/transform 양쪽에서 검사한다."""

    def __init__(
        self,
        *,
        feature_columns: tuple[str, ...],
        numeric_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
    ) -> None:
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

    def _validate(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise PreprocessingContractError("모델 입력 X는 pandas DataFrame이어야 합니다.")
        observed = tuple(str(column) for column in X.columns)
        if observed != self.feature_columns:
            raise PreprocessingContractError(
                "모델 입력 컬럼과 순서가 피처 역할 계약과 다릅니다."
            )
        if set(self.numeric_columns).intersection(self.categorical_columns):
            raise PreprocessingContractError("수치형과 범주형 피처 역할이 겹칩니다.")
        if set(self.numeric_columns).union(self.categorical_columns) != set(
            self.feature_columns
        ):
            raise PreprocessingContractError("모든 모델 피처에 하나의 역할이 필요합니다.")

        if self.numeric_columns:
            numeric = X.loc[:, self.numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            invalid_coercion = numeric.isna() & X.loc[
                :, self.numeric_columns
            ].notna()
            if bool(invalid_coercion.to_numpy().any()):
                raise PreprocessingContractError("수치형 피처에 숫자가 아닌 값이 있습니다.")
            values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
            if bool(np.isinf(values).any()):
                raise PreprocessingContractError("수치형 피처에 무한값이 있습니다.")
        return X

    def fit(self, X: Any, y: Any = None) -> FeatureContractValidator:
        frame = self._validate(X)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        return self._validate(X)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        return _feature_names(input_features, getattr(self, "feature_names_in_", None))


class SemanticValueTransformer(BaseEstimator, TransformerMixin):
    """구조적 해당 없음과 차량 보유자의 결측을 서로 다르게 처리한다."""

    def __init__(
        self,
        *,
        car_age_column: str = "OWN_CAR_AGE",
        car_not_applicable_column: str = "OWN_CAR_AGE_NOT_APPLICABLE",
        car_missing_column: str = "OWN_CAR_AGE_MISSING",
    ) -> None:
        self.car_age_column = car_age_column
        self.car_not_applicable_column = car_not_applicable_column
        self.car_missing_column = car_missing_column

    def _validated_car_values(
        self,
        X: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, pd.Series] | None:
        columns = set(X.columns)
        required = {
            self.car_age_column,
            self.car_not_applicable_column,
            self.car_missing_column,
        }
        if not columns.intersection(required):
            return None
        if not required.issubset(columns):
            raise PreprocessingContractError(
                "OWN_CAR_AGE 의미 처리를 위한 연식과 두 플래그가 모두 필요합니다."
            )

        ages = pd.to_numeric(X[self.car_age_column], errors="coerce")
        not_applicable_raw = pd.to_numeric(
            X[self.car_not_applicable_column], errors="coerce"
        )
        missing_raw = pd.to_numeric(X[self.car_missing_column], errors="coerce")
        for name, values in (
            (self.car_not_applicable_column, not_applicable_raw),
            (self.car_missing_column, missing_raw),
        ):
            if values.isna().any() or not values.isin((0, 1)).all():
                raise PreprocessingContractError(f"{name}은 결측 없는 0/1이어야 합니다.")

        not_applicable = not_applicable_raw.eq(1)
        owner_missing = missing_raw.eq(1)
        invalid = (
            (not_applicable & owner_missing)
            | (not_applicable & ages.notna())
            | (owner_missing & ages.notna())
            | (ages.isna() & ~not_applicable & ~owner_missing)
        )
        if invalid.any():
            raise PreprocessingContractError(
                "OWN_CAR_AGE와 구조적 해당 없음·일반 결측 플래그가 모순됩니다."
            )
        return ages, not_applicable, owner_missing

    def fit(self, X: Any, y: Any = None) -> SemanticValueTransformer:
        if not isinstance(X, pd.DataFrame):
            raise PreprocessingContractError("의미 변환 입력은 DataFrame이어야 합니다.")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.car_age_fill_value_ = 0.0
        car_values = self._validated_car_values(X)
        if car_values is not None:
            ages, not_applicable, owner_missing = car_values
            valid_owner_ages = ages.loc[
                ~not_applicable & ~owner_missing & ages.notna()
            ]
            if not valid_owner_ages.empty:
                self.car_age_fill_value_ = float(valid_owner_ages.median())
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise PreprocessingContractError("의미 변환 입력은 DataFrame이어야 합니다.")
        transformed = X.copy()
        car_values = self._validated_car_values(transformed)
        if car_values is not None:
            _, not_applicable, owner_missing = car_values
            transformed.loc[not_applicable, self.car_age_column] = 0.0
            transformed.loc[owner_missing, self.car_age_column] = (
                self.car_age_fill_value_
            )
        return transformed

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        return _feature_names(input_features, getattr(self, "feature_names_in_", None))


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """train에서 관측된 희소 범주만 ``__RARE__``로 묶는다.

    transform에서 처음 보는 범주는 그대로 OneHotEncoder로 전달되어
    ``handle_unknown='ignore'`` 규칙에 따라 모든 0으로 인코딩된다.
    """

    def __init__(
        self,
        *,
        min_frequency: float = DEFAULT_RARE_MIN_FREQUENCY,
        missing_token: str = MISSING_CATEGORY,
        rare_token: str = RARE_CATEGORY,
    ) -> None:
        self.min_frequency = min_frequency
        self.missing_token = missing_token
        self.rare_token = rare_token

    def _normalise(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            values = X.to_numpy(dtype=object)
        else:
            values = np.asarray(X, dtype=object)
        if values.ndim != 2:
            raise PreprocessingContractError("범주형 입력은 2차원이어야 합니다.")
        normalised = values.copy()
        normalised[pd.isna(normalised)] = self.missing_token
        return normalised

    def _reject_reserved_tokens(self, X: Any) -> None:
        raw = (
            X.to_numpy(dtype=object)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=object)
        )
        if raw.ndim != 2:
            raise PreprocessingContractError("범주형 입력은 2차원이어야 합니다.")
        observed_tokens = {str(value) for value in raw.ravel() if not pd.isna(value)}
        reserved = {self.missing_token, self.rare_token}.intersection(observed_tokens)
        if reserved:
            raise PreprocessingContractError(
                f"원본 범주에 예약 토큰이 있습니다: {', '.join(sorted(reserved))}"
            )

    def fit(self, X: Any, y: Any = None) -> RareCategoryGrouper:
        if not 0 <= self.min_frequency < 1:
            raise PreprocessingContractError("희소범주 비율은 0 이상 1 미만이어야 합니다.")
        if self.missing_token == self.rare_token:
            raise PreprocessingContractError("결측과 희소범주 토큰은 달라야 합니다.")

        self._reject_reserved_tokens(X)

        values = self._normalise(X)
        self.n_features_in_ = values.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.rare_categories_ = []
        self.seen_categories_ = []
        for index in range(values.shape[1]):
            counts = pd.Series(values[:, index], dtype="object").value_counts(
                dropna=False
            )
            rates = counts / len(values) if len(values) else counts.astype(float)
            rare = {
                value
                for value, rate in rates.items()
                if float(rate) < self.min_frequency and value != self.missing_token
            }
            self.rare_categories_.append(frozenset(rare))
            self.seen_categories_.append(frozenset(counts.index.tolist()))
        self.rare_categories_ = tuple(self.rare_categories_)
        self.seen_categories_ = tuple(self.seen_categories_)
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._reject_reserved_tokens(X)
        values = self._normalise(X)
        if values.shape[1] != self.n_features_in_:
            raise PreprocessingContractError("범주형 입력 컬럼 수가 fit 시점과 다릅니다.")
        transformed = values.copy()
        for index, rare in enumerate(self.rare_categories_):
            if rare:
                mask = np.fromiter(
                    (value in rare for value in transformed[:, index]),
                    dtype=bool,
                    count=len(transformed),
                )
                transformed[mask, index] = self.rare_token
        return transformed

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        fallback = getattr(self, "feature_names_in_", None)
        return _feature_names(input_features, fallback)


def _to_float32(values: Any) -> Any:
    if sparse.issparse(values):
        return values.astype(np.float32)
    return np.asarray(values, dtype=np.float32)


def _to_csr_float32(values: Any) -> sparse.csr_matrix:
    if sparse.issparse(values):
        return values.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(values, dtype=np.float32))


def make_preprocessor(
    roles: FeatureRolesLike,
    *,
    model_family: ModelFamily,
    rare_min_frequency: float = DEFAULT_RARE_MIN_FREQUENCY,
) -> Pipeline:
    """피처 역할에 맞는 재현 가능한 sklearn 전처리기를 만든다."""

    if model_family not in {"linear", "tree"}:
        raise PreprocessingContractError("model_family는 linear 또는 tree여야 합니다.")
    if not roles.numeric_columns or not roles.categorical_columns:
        raise PreprocessingContractError("수치형과 범주형 피처가 모두 필요합니다.")

    numeric_value_steps: list[tuple[str, Any]] = [
        (
            "impute",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        )
    ]
    if model_family == "linear":
        numeric_value_steps.append(("scale", RobustScaler(with_centering=False)))
    numeric_value_steps.append(
        (
            "float32",
            FunctionTransformer(
                _to_float32,
                accept_sparse=True,
                feature_names_out="one-to-one",
            ),
        )
    )
    numeric_pipeline = FeatureUnion(
        [
            ("values", Pipeline(numeric_value_steps)),
            (
                "missing",
                Pipeline(
                    [
                        (
                            "indicator",
                            MissingIndicator(
                                features="missing-only",
                                sparse=True,
                                error_on_new=False,
                            ),
                        ),
                        (
                            "float32",
                            FunctionTransformer(
                                _to_float32,
                                accept_sparse=True,
                                feature_names_out="one-to-one",
                            ),
                        ),
                    ]
                ),
            ),
        ],
        verbose_feature_names_out=False,
    )

    categorical_pipeline = Pipeline(
        [
            (
                "rare",
                RareCategoryGrouper(min_frequency=rare_min_frequency),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    columns = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(roles.numeric_columns)),
            ("categorical", categorical_pipeline, list(roles.categorical_columns)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    return Pipeline(
        [
            (
                "contract",
                FeatureContractValidator(
                    feature_columns=roles.feature_columns,
                    numeric_columns=roles.numeric_columns,
                    categorical_columns=roles.categorical_columns,
                ),
            ),
            ("semantic_values", SemanticValueTransformer()),
            ("columns", columns),
            (
                "csr_float32",
                FunctionTransformer(
                    _to_csr_float32,
                    accept_sparse=True,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )


def transformed_feature_names(preprocessor: Pipeline) -> tuple[str, ...]:
    """fit된 전처리기의 출력 피처명을 순서대로 반환한다."""

    try:
        names = preprocessor.get_feature_names_out()
    except Exception as error:  # sklearn의 미학습 예외를 프로젝트 오류로 통일
        raise PreprocessingContractError(
            "fit된 전처리기에서만 출력 피처명을 조회할 수 있습니다."
        ) from error
    return tuple(str(name) for name in names)
