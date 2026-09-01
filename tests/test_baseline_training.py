"""Stage 4 기준 모델 실행 계약 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import creditlens.modeling.train_baselines as baseline_module
from creditlens.modeling.data import DevelopmentDataset, ModelSplit
from creditlens.modeling.feature_roles import FeatureRoles, MartVersion
from creditlens.modeling.train_baselines import (
    EXPERIMENTS,
    BaselineTrainingError,
    _fit_experiment,
    _capture_alignment,
    _make_estimator,
    _validate_alignment,
    _validate_dataset,
    run_baseline_experiments,
    render_markdown_report,
)


def _dataset(version: MartVersion = "v1") -> DevelopmentDataset:
    roles = FeatureRoles(
        version=version,
        numeric=("AMT_INCOME_TOTAL", "AMT_CREDIT"),
        categorical=("NAME_CONTRACT_TYPE",),
    )
    train_ids = pd.Index(range(1, 81), name="SK_ID_CURR")
    validation_ids = pd.Index(range(101, 121), name="SK_ID_CURR")
    train_target = pd.Series(
        [index % 2 for index in range(80)],
        index=train_ids,
        dtype="int8",
        name="TARGET",
    )
    validation_target = pd.Series(
        [index % 2 for index in range(20)],
        index=validation_ids,
        dtype="int8",
        name="TARGET",
    )
    train = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": np.linspace(1.0, 80.0, 80),
            "AMT_CREDIT": np.where(train_target.to_numpy() == 1, 100.0, 10.0),
            "NAME_CONTRACT_TYPE": np.where(
                train_target.to_numpy() == 1, "Cash", "Revolving"
            ),
        },
        index=train_ids,
    )
    validation = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": np.linspace(10.0, 29.0, 20),
            "AMT_CREDIT": np.where(
                validation_target.to_numpy() == 1, 100.0, 10.0
            ),
            "NAME_CONTRACT_TYPE": np.where(
                validation_target.to_numpy() == 1, "Cash", "Revolving"
            ),
        },
        index=validation_ids,
    )
    return DevelopmentDataset(
        version=version,
        train=ModelSplit("train", train, train_target, train_ids),
        validation=ModelSplit(
            "validation", validation, validation_target, validation_ids
        ),
        roles=roles,
        parquet_sha256="a" * 64,
        schema_sha256="b" * 64,
        audit={
            "feature_rows_loaded": {"train": 80, "validation": 20, "test": 0},
            "test_feature_rows_used": 0,
            "test_sealed": True,
            "stage3_summary_verified": True,
            "parquet_sha256_verified": True,
        },
    )


def _spec(key: str):
    return next(spec for spec in EXPERIMENTS if spec.key == key)


def test_fixed_estimators_match_baseline_contract() -> None:
    dummy, dummy_settings = _make_estimator(
        _spec("dummy_prior"), random_forest_jobs=2
    )
    logistic, logistic_settings = _make_estimator(
        _spec("logistic_v1"), random_forest_jobs=2
    )
    forest, forest_settings = _make_estimator(
        _spec("random_forest_v3"), random_forest_jobs=2
    )

    assert isinstance(dummy, DummyClassifier)
    assert dummy_settings["strategy"] == "prior"
    assert isinstance(logistic, LogisticRegression)
    assert logistic_settings["solver"] == "lbfgs"
    assert logistic_settings["max_iter"] == 600
    assert logistic_settings["class_weight"] is None
    assert isinstance(forest, RandomForestClassifier)
    assert forest_settings["max_depth"] == 12
    assert forest_settings["n_jobs"] == 2
    assert forest_settings["class_weight"] == "balanced_subsample"


def test_dummy_and_logistic_fit_only_development_data(tmp_path: Path) -> None:
    dataset = _dataset()
    dummy_result, dummy_scores = _fit_experiment(
        _spec("dummy_prior"),
        dataset,
        artifact_dir=tmp_path,
        random_forest_jobs=1,
    )
    logistic_result, logistic_scores = _fit_experiment(
        _spec("logistic_v1"),
        dataset,
        artifact_dir=tmp_path,
        random_forest_jobs=1,
    )

    assert np.all(dummy_scores == pytest.approx(0.5))
    assert dummy_result["metrics"]["roc_auc"] == pytest.approx(0.5)
    assert dummy_result["metrics"]["top_k_metrics"]["lift"] == pytest.approx(1.0)
    assert logistic_result["metrics"]["roc_auc"] == pytest.approx(1.0)
    assert logistic_scores.shape == (20,)
    assert logistic_result["transformed_feature_columns"] >= 4
    assert logistic_result["convergence"]["converged"] is True
    assert (tmp_path / "dummy_prior.joblib").is_file()
    assert (tmp_path / "logistic_v1.joblib").is_file()


def test_test_usage_or_exposure_is_rejected() -> None:
    dataset = _dataset()
    invalid = DevelopmentDataset(
        version=dataset.version,
        train=dataset.train,
        validation=dataset.validation,
        roles=dataset.roles,
        parquet_sha256=dataset.parquet_sha256,
        schema_sha256=dataset.schema_sha256,
        audit={
            **dataset.audit,
            "test_feature_rows_used": 1,
        },
    )

    with pytest.raises(BaselineTrainingError, match="test 피처"):
        _validate_dataset(invalid)


def test_version_alignment_rejects_changed_train_target() -> None:
    reference = _capture_alignment(_dataset("v1"))
    candidate = _dataset("v2")
    changed_target = candidate.train.y.copy()
    changed_target.iloc[0] = 1 - changed_target.iloc[0]
    invalid = DevelopmentDataset(
        version=candidate.version,
        train=ModelSplit(
            "train",
            candidate.train.X,
            changed_target,
            candidate.train.customer_ids,
        ),
        validation=candidate.validation,
        roles=candidate.roles,
        parquet_sha256=candidate.parquet_sha256,
        schema_sha256=candidate.schema_sha256,
        audit=candidate.audit,
    )

    with pytest.raises(BaselineTrainingError, match="train TARGET"):
        _validate_alignment(reference, invalid)


def test_full_runner_executes_all_baselines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        baseline_module,
        "load_development_data",
        lambda version: _dataset(version),
    )

    payload = run_baseline_experiments(
        output_path=tmp_path / "results.json",
        report_path=tmp_path / "report.md",
        artifact_dir=tmp_path / "models",
        random_forest_jobs=1,
    )

    assert payload["run_status"] == "complete"
    assert [item["key"] for item in payload["experiments"]] == [
        spec.key for spec in EXPERIMENTS
    ]
    assert payload["data_scope"]["test_feature_rows_used"] == 0
    assert payload["data_scope"]["test_predictions_created"] is False
    assert (tmp_path / "models" / "random_forest_v3.joblib").is_file()
    assert (tmp_path / "models" / "validation_scores.joblib").is_file()
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "report.md").is_file()


def test_markdown_report_uses_only_aggregate_results(tmp_path: Path) -> None:
    result, _ = _fit_experiment(
        _spec("dummy_prior"),
        _dataset(),
        artifact_dir=tmp_path,
        random_forest_jobs=1,
    )
    payload = {
        "settings": {
            "random_seed": 42,
            "classification_threshold": 0.5,
            "top_fraction": 0.1,
        },
        "data_scope": {
            "train_rows": 80,
            "validation_rows": 20,
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
        },
        "experiments": [result],
        "comparisons": {
            "logistic_v2_minus_v1": {
                "roc_auc": 0.0,
                "pr_auc": 0.0,
                "ks": 0.0,
                "brier_score": 0.0,
            },
            "logistic_v3_minus_v2": {
                "roc_auc": 0.0,
                "pr_auc": 0.0,
                "ks": 0.0,
                "brier_score": 0.0,
            },
        },
        "resources": {
            "total_seconds": 1.0,
            "process_peak_rss_mb": 10.0,
        },
    }

    report = render_markdown_report(payload)

    assert "Dummy Prior" in report
    assert "test 피처 사용: 0행" in report
    assert "SK_ID_CURR" not in report
    json.dumps(result, allow_nan=False)
