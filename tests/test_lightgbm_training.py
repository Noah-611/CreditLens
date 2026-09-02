"""Stage 5 1/3 LightGBM 실행 계약 테스트."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import creditlens.modeling.train_lightgbm as lightgbm_module
from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.data import DevelopmentDataset, ModelSplit
from creditlens.modeling.feature_roles import FeatureRoles, MartVersion
from creditlens.modeling.train_lightgbm import (
    EXPERIMENTS,
    LightGBMTrainingError,
    _capture_alignment,
    _fit_experiment,
    _lightgbm_settings,
    _load_baseline_reference,
    _validate_alignment,
    _validate_baseline_dataset,
    _validate_dataset,
    render_markdown_report,
    run_lightgbm_experiments,
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
        parquet_sha256=version * 32,
        schema_sha256=version.upper() * 32,
        audit={
            "feature_rows_loaded": {"train": 80, "validation": 20, "test": 0},
            "test_feature_rows_used": 0,
            "test_sealed": True,
            "stage3_summary_verified": True,
            "parquet_sha256_verified": True,
        },
    )


def _baseline_payload() -> dict:
    datasets = {version: _dataset(version) for version in ("v1", "v2", "v3")}
    validation_y = datasets["v1"].validation.y.to_numpy()
    scores = np.where(validation_y == 1, 0.8, 0.2)
    metrics = evaluate_binary_metrics(validation_y, scores, top_fraction=0.1)
    display_names = {
        "logistic_v1": "V1 Logistic Regression",
        "logistic_v2": "V2 Logistic Regression",
        "logistic_v3": "V3 Logistic Regression",
        "random_forest_v3": "V3 Random Forest",
    }
    versions = {
        "logistic_v1": "v1",
        "logistic_v2": "v2",
        "logistic_v3": "v3",
        "random_forest_v3": "v3",
    }
    return {
        "run_version": "stage4-test",
        "run_status": "complete",
        "settings": {
            "classification_threshold": 0.5,
            "top_fraction": 0.1,
        },
        "data_scope": {
            "train_rows": 80,
            "validation_rows": 20,
            "validation_positive_rate": 0.5,
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
        },
        "data_versions": {
            version: {
                "schema_sha256": dataset.schema_sha256,
                "parquet_sha256": dataset.parquet_sha256,
                "model_feature_columns": len(dataset.roles.model_features),
                "numeric_feature_columns": len(dataset.roles.numeric),
                "categorical_feature_columns": len(dataset.roles.categorical),
                "train_rows": len(dataset.train.y),
                "validation_rows": len(dataset.validation.y),
            }
            for version, dataset in datasets.items()
        },
        "experiments": [
            {
                "key": key,
                "display_name": display_names[key],
                "data_version": versions[key],
                "metrics": metrics,
            }
            for key in display_names
        ],
    }


def _write_baseline(path: Path) -> None:
    path.write_text(
        json.dumps(_baseline_payload(), ensure_ascii=False),
        encoding="utf-8",
    )


def test_lightgbm_settings_are_fixed_and_unweighted() -> None:
    first = _lightgbm_settings(2)
    second = _lightgbm_settings(2)

    assert first == second
    assert first["n_estimators"] == 500
    assert first["learning_rate"] == pytest.approx(0.05)
    assert first["subsample"] == pytest.approx(1.0)
    assert first["colsample_bytree"] == pytest.approx(1.0)
    assert first["class_weight"] is None
    assert first["deterministic"] is True
    assert first["force_col_wise"] is True


def test_lightgbm_pipeline_fits_sparse_preprocessing(tmp_path: Path) -> None:
    result, scores = _fit_experiment(
        EXPERIMENTS[0],
        _dataset(),
        artifact_dir=tmp_path,
        lightgbm_jobs=1,
    )

    assert scores.shape == (20,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    assert result["transformed_feature_columns"] >= 4
    assert result["iteration_details"]["early_stopping_used"] is False
    assert 1 <= result["iteration_details"]["n_estimators_fitted"] <= 500
    assert (tmp_path / "lightgbm_v1.joblib").is_file()


def test_fixed_seed_repeats_identical_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_settings = lightgbm_module._lightgbm_settings

    def small_settings(n_jobs: int) -> dict:
        settings = original_settings(n_jobs)
        settings.update({"n_estimators": 30, "min_child_samples": 5})
        return settings

    monkeypatch.setattr(lightgbm_module, "_lightgbm_settings", small_settings)
    _, first = _fit_experiment(
        EXPERIMENTS[0],
        _dataset(),
        artifact_dir=tmp_path / "first",
        lightgbm_jobs=1,
    )
    _, second = _fit_experiment(
        EXPERIMENTS[0],
        _dataset(),
        artifact_dir=tmp_path / "second",
        lightgbm_jobs=1,
    )

    np.testing.assert_array_equal(first, second)


def test_test_usage_and_version_misalignment_are_rejected() -> None:
    dataset = _dataset()
    invalid_test = DevelopmentDataset(
        version=dataset.version,
        train=dataset.train,
        validation=dataset.validation,
        roles=dataset.roles,
        parquet_sha256=dataset.parquet_sha256,
        schema_sha256=dataset.schema_sha256,
        audit={**dataset.audit, "test_feature_rows_used": 1},
    )
    with pytest.raises(LightGBMTrainingError, match="test 피처"):
        _validate_dataset(invalid_test)

    reference = _capture_alignment(_dataset("v1"))
    candidate = _dataset("v2")
    changed_target = candidate.train.y.copy()
    changed_target.iloc[0] = 1 - changed_target.iloc[0]
    invalid_alignment = DevelopmentDataset(
        version=candidate.version,
        train=ModelSplit(
            "train", candidate.train.X, changed_target, candidate.train.customer_ids
        ),
        validation=candidate.validation,
        roles=candidate.roles,
        parquet_sha256=candidate.parquet_sha256,
        schema_sha256=candidate.schema_sha256,
        audit=candidate.audit,
    )
    with pytest.raises(LightGBMTrainingError, match="train TARGET"):
        _validate_alignment(reference, invalid_alignment)


def test_stage4_reference_and_current_dataset_must_match(tmp_path: Path) -> None:
    reference_path = tmp_path / "baseline.json"
    _write_baseline(reference_path)
    reference = _load_baseline_reference(reference_path)
    _validate_baseline_dataset(reference, _dataset("v1"))

    changed = _dataset("v1")
    invalid = DevelopmentDataset(
        version=changed.version,
        train=changed.train,
        validation=changed.validation,
        roles=changed.roles,
        parquet_sha256="changed",
        schema_sha256=changed.schema_sha256,
        audit=changed.audit,
    )
    with pytest.raises(LightGBMTrainingError, match="parquet_sha256"):
        _validate_baseline_dataset(reference, invalid)

    payload = _baseline_payload()
    payload["run_status"] = "in_progress"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LightGBMTrainingError, match="완료된 Stage 4"):
        _load_baseline_reference(reference_path)


def test_stage4_reference_rejects_wrong_version_and_nonfinite_metric(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "baseline.json"
    payload = _baseline_payload()
    payload["experiments"][0]["data_version"] = "v2"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LightGBMTrainingError, match="데이터 버전"):
        _load_baseline_reference(reference_path)

    payload = _baseline_payload()
    payload["experiments"][0]["metrics"]["roc_auc"] = float("nan")
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LightGBMTrainingError, match="유한한 숫자"):
        _load_baseline_reference(reference_path)


def test_full_runner_executes_three_versions_without_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path)
    monkeypatch.setattr(
        lightgbm_module,
        "load_development_data",
        lambda version: _dataset(version),
    )

    payload = run_lightgbm_experiments(
        output_path=tmp_path / "results.json",
        report_path=tmp_path / "report.md",
        artifact_dir=tmp_path / "models",
        baseline_reference_path=baseline_path,
        lightgbm_jobs=1,
    )

    assert payload["run_status"] == "complete"
    assert [item["key"] for item in payload["experiments"]] == [
        "lightgbm_v1",
        "lightgbm_v2",
        "lightgbm_v3",
    ]
    assert payload["data_scope"]["test_feature_rows_used"] == 0
    assert payload["data_scope"]["test_predictions_created"] is False
    assert payload["settings"]["validation_used_for_fit"] is False
    assert payload["settings"]["validation_used_for_early_stopping"] is False
    assert (tmp_path / "models" / "lightgbm_v3.joblib").is_file()
    score_path = tmp_path / "models" / "lightgbm_validation_scores.joblib"
    assert score_path.is_file()
    stored = joblib.load(score_path)
    assert stored["customer_ids_included"] is False
    assert set(stored["scores"]) == {
        "lightgbm_v1",
        "lightgbm_v2",
        "lightgbm_v3",
    }
    recomputed = evaluate_binary_metrics(
        stored["y_true"], stored["scores"]["lightgbm_v3"], top_fraction=0.1
    )
    actual = payload["experiments"][2]["metrics"]
    assert recomputed["roc_auc"] == pytest.approx(actual["roc_auc"])
    assert recomputed["pr_auc"] == pytest.approx(actual["pr_auc"])
    assert payload["local_prediction_artifact"]["sha256"] == (
        lightgbm_module._sha256(score_path)
    )

    report = render_markdown_report(payload)
    assert "Stage 5 1/3" in report
    assert "test 피처 사용: 0행" in report
    assert "SK_ID_CURR" not in report
    assert "/home/" not in report
    json.dumps(payload, allow_nan=False)


def test_runner_rejects_baseline_output_collision(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path)
    original = baseline_path.read_bytes()

    with pytest.raises(LightGBMTrainingError, match="서로 달라야"):
        run_lightgbm_experiments(
            output_path=baseline_path,
            report_path=tmp_path / "report.md",
            artifact_dir=tmp_path / "models",
            baseline_reference_path=baseline_path,
            lightgbm_jobs=1,
        )

    assert baseline_path.read_bytes() == original


def test_git_safety_rejects_tracked_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / ".gitignore").write_text("models/**\n", encoding="utf-8")
    artifact = repository / "models" / "tracked.joblib"
    artifact.parent.mkdir()
    artifact.write_bytes(b"tracked")
    subprocess.run(
        ["git", "add", "-f", "models/tracked.joblib"],
        cwd=repository,
        check=True,
    )
    monkeypatch.chdir(repository)

    with pytest.raises(LightGBMTrainingError, match="이미 Git에 추적"):
        lightgbm_module._git_ignored(artifact)


@pytest.mark.parametrize("jobs", [0, 5])
def test_runner_rejects_unsafe_thread_counts(
    jobs: int,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path)
    with pytest.raises(LightGBMTrainingError, match="n_jobs"):
        run_lightgbm_experiments(
            output_path=tmp_path / "results.json",
            report_path=tmp_path / "report.md",
            artifact_dir=tmp_path / "models",
            baseline_reference_path=baseline_path,
            lightgbm_jobs=jobs,
        )
