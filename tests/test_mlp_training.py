"""Stage 5 2/3 TensorFlow MLP 실행 계약 테스트."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import creditlens.modeling.train_mlp as mlp_module
from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.data import DevelopmentDataset, ModelSplit
from creditlens.modeling.feature_roles import FeatureRoles
from creditlens.modeling.train_mlp import (
    MLPTrainingError,
    _balanced_class_weights,
    _build_mlp,
    _dense_float32,
    _load_references,
    _make_inner_split,
    _mlp_settings,
    _reset_tensorflow,
    _run_discovery,
    _validate_dataset,
    render_markdown_report,
    run_mlp_experiment,
)


def _dataset() -> DevelopmentDataset:
    roles = FeatureRoles(
        version="v3",
        numeric=("AMT_INCOME_TOTAL", "AMT_CREDIT"),
        categorical=("NAME_CONTRACT_TYPE",),
    )
    train_ids = pd.Index(range(1, 241), name="SK_ID_CURR")
    validation_ids = pd.Index(range(1001, 1061), name="SK_ID_CURR")
    train_target = pd.Series(
        [1 if index % 5 == 0 else 0 for index in range(240)],
        index=train_ids,
        dtype="int8",
        name="TARGET",
    )
    validation_target = pd.Series(
        [1 if index % 5 == 0 else 0 for index in range(60)],
        index=validation_ids,
        dtype="int8",
        name="TARGET",
    )
    rng = np.random.default_rng(42)

    def frame(y: np.ndarray, rows: int, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "AMT_INCOME_TOTAL": rng.normal(100.0, 15.0, rows),
                "AMT_CREDIT": rng.normal(50.0 + 35.0 * y, 8.0, rows),
                "NAME_CONTRACT_TYPE": np.where(y == 1, "Cash", "Revolving"),
            },
            index=index,
        )

    train = frame(train_target.to_numpy(), 240, train_ids)
    validation = frame(validation_target.to_numpy(), 60, validation_ids)
    return DevelopmentDataset(
        version="v3",
        train=ModelSplit("train", train, train_target, train_ids),
        validation=ModelSplit(
            "validation", validation, validation_target, validation_ids
        ),
        roles=roles,
        parquet_sha256="a" * 64,
        schema_sha256="b" * 64,
        audit={
            "feature_rows_loaded": {"train": 240, "validation": 60, "test": 0},
            "test_feature_rows_used": 0,
            "test_sealed": True,
            "stage3_summary_verified": True,
            "parquet_sha256_verified": True,
        },
    )


def _reference_metrics() -> dict:
    labels = _dataset().validation.y.to_numpy()
    scores = np.where(labels == 1, 0.75, 0.15)
    return evaluate_binary_metrics(
        labels,
        scores,
        threshold=0.5,
        top_fraction=0.1,
    )


def _write_references(directory: Path) -> tuple[Path, Path]:
    dataset = _dataset()
    metrics = _reference_metrics()
    stage4_path = directory / "stage4.json"
    stage4 = {
        "run_version": "stage4-test",
        "run_status": "complete",
        "settings": {
            "classification_threshold": 0.5,
            "top_fraction": 0.1,
        },
        "data_scope": {
            "train_rows": 240,
            "validation_rows": 60,
            "validation_positive_rate": 0.2,
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
        },
        "data_versions": {
            "v1": {},
            "v2": {},
            "v3": {
                "schema_sha256": dataset.schema_sha256,
                "parquet_sha256": dataset.parquet_sha256,
                "model_feature_columns": 3,
                "numeric_feature_columns": 2,
                "categorical_feature_columns": 1,
                "train_rows": 240,
                "validation_rows": 60,
            },
        },
        "experiments": [
            {
                "key": key,
                "display_name": name,
                "data_version": version,
                "metrics": copy.deepcopy(metrics),
            }
            for key, name, version in (
                ("logistic_v1", "V1 Logistic Regression", "v1"),
                ("logistic_v2", "V2 Logistic Regression", "v2"),
                ("logistic_v3", "V3 Logistic Regression", "v3"),
                ("random_forest_v3", "V3 Random Forest", "v3"),
            )
        ],
    }
    stage4_path.write_text(json.dumps(stage4), encoding="utf-8")
    lightgbm_path = directory / "lightgbm.json"
    lightgbm = {
        "run_version": "stage5-lightgbm-test",
        "stage_part": "1/3",
        "run_status": "complete",
        "settings": {
            "classification_threshold": 0.5,
            "top_fraction": 0.1,
        },
        "data_scope": {
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
            "row_level_predictions_in_shared_outputs": False,
        },
        "baseline_reference": {
            "sha256": mlp_module._sha256(stage4_path),
        },
        "data_versions": {
            "v3": copy.deepcopy(stage4["data_versions"]["v3"]),
        },
        "experiments": [
            {
                "key": "lightgbm_v3",
                "display_name": "V3 LightGBM",
                "data_version": "v3",
                "metrics": copy.deepcopy(metrics),
            }
        ],
    }
    lightgbm_path.write_text(json.dumps(lightgbm), encoding="utf-8")
    return stage4_path, lightgbm_path


def _small_settings() -> dict:
    settings = _mlp_settings()
    settings.update(
        {
            "hidden_units": [8, 4],
            "dropout_rates": [0.1, 0.0],
            "metric_thresholds": 50,
            "batch_size": 32,
            "predict_batch_size": 64,
            "max_epochs": 4,
            "early_stopping_patience": 1,
        }
    )
    return settings


def test_mlp_architecture_matches_fixed_contract() -> None:
    settings = _mlp_settings()
    _reset_tensorflow(intra_threads=2, inter_threads=1)
    model = _build_mlp(420, settings)

    assert model.input_shape == (None, 420)
    assert model.output_shape == (None, 1)
    assert model.count_params() == 62_209
    assert settings["hidden_units"] == [128, 64]
    assert settings["batch_size"] == 1024
    assert settings["early_stopping_monitor"] == "val_pr_auc_approx"
    assert settings["max_epochs"] == 50


def test_inner_split_is_reproducible_stratified_and_disjoint() -> None:
    labels = _dataset().train.y.to_numpy()
    first = _make_inner_split(labels)
    second = _make_inner_split(labels)

    np.testing.assert_array_equal(first.fit_positions, second.fit_positions)
    np.testing.assert_array_equal(
        first.early_stop_positions, second.early_stop_positions
    )
    assert len(first.fit_positions) == 216
    assert len(first.early_stop_positions) == 24
    assert np.intersect1d(
        first.fit_positions, first.early_stop_positions
    ).size == 0
    assert first.audit["fit"]["positive_rate"] == pytest.approx(
        0.2, abs=1 / len(first.fit_positions)
    )
    assert first.audit["early_stop"]["positive_rate"] == pytest.approx(
        0.2, abs=1 / len(first.early_stop_positions)
    )


def test_balanced_class_weight_uses_only_given_labels() -> None:
    labels = np.array([0] * 8 + [1] * 2, dtype=np.int8)
    weights = _balanced_class_weights(labels)

    assert weights[0] == pytest.approx(10 / 16)
    assert weights[1] == pytest.approx(10 / 4)
    assert np.mean([weights[int(label)] for label in labels]) == pytest.approx(1.0)


def test_dense_conversion_is_float32_finite_and_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    dense, audit = _dense_float32(values, label="test")

    assert dense.dtype == np.float32
    assert dense.flags.c_contiguous
    assert audit["source_sparse"] is True
    assert audit["full_dense_materialization"] is True
    assert audit["source_nnz"] == 2

    with pytest.raises(MLPTrainingError, match="결측값이나 무한값"):
        _dense_float32(np.array([[np.nan]], dtype=np.float32), label="invalid")
    monkeypatch.setattr(mlp_module, "MAX_DENSE_MATRIX_MIB", 0.000001)
    with pytest.raises(MLPTrainingError, match="메모리 제한"):
        _dense_float32(values, label="too-large")


def test_references_and_test_seal_are_verified(tmp_path: Path) -> None:
    stage4_path, lightgbm_path = _write_references(tmp_path)
    stage4, lightgbm = _load_references(stage4_path, lightgbm_path)
    _validate_dataset(_dataset(), stage4, lightgbm)

    dataset = _dataset()
    invalid = DevelopmentDataset(
        version=dataset.version,
        train=dataset.train,
        validation=dataset.validation,
        roles=dataset.roles,
        parquet_sha256=dataset.parquet_sha256,
        schema_sha256=dataset.schema_sha256,
        audit={**dataset.audit, "test_feature_rows_used": 1},
    )
    with pytest.raises(MLPTrainingError, match="test 봉인"):
        _validate_dataset(invalid, stage4, lightgbm)

    changed = json.loads(lightgbm_path.read_text())
    changed["baseline_reference"]["sha256"] = "0" * 64
    lightgbm_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(MLPTrainingError, match="Stage 4 결과"):
        _load_references(stage4_path, lightgbm_path)


def test_discovery_does_not_access_official_validation() -> None:
    dataset = _dataset()
    poisoned = DevelopmentDataset(
        version=dataset.version,
        train=dataset.train,
        validation=ModelSplit(
            "validation",
            None,  # type: ignore[arg-type]
            dataset.validation.y,
            dataset.validation.customer_ids,
        ),
        roles=dataset.roles,
        parquet_sha256=dataset.parquet_sha256,
        schema_sha256=dataset.schema_sha256,
        audit=dataset.audit,
    )
    split = _make_inner_split(dataset.train.y.to_numpy())
    result = _run_discovery(
        poisoned,
        split,
        _small_settings(),
        intra_threads=2,
        inter_threads=1,
        fit_verbose=0,
    )

    assert result["official_validation_used"] is False
    assert 1 <= result["best_epoch"] <= result["epochs_ran"] <= 4
    assert result["best_epoch"] == (
        int(np.argmax(result["history"]["val_pr_auc_approx"])) + 1
    )
    assert result["preprocessor_fit_scope"] == "inner_fit_only"


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)
def test_full_runner_trains_refits_and_saves_ignored_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage4_path, lightgbm_path = _write_references(tmp_path)
    monkeypatch.setattr(mlp_module, "load_development_data", lambda version: _dataset())
    monkeypatch.setattr(mlp_module, "_mlp_settings", _small_settings)

    payload = run_mlp_experiment(
        output_path=tmp_path / "results.json",
        report_path=tmp_path / "report.md",
        figure_path=tmp_path / "history.png",
        artifact_dir=tmp_path / "models",
        stage4_reference_path=stage4_path,
        lightgbm_reference_path=lightgbm_path,
        intra_threads=2,
        inter_threads=1,
        fit_verbose=0,
    )

    assert payload["run_status"] == "complete"
    assert payload["stage_part"] == "2/3"
    assert payload["data_scope"]["test_feature_rows_used"] == 0
    assert payload["settings"]["official_validation_used_for_early_stopping"] is False
    experiment = payload["experiment"]
    assert experiment["best_epoch"] == payload["inner_development"]["best_epoch"]
    assert experiment["full_refit_validation_data_used"] is False
    assert experiment["full_refit_callbacks_used"] is False
    assert experiment["official_validation_prediction_calls"] == 1
    assert experiment["probability_calibrated"] is False
    assert experiment["brier_comparable_to_unweighted_models"] is False
    assert (tmp_path / "models" / "mlp_v3.keras").is_file()
    assert (tmp_path / "models" / "mlp_v3_preprocessor.joblib").is_file()
    assert (tmp_path / "models" / "mlp_validation_scores.joblib").is_file()
    assert (tmp_path / "models" / "mlp_v3_manifest.json").is_file()
    assert (tmp_path / "history.png").is_file()

    score_artifact = joblib.load(tmp_path / "models" / "mlp_validation_scores.joblib")
    assert score_artifact["customer_ids_included"] is False
    stored_scores = score_artifact["scores"]["mlp_v3"]
    recomputed = evaluate_binary_metrics(
        score_artifact["y_true"], stored_scores, threshold=0.5, top_fraction=0.1
    )
    assert recomputed["roc_auc"] == pytest.approx(experiment["metrics"]["roc_auc"])
    assert recomputed["pr_auc"] == pytest.approx(experiment["metrics"]["pr_auc"])

    model = mlp_module.keras.models.load_model(tmp_path / "models" / "mlp_v3.keras")
    preprocessor = joblib.load(tmp_path / "models" / "mlp_v3_preprocessor.joblib")
    transformed, _ = _dense_float32(
        preprocessor.transform(_dataset().validation.X), label="replay"
    )
    replayed = model.predict(transformed, batch_size=64, verbose=0).reshape(-1)
    np.testing.assert_array_equal(replayed.astype(np.float32), stored_scores)

    report = render_markdown_report(payload)
    assert "Stage 5 2/3" in report
    assert "test 피처 사용: 0행" in report
    assert "보정된 실제 확률이 아니" in report
    assert "SK_ID_CURR" not in report
    assert "/home/" not in report
    json.dumps(payload, allow_nan=False)


def test_same_seed_repeats_tiny_training_scores() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(64, 6)).astype(np.float32)
    labels = np.array([0, 1] * 32, dtype=np.int8)
    settings = _small_settings()

    predictions = []
    for _ in range(2):
        _reset_tensorflow(intra_threads=2, inter_threads=1)
        model = _build_mlp(6, settings)
        model.fit(values, labels, epochs=2, batch_size=16, shuffle=True, verbose=0)
        predictions.append(model.predict(values, batch_size=32, verbose=0))

    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_runner_rejects_reference_output_collision(tmp_path: Path) -> None:
    stage4_path, lightgbm_path = _write_references(tmp_path)
    original = stage4_path.read_bytes()
    with pytest.raises(MLPTrainingError, match="서로 달라야"):
        run_mlp_experiment(
            output_path=stage4_path,
            report_path=tmp_path / "report.md",
            figure_path=tmp_path / "history.png",
            artifact_dir=tmp_path / "models",
            stage4_reference_path=stage4_path,
            lightgbm_reference_path=lightgbm_path,
            fit_verbose=0,
        )
    assert stage4_path.read_bytes() == original


def test_mlp_git_safety_rejects_tracked_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / ".gitignore").write_text("models/**\n", encoding="utf-8")
    artifact = repository / "models" / "tracked.keras"
    artifact.parent.mkdir()
    artifact.write_bytes(b"tracked")
    subprocess.run(
        ["git", "add", "-f", "models/tracked.keras"],
        cwd=repository,
        check=True,
    )
    monkeypatch.chdir(repository)

    with pytest.raises(MLPTrainingError, match="이미 Git에 추적"):
        mlp_module._artifact_ignore_status(artifact)
