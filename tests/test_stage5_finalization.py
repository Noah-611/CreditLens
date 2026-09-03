"""Stage 5 3/3 제한 튜닝·피처군·확률 보정 실행 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import creditlens.modeling.finalize_stage5 as final_module
from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.calibration import (
    IdentityCalibrator,
    IsotonicScoreCalibrator,
    SigmoidLogitCalibrator,
)
from creditlens.modeling.data import DevelopmentDataset, ModelSplit
from creditlens.modeling.feature_roles import FeatureRoles
from creditlens.modeling.finalize_stage5 import (
    TuningCandidate,
    _crossfit_calibrators,
    _lightgbm_candidates,
    _make_cv_splits,
    _resolve_feature_groups,
    _select_calibration_method,
    _select_tuning_candidate,
    _write_completed_outputs,
    run_stage5_finalization,
)
from creditlens.modeling.train_lightgbm import _lightgbm_settings


def _roles_by_version() -> dict[str, FeatureRoles]:
    application_numeric = tuple(f"APP_NUM_{index:03d}" for index in range(131))
    application_categorical = ("APP_CATEGORY",)
    bureau = tuple(f"BUREAU_FEATURE_{index:02d}" for index in range(37))
    installments = tuple(f"INST_FEATURE_{index:02d}" for index in range(29))
    return {
        "v1": FeatureRoles(
            version="v1",
            numeric=application_numeric,
            categorical=application_categorical,
        ),
        "v2": FeatureRoles(
            version="v2",
            numeric=application_numeric + bureau,
            categorical=application_categorical,
        ),
        "v3": FeatureRoles(
            version="v3",
            numeric=application_numeric + bureau + installments,
            categorical=application_categorical,
        ),
    }


def _small_dataset() -> DevelopmentDataset:
    roles = _roles_by_version()["v3"]
    rng = np.random.default_rng(42)
    train_rows = 90
    validation_rows = 30
    train_y = np.asarray([0, 0, 0, 0, 1] * 18, dtype=np.int8)
    validation_y = np.asarray([0, 0, 0, 0, 1] * 6, dtype=np.int8)

    def frame(rows: int, labels: np.ndarray) -> pd.DataFrame:
        values: dict[str, object] = {}
        for column in roles.numeric:
            values[column] = rng.normal(size=rows)
        # 각 정보 원천에 작은 신호를 넣어 보정기의 단조 방향을 안정적으로 만든다.
        values["APP_NUM_000"] = labels + rng.normal(scale=0.1, size=rows)
        values["BUREAU_FEATURE_00"] = labels + rng.normal(scale=0.2, size=rows)
        values["INST_FEATURE_00"] = labels + rng.normal(scale=0.2, size=rows)
        values["APP_CATEGORY"] = np.where(labels == 1, "risk", "normal")
        return pd.DataFrame(values).loc[:, roles.model_features]

    train_ids = pd.Index(np.arange(1, train_rows + 1), name="SK_ID_CURR")
    validation_ids = pd.Index(
        np.arange(1_001, 1_001 + validation_rows), name="SK_ID_CURR"
    )
    train_target = pd.Series(train_y, index=train_ids, name="TARGET", dtype="int8")
    validation_target = pd.Series(
        validation_y, index=validation_ids, name="TARGET", dtype="int8"
    )
    train_X = frame(train_rows, train_y)
    validation_X = frame(validation_rows, validation_y)
    train_X.index = train_ids
    validation_X.index = validation_ids
    return DevelopmentDataset(
        version="v3",
        train=ModelSplit("train", train_X, train_target, train_ids),
        validation=ModelSplit(
            "validation", validation_X, validation_target, validation_ids
        ),
        roles=roles,
        parquet_sha256="p3",
        schema_sha256="h3",
        audit={
            "feature_rows_loaded": {
                "train": train_rows,
                "validation": validation_rows,
                "test": 0,
            },
            "test_feature_rows_used": 0,
            "test_sealed": True,
            "stage3_summary_verified": True,
            "parquet_sha256_verified": True,
        },
    )


def _small_candidates(n_jobs: int = 1) -> tuple[TuningCandidate, ...]:
    base = final_module._base_lightgbm_settings(n_jobs)
    base.update(
        {
            "n_estimators": 12,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 2,
        }
    )
    return (
        TuningCandidate("baseline", "기준", dict(base), 0),
        TuningCandidate(
            "regularized_sampling",
            "규제",
            {**base, "reg_lambda": 2.0, "colsample_bytree": 0.8},
            1,
        ),
        TuningCandidate(
            "higher_capacity_regularized",
            "확대",
            {**base, "num_leaves": 15, "reg_lambda": 3.0},
            2,
        ),
    )


def _reference_payloads(
    dataset: DevelopmentDataset,
) -> tuple[dict, dict, dict, dict[str, str]]:
    roles = _roles_by_version()
    hashes = {"v1": "h1", "v2": "h2", "v3": "h3"}
    parquet = {"v1": "p1", "v2": "p2", "v3": "p3"}
    versions = {
        version: {
            "schema_sha256": hashes[version],
            "parquet_sha256": parquet[version],
            "model_feature_columns": len(role.model_features),
            "numeric_feature_columns": len(role.numeric),
            "categorical_feature_columns": len(role.categorical),
            "train_rows": len(dataset.train.y),
            "validation_rows": len(dataset.validation.y),
        }
        for version, role in roles.items()
    }
    validation_y = dataset.validation.y.to_numpy()
    reference_scores = np.where(validation_y == 1, 0.8, 0.1)
    metrics = evaluate_binary_metrics(validation_y, reference_scores)
    stage4 = {
        "run_version": "stage4-test",
        "data_scope": {
            "train_rows": len(dataset.train.y),
            "validation_rows": len(dataset.validation.y),
            "validation_positive_rate": float(dataset.validation.y.mean()),
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
            "row_level_predictions_in_shared_outputs": False,
        },
        "data_versions": versions,
        "experiments": [
            {
                "key": "logistic_v3",
                "display_name": "V3 Logistic Regression",
                "metrics": metrics,
            },
            {
                "key": "random_forest_v3",
                "display_name": "V3 Random Forest",
                "metrics": metrics,
            },
        ],
    }
    lightgbm = {
        "run_version": "lightgbm-test",
        "settings": {"lightgbm": dict(_small_candidates(1)[0].settings)},
        "data_scope": dict(stage4["data_scope"]),
        "data_versions": versions,
        "experiments": [
            {
                "key": "lightgbm_v3",
                "display_name": "V3 LightGBM",
                "metrics": metrics,
            }
        ],
    }
    mlp = {
        "run_version": "mlp-test",
        "data_scope": dict(stage4["data_scope"]),
        "data_version": {
            key: value
            for key, value in versions["v3"].items()
            if key not in {"train_rows", "validation_rows"}
        },
        "experiment": {
            "key": "mlp_v3",
            "display_name": "V3 TensorFlow MLP",
            "metrics": metrics,
        },
    }
    return stage4, lightgbm, mlp, hashes


def _selection_result(
    key: str,
    *,
    pr_auc: float,
    roc_auc: float = 0.75,
    recall: float = 0.33,
    complexity_order: int = 1,
) -> dict:
    folds = [
        {
            "metrics": {
                "pr_auc": pr_auc + offset,
                "roc_auc": roc_auc + offset,
                "top_k_metrics": {"recall": recall + offset},
            }
        }
        for offset in (-0.0001, 0.0, 0.0001)
    ]
    return {
        "key": key,
        "display_name": key,
        "complexity_order": complexity_order,
        "folds": folds,
        "fold_aggregate": {
            "pr_auc": {"mean": pr_auc},
            "roc_auc": {"mean": roc_auc},
            "recall_at_10pct": {"mean": recall},
        },
    }


def test_tuning_registry_is_small_fixed_deterministic_and_unweighted() -> None:
    first = _lightgbm_candidates(2)
    second = _lightgbm_candidates(2)

    assert first == second
    assert [candidate.key for candidate in first] == [
        "baseline",
        "regularized_sampling",
        "higher_capacity_regularized",
    ]
    assert len(first) == 3
    assert all(candidate.settings["class_weight"] is None for candidate in first)
    assert all(candidate.settings["deterministic"] is True for candidate in first)
    assert all(candidate.settings["n_estimators"] == 500 for candidate in first)
    assert first[0].settings == _lightgbm_settings(2)


def test_cv_splits_are_reproducible_disjoint_stratified_and_cover_once() -> None:
    y = np.asarray([0] * 90 + [1] * 10, dtype=np.int8)
    first = _make_cv_splits(y, n_splits=3, seed=42)
    second = _make_cv_splits(y, n_splits=3, seed=42)
    coverage = np.zeros(y.size, dtype=np.int8)

    for (fit, holdout), (fit_again, holdout_again) in zip(
        first, second, strict=True
    ):
        np.testing.assert_array_equal(fit, fit_again)
        np.testing.assert_array_equal(holdout, holdout_again)
        assert np.intersect1d(fit, holdout).size == 0
        assert set(np.unique(y[fit])) == {0, 1}
        assert set(np.unique(y[holdout])) == {0, 1}
        coverage[holdout] += 1

    np.testing.assert_array_equal(coverage, np.ones(y.size, dtype=np.int8))


def test_feature_groups_use_version_differences_and_cover_v3() -> None:
    roles = _roles_by_version()
    groups = _resolve_feature_groups(roles["v1"], roles["v2"], roles["v3"])
    group_sets = [set(values) for values in groups.as_dict().values()]

    assert len(groups.application) == 132
    assert len(groups.bureau) == 37
    assert len(groups.installments) == 29
    assert all(left.isdisjoint(right) for left, right in combinations(group_sets, 2))
    assert set().union(*group_sets) == set(roles["v3"].model_features)
    assert "APP_CATEGORY" in groups.application


def test_tuning_selection_requires_material_paired_improvement() -> None:
    baseline = _selection_result("baseline", pr_auc=0.2400, complexity_order=0)
    too_small = _selection_result("regularized_sampling", pr_auc=0.2409)
    accepted = _selection_result(
        "higher_capacity_regularized",
        pr_auc=0.2420,
        roc_auc=0.7505,
        recall=0.331,
        complexity_order=2,
    )

    selected = _select_tuning_candidate([baseline, too_small, accepted])
    fallback = _select_tuning_candidate([baseline, too_small])
    simpler = _selection_result(
        "regularized_sampling", pr_auc=0.2421, complexity_order=1
    )
    marginally_higher_complex = _selection_result(
        "higher_capacity_regularized", pr_auc=0.2425, complexity_order=2
    )
    practical_tie = _select_tuning_candidate(
        [baseline, simpler, marginally_higher_complex]
    )

    assert selected["selected_key"] == "higher_capacity_regularized"
    assert fallback["selected_key"] == "baseline"
    assert practical_tie["selected_key"] == "regularized_sampling"
    assert selected["official_validation_used"] is False


def test_calibration_selection_can_improve_or_keep_identity() -> None:
    def experiment(method: str, brier: float, loss: float) -> dict:
        return {
            "method": method,
            "crossfit_metrics": {"brier_score": brier},
            "crossfit_log_loss": loss,
            "folds": [
                {"metrics": {"brier_score": brier + offset}}
                for offset in (-0.00002, -0.00001, 0.0, 0.00001, 0.00002)
            ],
        }

    improved = _select_calibration_method(
        [
            experiment("identity", 0.0700, 0.2500),
            experiment("sigmoid", 0.0680, 0.2450),
            experiment("isotonic", 0.0690, 0.2470),
        ]
    )
    fallback = _select_calibration_method(
        [
            experiment("identity", 0.0700, 0.2500),
            experiment("sigmoid", 0.0701, 0.2499),
            experiment("isotonic", 0.0702, 0.2501),
        ]
    )
    practical_tie = _select_calibration_method(
        [
            experiment("identity", 0.0700, 0.2500),
            experiment("sigmoid", 0.06805, 0.2450),
            experiment("isotonic", 0.06800, 0.2449),
        ]
    )

    assert improved["selected_method"] == "sigmoid"
    assert fallback["selected_method"] == "identity"
    assert practical_tie["selected_method"] == "sigmoid"
    assert improved["official_validation_used"] is False


def test_calibrators_are_cross_fitted_and_return_finite_scores() -> None:
    rng = np.random.default_rng(7)
    y = np.asarray([0] * 80 + [1] * 20, dtype=np.int8)
    raw = np.clip(0.08 + 0.45 * y + rng.normal(scale=0.05, size=y.size), 0.001, 0.999)

    result = _crossfit_calibrators(y, raw, n_splits=5, seed=43)

    assert result.selected_method in {"identity", "sigmoid", "isotonic"}
    assert {item["method"] for item in result.experiments} == {
        "identity",
        "sigmoid",
        "isotonic",
    }
    for scores in result.crossfit_scores.values():
        assert scores.shape == y.shape
        assert np.isfinite(scores).all()
        assert ((scores >= 0.0) & (scores <= 1.0)).all()
    assert all(len(item["folds"]) == 5 for item in result.experiments)
    assert result.fitted_calibrator.fit_rows_ == y.size
    assert result.fitted_calibrator.sample_weight_used_ is False


def test_calibrator_artifacts_use_an_importable_module(tmp_path: Path) -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)

    for calibrator_class in (
        IdentityCalibrator,
        SigmoidLogitCalibrator,
        IsotonicScoreCalibrator,
    ):
        assert calibrator_class.__module__ == "creditlens.modeling.calibration"
        fitted = calibrator_class().fit(scores, labels)
        path = tmp_path / f"{fitted.method}.joblib"
        joblib.dump(fitted, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(restored.predict(scores), fitted.predict(scores))


@pytest.mark.filterwarnings("ignore:Glyph.*missing from font.*:UserWarning")
def test_full_runner_uses_one_validation_prediction_and_seals_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _small_dataset()
    roles = _roles_by_version()
    stage4, lightgbm, mlp, hashes = _reference_payloads(dataset)
    reference_paths = [
        tmp_path / "stage4.json",
        tmp_path / "lightgbm.json",
        tmp_path / "mlp.json",
    ]
    for path in reference_paths:
        path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        final_module,
        "_load_references",
        lambda *_: (stage4, lightgbm, mlp),
    )
    monkeypatch.setattr(
        final_module,
        "_load_roles_from_schema",
        lambda version: (roles[version], hashes[version]),
    )
    monkeypatch.setattr(final_module, "_lightgbm_candidates", _small_candidates)
    original_positive_scores = final_module._positive_scores
    validation_calls = 0
    split_load_calls: list[str] = []

    def counted_scores(model: object, X: object) -> np.ndarray:
        nonlocal validation_calls
        if X is dataset.validation.X:
            validation_calls += 1
        return original_positive_scores(model, X)

    monkeypatch.setattr(final_module, "_positive_scores", counted_scores)
    output = tmp_path / "stage5_final.json"
    report = tmp_path / "stage5_final.md"
    calibration_figure = tmp_path / "calibration.png"
    ablation_figure = tmp_path / "ablation.png"
    artifact_dir = tmp_path / "models"

    def load_split(path: Path, version: str, split: str) -> ModelSplit:
        split_load_calls.append(split)
        assert version == "v3"
        if split == "train":
            return dataset.train
        assert split == "validation"
        assert (artifact_dir / "stage5_v3_candidate_lock_manifest.json").is_file()
        return dataset.validation

    monkeypatch.setattr(final_module, "load_model_split", load_split)

    payload = run_stage5_finalization(
        output_path=output,
        report_path=report,
        calibration_figure_path=calibration_figure,
        ablation_figure_path=ablation_figure,
        artifact_dir=artifact_dir,
        stage4_reference_path=reference_paths[0],
        lightgbm_reference_path=reference_paths[1],
        mlp_reference_path=reference_paths[2],
        lightgbm_jobs=1,
    )

    assert payload["run_status"] == "complete"
    assert validation_calls == 1
    assert split_load_calls == ["train", "validation"]
    assert payload["data_scope"]["official_validation_prediction_calls"] == 1
    assert payload["data_scope"]["test_feature_rows_used"] == 0
    assert payload["data_scope"]["test_predictions_created"] is False
    assert payload["settings"]["official_validation_used_for_tuning"] is False
    assert payload["settings"]["official_validation_used_for_calibration_fit"] is False
    assert len(payload["inner_development"]["tuning"]["candidates"]) == 3
    ablations = payload["inner_development"]["feature_ablation"]["experiments"]
    assert [item["input_feature_columns"] for item in ablations] == [198, 66, 161, 169]
    assert payload["stage6_candidate"]["test_evaluated"] is False
    assert payload["full_train_refit"][
        "official_validation_uses_reloaded_locked_artifacts"
    ] is True
    assert output.is_file() and report.is_file()
    assert calibration_figure.is_file() and ablation_figure.is_file()
    assert (artifact_dir / "stage5_v3_lightgbm_candidate.joblib").is_file()
    assert (artifact_dir / "stage5_v3_probability_calibrator.joblib").is_file()
    stored = joblib.load(artifact_dir / "stage5_final_validation_scores.joblib")
    assert stored["customer_ids_included"] is False
    assert set(stored["scores"]) == {"raw", "calibrated"}

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for key, filename in (
        ("model", "stage5_v3_lightgbm_candidate.joblib"),
        ("calibrator", "stage5_v3_probability_calibrator.joblib"),
        ("train_oof_scores", "stage5_final_oof_scores.joblib"),
        ("validation_scores", "stage5_final_validation_scores.joblib"),
    ):
        artifact_path = artifact_dir / filename
        assert payload["artifacts"][key]["sha256"] == digest(artifact_path)
        assert payload["artifacts"][key]["bytes"] == artifact_path.stat().st_size

    lock_path = artifact_dir / "stage5_v3_candidate_lock_manifest.json"
    assert payload["artifacts"]["lock_manifest"]["sha256"] == digest(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["model_artifact"]["sha256"] == payload["artifacts"]["model"]["sha256"]
    assert lock["calibrator_artifact"]["sha256"] == payload["artifacts"]["calibrator"]["sha256"]
    assert lock["oof_artifact"]["sha256"] == payload["artifacts"]["train_oof_scores"]["sha256"]

    reloaded_model = joblib.load(artifact_dir / "stage5_v3_lightgbm_candidate.joblib")
    reloaded_calibrator = joblib.load(
        artifact_dir / "stage5_v3_probability_calibrator.joblib"
    )
    replay_raw = original_positive_scores(reloaded_model, dataset.validation.X)
    replay_calibrated = reloaded_calibrator.predict(replay_raw)
    np.testing.assert_allclose(stored["scores"]["raw"], replay_raw, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        stored["scores"]["calibrated"], replay_calibrated, rtol=1e-6, atol=1e-7
    )
    replay_metrics = evaluate_binary_metrics(
        stored["y_true"], stored["scores"]["raw"]
    )
    for metric in ("roc_auc", "pr_auc", "ks", "gini", "brier_score"):
        assert replay_metrics[metric] == pytest.approx(
            payload["official_validation"]["raw_metrics"][metric], abs=1e-6
        )

    forbidden_keys = {"customer_ids", "SK_ID_CURR", "scores", "y_true", "predictions"}

    def assert_aggregate_only(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for nested in value.values():
                assert_aggregate_only(nested)
        elif isinstance(value, list):
            assert len(value) not in {len(dataset.train.y), len(dataset.validation.y)}
            for nested in value:
                assert_aggregate_only(nested)

    assert_aggregate_only(json.loads(output.read_text(encoding="utf-8")))
    shared = output.read_text(encoding="utf-8") + report.read_text(encoding="utf-8")
    assert "SK_ID_CURR" not in shared
    assert "/home/" not in shared
    assert "![피처군 제거 비교](ablation.png)" in shared
    assert "![확률 보정 비교](calibration.png)" in shared
    assert "V3 Random Forest |" in shared
    assert "V3 Random Forest | 1.0000 | 1.0000 | 1.0000 | 비교 제외 |" in shared
    json.loads(output.read_text(encoding="utf-8"))


def test_completed_json_is_not_written_when_report_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    report = tmp_path / "report.md"
    output.write_text('{"run_status": "in_progress"}\n', encoding="utf-8")

    monkeypatch.setattr(
        final_module,
        "render_markdown_report",
        lambda *args, **kwargs: "# complete\n",
    )

    def fail_report_write(path: Path, content: str) -> None:
        raise OSError("simulated report write failure")

    monkeypatch.setattr(final_module, "_atomic_write_text", fail_report_write)

    with pytest.raises(OSError, match="simulated report write failure"):
        _write_completed_outputs(
            output=output,
            report=report,
            payload={"run_status": "complete"},
            calibration_figure=tmp_path / "calibration.png",
            ablation_figure=tmp_path / "ablation.png",
        )

    assert json.loads(output.read_text(encoding="utf-8"))["run_status"] == "in_progress"
