"""Stage 6 1/3 SHAP·Top-K 오류 분석 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.pipeline import Pipeline

import creditlens.analysis.stage6_shap_analysis as stage6_module
from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.calibration import IdentityCalibrator
from creditlens.modeling.data import ModelSplit
from creditlens.modeling.feature_roles import FeatureRoles
from creditlens.modeling.preprocessing import make_preprocessor
from creditlens.analysis.stage6_shap_analysis import (
    _aggregate_source_shap,
    _dense_shap_values,
    _error_analysis,
    _map_transformed_features,
    _ordered_sha256,
    _top_k_mask,
    run_stage6_shap_analysis,
)


def _metadata(path: Path) -> dict[str, object]:
    return {
        "display_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_ignored": None,
    }


def test_transformed_features_map_back_to_original_sources() -> None:
    sources = ("APP_NUM", "BUREAU_NUM", "INST_NUM", "APP_CAT")
    transformed = (
        "numeric__APP_NUM",
        "numeric__missingindicator_APP_NUM",
        "numeric__BUREAU_NUM",
        "numeric__INST_NUM",
        "categorical__APP_CAT_alpha_value",
    )

    mapping, audit = _map_transformed_features(
        transformed,
        sources,
        numeric=sources[:3],
        categorical=("APP_CAT",),
    )

    np.testing.assert_array_equal(mapping, np.asarray([0, 0, 1, 2, 3]))
    assert [item["component_kind"] for item in audit] == [
        "numeric_value",
        "missing_indicator",
        "numeric_value",
        "numeric_value",
        "category_level",
    ]


def test_source_aggregation_preserves_row_shap_sums() -> None:
    values = np.asarray(
        [
            [0.1, 0.2, -0.1, 0.4, 0.5],
            [-0.3, 0.1, 0.2, -0.4, 0.6],
        ]
    )
    mapping = np.asarray([0, 0, 1, 2, 3])

    aggregated = _aggregate_source_shap(values, mapping, source_count=4)

    np.testing.assert_allclose(
        aggregated,
        np.asarray([[0.3, -0.1, 0.4, 0.5], [-0.2, 0.2, -0.4, 0.6]]),
    )
    np.testing.assert_allclose(aggregated.sum(axis=1), values.sum(axis=1))


def test_dense_shap_values_accepts_sparse_explanations() -> None:
    values = sparse.csr_matrix([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)

    dense = _dense_shap_values(values)

    assert dense.dtype == np.float64
    np.testing.assert_allclose(dense, [[0.1, -0.2], [0.3, 0.4]])


def test_top_k_mask_uses_stable_exact_capacity() -> None:
    scores = np.asarray([0.9, 0.8, 0.8, 0.7, 0.1])

    selected, audit = _top_k_mask(scores, 0.4)

    np.testing.assert_array_equal(selected, [True, True, False, False, False])
    assert audit["k"] == 2
    assert audit["cutoff_score"] == 0.8
    assert audit["boundary_tie_count"] == 2
    assert audit["tie_policy_for_group_analysis"] == "stable_row_order_exact_k"


def test_error_analysis_separates_captured_and_missed_positive_cases() -> None:
    scores = np.linspace(1.0, 0.01, 20)
    labels = np.asarray([1, 0] + [1, 0] * 9, dtype=np.int8)
    source_values = np.column_stack((scores - 0.5, 0.5 - scores))
    features = ("APP_NUM", "BUREAU_NUM")
    source_group = {"APP_NUM": "application", "BUREAU_NUM": "bureau"}

    result, masks = _error_analysis(
        labels, scores, source_values, features, source_group
    )

    assert result["review_policy"]["k"] == 2
    assert result["groups"]["captured_positive"]["rows"] == 1
    assert result["groups"]["reviewed_negative"]["rows"] == 1
    assert result["groups"]["missed_positive"]["rows"] == 9
    assert result["groups"]["not_reviewed_negative"]["rows"] == 9
    assert all(mask.dtype == bool for mask in masks.values())


def test_full_runner_uses_only_validation_and_writes_aggregate_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    roles = FeatureRoles(
        version="v3",
        numeric=("APP_NUM", "BUREAU_NUM", "INST_NUM"),
        categorical=("APP_CAT",),
    )

    def frame(rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "APP_NUM": rng.normal(size=rows),
                "BUREAU_NUM": rng.normal(size=rows),
                "INST_NUM": rng.normal(size=rows),
                "APP_CAT": rng.choice(["a", "b", "c"], size=rows),
            }
        ).loc[:, roles.model_features]

    train_X = frame(200)
    train_y = rng.integers(0, 2, size=len(train_X), dtype=np.int8)
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(roles, model_family="tree")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=12,
                    learning_rate=0.1,
                    num_leaves=7,
                    min_child_samples=5,
                    random_state=42,
                    n_jobs=1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    pipeline.fit(train_X, train_y)

    validation_X = frame(200)
    scores = pipeline.predict_proba(validation_X)[:, 1]
    order = np.argsort(-scores, kind="stable")
    validation_y = np.zeros(len(validation_X), dtype=np.int8)
    validation_y[order[0]] = 1
    validation_y[order[20::10]] = 1
    validation_ids = pd.Index(
        np.arange(10_000, 10_000 + len(validation_X)), name="SK_ID_CURR"
    )
    validation_X.index = validation_ids
    validation_target = pd.Series(
        validation_y, index=validation_ids, name="TARGET", dtype="int8"
    )
    validation = ModelSplit(
        "validation", validation_X, validation_target, validation_ids
    )

    model_path = tmp_path / "model.joblib"
    calibrator_path = tmp_path / "calibrator.joblib"
    validation_scores_path = tmp_path / "validation_scores.joblib"
    lock_path = tmp_path / "lock.json"
    joblib.dump(pipeline, model_path)
    joblib.dump(IdentityCalibrator().fit(scores, validation_y), calibrator_path)
    joblib.dump(
        {
            "customer_ids_included": False,
            "y_true": validation_y,
            "scores": {"raw": scores.astype(np.float32)},
        },
        validation_scores_path,
    )
    model_metadata = _metadata(model_path)
    calibrator_metadata = _metadata(calibrator_path)
    lock_path.write_text(
        json.dumps(
            {
                "model_artifact": model_metadata,
                "calibrator_artifact": calibrator_metadata,
                "test_feature_rows_used": 0,
            }
        ),
        encoding="utf-8",
    )
    groups = {
        "application": ("APP_NUM", "APP_CAT"),
        "bureau": ("BUREAU_NUM",),
        "installments": ("INST_NUM",),
    }
    stage5_path = tmp_path / "stage5.json"
    stage5_path.write_text(
        json.dumps(
            {
                "run_status": "complete",
                "run_version": "stage5-test",
                "data_scope": {
                    "validation_rows": len(validation_y),
                    "validation_positive_rate": float(validation_y.mean()),
                    "test_feature_rows_used": 0,
                },
                "stage6_candidate": {
                    "base_model_key": "stage5_selected_lightgbm_v3",
                    "test_evaluated": False,
                },
                "calibration": {"decision": {"selected_method": "identity"}},
                "data_version": {
                    "feature_groups": {
                        name: {
                            "columns": len(columns),
                            "columns_sha256": _ordered_sha256(columns),
                        }
                        for name, columns in groups.items()
                    }
                },
                "official_validation": {
                    "raw_metrics": evaluate_binary_metrics(validation_y, scores)
                },
                "artifacts": {
                    "model": model_metadata,
                    "calibrator": calibrator_metadata,
                    "lock_manifest": _metadata(lock_path),
                    "validation_scores": _metadata(validation_scores_path),
                },
            }
        ),
        encoding="utf-8",
    )
    load_calls: list[str] = []

    def load_split(path: Path, version: str, split: str) -> ModelSplit:
        load_calls.append(split)
        assert version == "v3"
        assert split == "validation"
        return validation

    monkeypatch.setattr(stage6_module, "load_model_split", load_split)
    output = tmp_path / "stage6.json"
    report = tmp_path / "stage6.md"
    global_figure = tmp_path / "global.png"
    direction_figure = tmp_path / "direction.png"
    error_figure = tmp_path / "error.png"
    local_artifact = tmp_path / "models" / "local.joblib"

    payload = run_stage6_shap_analysis(
        stage5_result_path=stage5_path,
        model_path=model_path,
        calibrator_path=calibrator_path,
        lock_manifest_path=lock_path,
        validation_scores_path=validation_scores_path,
        local_artifact_path=local_artifact,
        output_path=output,
        report_path=report,
        global_figure_path=global_figure,
        direction_figure_path=direction_figure,
        error_figure_path=error_figure,
    )

    assert load_calls == ["validation"]
    assert payload["run_status"] == "complete"
    assert payload["data_scope"]["shap_rows"] == len(validation_y)
    assert payload["data_scope"]["test_feature_rows_used"] == 0
    assert payload["model_contract"]["source_feature_columns"] == 4
    assert payload["model_contract"]["identifier_target_split_excluded"] is True
    assert payload["validation_replay"]["stage5_metrics_matched"] is True
    assert payload["validation_replay"]["max_abs_additivity_error"] < 1e-5
    assert output.is_file() and report.is_file()
    assert global_figure.is_file() and direction_figure.is_file()
    assert error_figure.is_file() and local_artifact.is_file()
    local = joblib.load(local_artifact)
    assert local["customer_ids_included"] is False
    assert local["row_positions_included"] is False
    assert len(local["representatives"]) == 4
    shared = output.read_text(encoding="utf-8") + report.read_text(encoding="utf-8")
    assert "SK_ID_CURR" not in shared
    assert "/home/" not in shared
    assert "![전역 SHAP 중요도](global.png)" in shared
    assert "![SHAP 방향](direction.png)" in shared
    assert "![포착·누락 SHAP 차이](error.png)" in shared
