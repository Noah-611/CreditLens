"""Stage 6 2/3 위험전략·하위그룹 분석 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

import creditlens.analysis.stage6_strategy_analysis as strategy_module
from creditlens.analysis.stage4_validation_analysis import build_top_k_scenarios
from creditlens.analysis.stage6_strategy_analysis import (
    _build_risk_bands,
    _define_subgroup_families,
    _enrich_top_k_scenarios,
    _subgroup_metrics,
    run_stage6_strategy_analysis,
)
from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.calibration import IdentityCalibrator
from creditlens.modeling.data import ModelSplit
from creditlens.modeling.feature_roles import FeatureRoles
from creditlens.modeling.preprocessing import make_preprocessor


def _metadata(path: Path) -> dict[str, object]:
    return {
        "display_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_ignored": None,
    }


def test_risk_bands_cover_all_rows_and_use_top10_top30_boundaries() -> None:
    scores = np.linspace(1.0, 0.01, 100)
    labels = np.asarray(
        ([1] * 10) + ([1, 0] * 10) + ([1] * 10) + ([0] * 60),
        dtype=np.int8,
    )
    scenarios = _enrich_top_k_scenarios(
        labels,
        build_top_k_scenarios(
            labels,
            scores,
            fractions=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        ),
    )

    result = _build_risk_bands(labels, scores, scenarios)

    assert [item["rows"] for item in result["bands"]] == [10, 20, 70]
    assert sum(item["positive_count"] for item in result["bands"]) == 30
    assert result["observed_risk_strictly_descends"] is True
    assert result["status"] == "provisional_until_stage6_part3_lock"


def test_subgroup_definitions_assign_every_validation_row() -> None:
    index = pd.Index(range(8), name="SK_ID_CURR")
    features = pd.DataFrame(
        {
            "BUREAU_HAS_HISTORY": [1, 0, 1, 0, 1, 1, 1, 1],
            "INST_HAS_HISTORY": [1, 1, 0, 0, 1, 1, 1, 1],
            "APP_EXT_SOURCE_OBSERVED_COUNT": [0, 1, 2, 3, 1, 2, 3, 3],
            "APP_AGE_YEARS": [25, 35, 55, np.nan, 29, 49, 50, 70],
        },
        index=index,
    )
    gender = pd.Series(["F", "M", "XNA", None, "F", "M", "F", "M"], index=index)

    families = _define_subgroup_families(features, gender)

    assert set(families) == {
        "financial_history_coverage",
        "external_score_coverage",
        "age_band",
        "gender_audit",
    }
    assert families["financial_history_coverage"].value_counts().to_dict() == {
        "both_histories": 5,
        "installments_only": 1,
        "bureau_only": 1,
        "neither_history": 1,
    }
    assert families["external_score_coverage"].value_counts().to_dict() == {
        "three_scores": 3,
        "zero_or_one_score": 3,
        "two_scores": 2,
    }
    assert families["gender_audit"].eq("other_or_unknown").sum() == 2
    assert all(not values.isna().any() for values in families.values())


def test_subgroup_metrics_suppress_tiny_groups_and_flag_reliable_gaps() -> None:
    small_labels = np.asarray([0, 1] * 5, dtype=np.int8)
    small_scores = np.linspace(0.1, 0.9, len(small_labels))
    small = _subgroup_metrics(
        small_labels,
        small_scores,
        np.ones(len(small_labels), dtype=bool),
        global_cutoff=0.5,
        overall={
            "roc_auc": 0.8,
            "brier_score": 0.07,
            "global_cutoff_recall": 0.4,
            "global_cutoff_selection_rate": 0.1,
        },
    )
    assert small["detail_status"] == "suppressed_small_group"
    assert small["positive_count"] is None

    labels = np.asarray(([0] * 1_050) + ([1] * 150), dtype=np.int8)
    scores = np.full(len(labels), 0.2)
    result = _subgroup_metrics(
        labels,
        scores,
        np.ones(len(labels), dtype=bool),
        global_cutoff=0.5,
        overall={
            "roc_auc": 0.8,
            "brier_score": 0.07,
            "global_cutoff_recall": 0.4,
            "global_cutoff_selection_rate": 0.1,
        },
    )
    assert result["reliable_for_alerts"] is True
    assert "roc_auc_drop" in result["diagnostic_alerts"]
    assert "global_cutoff_recall_drop" in result["diagnostic_alerts"]


def test_full_runner_uses_validation_only_and_writes_aggregate_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    roles = FeatureRoles(
        version="v3",
        numeric=(
            "BUREAU_HAS_HISTORY",
            "INST_HAS_HISTORY",
            "APP_EXT_SOURCE_OBSERVED_COUNT",
            "APP_AGE_YEARS",
        ),
        categorical=("APP_CAT",),
    )

    def frame(rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "BUREAU_HAS_HISTORY": rng.integers(0, 2, rows),
                "INST_HAS_HISTORY": rng.integers(0, 2, rows),
                "APP_EXT_SOURCE_OBSERVED_COUNT": rng.integers(0, 4, rows),
                "APP_AGE_YEARS": rng.uniform(21, 70, rows),
                "APP_CAT": rng.choice(["a", "b", "c"], rows),
            }
        ).loc[:, roles.model_features]

    train_X = frame(400)
    train_y = rng.integers(0, 2, len(train_X), dtype=np.int8)
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(roles, model_family="tree")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=20,
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

    validation_X = frame(300)
    validation_ids = pd.Index(range(10_000, 10_300), name="SK_ID_CURR")
    validation_X.index = validation_ids
    scores = pipeline.predict_proba(validation_X)[:, 1]
    order = np.argsort(scores)
    validation_y = np.zeros(len(validation_X), dtype=np.int8)
    validation_y[order[::5]] = 1
    target = pd.Series(validation_y, index=validation_ids, name="TARGET", dtype="int8")
    validation = ModelSplit("validation", validation_X, target, validation_ids)

    model_path = tmp_path / "model.joblib"
    calibrator_path = tmp_path / "calibrator.joblib"
    scores_path = tmp_path / "scores.joblib"
    lock_path = tmp_path / "lock.json"
    mart_path = tmp_path / "mart.parquet"
    mart_path.touch()
    joblib.dump(pipeline, model_path)
    calibrator = IdentityCalibrator().fit(scores, validation_y)
    joblib.dump(calibrator, calibrator_path)
    joblib.dump(
        {
            "y_true": validation_y,
            "scores": {
                "raw": scores.astype(np.float32),
                "calibrated": scores.astype(np.float32),
            },
        },
        scores_path,
    )
    model_metadata = _metadata(model_path)
    calibrator_metadata = _metadata(calibrator_path)
    lock_path.write_text(
        json.dumps(
            {
                "test_feature_rows_used": 0,
                "model_artifact": model_metadata,
                "calibrator_artifact": calibrator_metadata,
            }
        ),
        encoding="utf-8",
    )
    stage5_path = tmp_path / "stage5.json"
    stage5_path.write_text(
        json.dumps(
            {
                "run_status": "complete",
                "run_version": "stage5-test",
                "data_scope": {"test_feature_rows_used": 0},
                "stage6_candidate": {
                    "base_model_key": "stage5_selected_lightgbm_v3",
                    "test_evaluated": False,
                },
                "artifacts": {
                    "model": model_metadata,
                    "calibrator": calibrator_metadata,
                    "lock_manifest": _metadata(lock_path),
                    "validation_scores": _metadata(scores_path),
                },
            }
        ),
        encoding="utf-8",
    )
    stage6_path = tmp_path / "stage6_shap.json"
    stage6_path.write_text(
        json.dumps(
            {
                "run_status": "complete",
                "run_version": "stage6-shap-test",
                "stage_part": "1/3",
                "data_scope": {
                    "test_feature_rows_used": 0,
                    "test_predictions_created": False,
                    "customer_ids_in_shared_outputs": False,
                    "row_level_values_in_shared_outputs": False,
                },
                "settings": {
                    "model_or_preprocessor_refit": False,
                    "operating_cutoff_finalized": False,
                },
                "references": {
                    "stage5_result": {
                        "sha256": hashlib.sha256(stage5_path.read_bytes()).hexdigest()
                    },
                    "model": model_metadata,
                    "calibrator": calibrator_metadata,
                    "lock_manifest": _metadata(lock_path),
                    "validation_scores": _metadata(scores_path),
                },
                "validation_replay": {
                    "metrics": evaluate_binary_metrics(validation_y, scores)
                },
            }
        ),
        encoding="utf-8",
    )
    load_calls: list[str] = []

    def load_split(path: Path, version: str, split: str) -> ModelSplit:
        load_calls.append(split)
        assert path == mart_path and version == "v3" and split == "validation"
        return validation

    monkeypatch.setattr(strategy_module, "load_model_split", load_split)
    gender_calls: list[int] = []

    def load_gender(path: Path, ids: pd.Index) -> pd.Series:
        assert path == mart_path and ids.equals(validation_ids)
        gender_calls.append(len(ids))
        return pd.Series(
            np.where(np.arange(len(ids)) % 2 == 0, "F", "M"), index=ids
        )

    output = tmp_path / "stage6_strategy.json"
    report = tmp_path / "stage6_strategy.md"
    risk_figure = tmp_path / "risk.png"
    topk_figure = tmp_path / "topk.png"
    subgroup_figure = tmp_path / "subgroup.png"
    payload = run_stage6_strategy_analysis(
        stage5_result_path=stage5_path,
        stage6_shap_result_path=stage6_path,
        model_path=model_path,
        calibrator_path=calibrator_path,
        lock_manifest_path=lock_path,
        validation_scores_path=scores_path,
        mart_path=mart_path,
        output_path=output,
        report_path=report,
        risk_figure_path=risk_figure,
        topk_figure_path=topk_figure,
        subgroup_figure_path=subgroup_figure,
        gender_loader=load_gender,
    )

    assert load_calls == ["validation"]
    assert gender_calls == [len(validation_y)]
    assert payload["run_status"] == "complete"
    assert payload["stage_part"] == "2/3"
    assert payload["data_scope"]["test_feature_rows_used"] == 0
    assert payload["model_contract"]["model_or_preprocessor_refit"] is False
    assert payload["diagnostic_summary"]["operating_cutoff_finalized"] is False
    assert payload["stage6_next"]["test_first_use_stage"] == 8
    assert output.is_file() and report.is_file()
    assert risk_figure.is_file() and topk_figure.is_file() and subgroup_figure.is_file()
    shared = output.read_text(encoding="utf-8") + report.read_text(encoding="utf-8")
    assert "SK_ID_CURR" not in shared
    assert "/home/" not in shared
