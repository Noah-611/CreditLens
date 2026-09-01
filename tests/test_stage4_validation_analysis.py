"""Stage 4 validation 상세 분석 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from creditlens.analysis.stage4_validation_analysis import (
    FIGURE_FILENAMES,
    Stage4ValidationAnalysisError,
    _git_artifact_safety,
    build_calibration_summary,
    build_risk_deciles,
    build_top_k_scenarios,
    run_stage4_validation_analysis,
)
from creditlens.evaluation import MetricInputError, evaluate_binary_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixture_scores() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = np.array([0] * 80 + [1] * 20, dtype=np.int8)
    base = np.linspace(0.01, 0.99, num=labels.size, dtype=np.float32)
    return labels, {
        "dummy_prior": np.full(labels.size, labels.mean(), dtype=np.float32),
        "logistic_v1": base,
        "logistic_v2": np.power(base, 1.05).astype(np.float32),
        "logistic_v3": np.power(base, 1.10).astype(np.float32),
        "random_forest_v3": (0.15 + 0.7 * base).astype(np.float32),
    }


def _write_inputs(
    tmp_path: Path,
    *,
    artifact_extra: dict[str, object] | None = None,
    test_feature_rows_used: int = 0,
) -> tuple[Path, Path]:
    labels, scores = _fixture_scores()
    artifact = {
        "schema_version": "1.0",
        "split": "validation",
        "customer_ids_included": False,
        "y_true": labels,
        "scores": scores,
        **(artifact_extra or {}),
    }
    artifact_path = tmp_path / "validation_scores.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    experiment_meta = [
        ("dummy_prior", "Dummy Prior", "v1", "dummy"),
        ("logistic_v1", "V1 Logistic Regression", "v1", "logistic"),
        ("logistic_v2", "V2 Logistic Regression", "v2", "logistic"),
        ("logistic_v3", "V3 Logistic Regression", "v3", "logistic"),
        ("random_forest_v3", "V3 Random Forest", "v3", "random_forest"),
    ]
    baseline = {
        "schema_version": "1.0",
        "run_version": "synthetic-baseline",
        "run_status": "complete",
        "data_scope": {
            "validation_rows": int(labels.size),
            "validation_positive_rate": float(labels.mean()),
            "test_feature_rows_used": test_feature_rows_used,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
        },
        "experiments": [
            {
                "key": key,
                "display_name": name,
                "data_version": version,
                "estimator": estimator,
                "metrics": evaluate_binary_metrics(
                    labels,
                    scores[key],
                    top_fraction=0.1,
                ),
            }
            for key, name, version, estimator in experiment_meta
        ],
        "local_prediction_artifact": {
            "display_path": "models/stage4/validation_scores.joblib",
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
            "git_ignored": True,
        },
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(baseline, allow_nan=False),
        encoding="utf-8",
    )
    return baseline_path, artifact_path


def test_calibration_all_equal_scores_form_one_bin() -> None:
    result = build_calibration_summary(
        [0, 1, 0, 1],
        [0.25, 0.25, 0.25, 0.25],
        n_bins=2,
    )

    assert result["requested_bin_count"] == 2
    assert result["effective_bin_count"] == 1
    assert result["bins"][0]["customer_count"] == 4
    assert result["bins"][0]["observed_positive_rate"] == pytest.approx(0.5)
    assert result["expected_calibration_error"] == pytest.approx(0.25)


def test_calibration_equal_score_groups_are_shuffle_invariant() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.9])
    order = np.array([7, 2, 5, 0, 4, 1, 6, 3])

    original = build_calibration_summary(labels, scores, n_bins=4)
    shuffled = build_calibration_summary(labels[order], scores[order], n_bins=4)

    assert original == shuffled
    assert sum(item["customer_count"] for item in original["bins"]) == 8
    for score in np.unique(scores):
        containing_bins = [
            item
            for item in original["bins"]
            if item["score_min"] <= score <= item["score_max"]
        ]
        assert len(containing_bins) == 1


def test_fractional_deciles_preserve_totals_and_equal_score_neutrality() -> None:
    labels = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    scores = np.full(labels.size, 0.2)

    result = build_risk_deciles(labels, scores, n_bands=10)
    bands = result["bands"]

    assert [item["cumulative_customer_count"] for item in bands] == list(
        range(2, 12)
    )
    assert sum(item["customer_count"] for item in bands) == 11
    assert sum(item["positive_weight"] for item in bands) == pytest.approx(2.0)
    assert all(item["observed_positive_rate"] == pytest.approx(2 / 11) for item in bands)
    assert all(item["lift"] == pytest.approx(1.0) for item in bands)
    assert bands[-1]["cumulative_recall"] == pytest.approx(1.0)


def test_fractional_deciles_are_order_invariant_at_ties() -> None:
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1])
    order = np.array([10, 3, 4, 0, 7, 6, 1, 9, 5, 2, 8])

    original = build_risk_deciles(labels, scores, n_bands=10)
    shuffled = build_risk_deciles(labels[order], scores[order], n_bands=10)

    assert original == shuffled


def test_top_k_scenarios_reuse_common_metric_contract() -> None:
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    scenarios = build_top_k_scenarios(labels, scores, fractions=(0.2, 0.5, 1.0))

    for scenario in scenarios:
        expected = evaluate_binary_metrics(
            labels,
            scores,
            top_fraction=scenario["review_fraction"],
        )["top_k_metrics"]
        assert scenario["selected_customer_count"] == expected["k"]
        assert scenario["true_positive_weight"] == pytest.approx(
            expected["true_positive_weight"]
        )
        assert scenario["precision"] == pytest.approx(expected["precision"])
        assert scenario["recall"] == pytest.approx(expected["recall"])
        assert scenario["lift"] == pytest.approx(expected["lift"])
    assert scenarios[-1]["recall"] == pytest.approx(1.0)
    assert scenarios[-1]["lift"] == pytest.approx(1.0)


def test_runner_creates_only_aggregate_outputs(tmp_path: Path) -> None:
    baseline_path, artifact_path = _write_inputs(tmp_path)
    output_path = tmp_path / "analysis.json"
    report_path = tmp_path / "analysis.md"
    figures_dir = tmp_path / "figures"

    result = run_stage4_validation_analysis(
        baseline_result_path=baseline_path,
        score_artifact_path=artifact_path,
        output_path=output_path,
        report_path=report_path,
        figures_dir=figures_dir,
    )

    assert result["run_status"] == "complete"
    assert result["analysis_scope"]["test_feature_rows_used"] == 0
    assert result["analysis_scope"]["test_predictions_created"] is False
    assert result["analysis_scope"]["customer_ids_in_input"] is False
    assert len(result["models"]) == 5
    assert len(result["figures"]) == len(FIGURE_FILENAMES)
    assert output_path.is_file()
    assert report_path.is_file()
    assert all((figures_dir / name).stat().st_size > 0 for name in FIGURE_FILENAMES)
    for figure in result["figures"]:
        relative = Path(figure["report_relative_path"])
        assert not relative.is_absolute()
        assert (report_path.parent / relative).resolve().is_file()

    shared = output_path.read_text(encoding="utf-8") + report_path.read_text(
        encoding="utf-8"
    )
    assert "SK_ID_CURR" not in shared
    assert "y_true" not in shared
    assert str(tmp_path) not in shared
    assert "false_positive_rate" not in shared
    json.loads(output_path.read_text(encoding="utf-8"))


def test_runner_rejects_test_usage_before_outputs(tmp_path: Path) -> None:
    baseline_path, artifact_path = _write_inputs(
        tmp_path,
        test_feature_rows_used=1,
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(Stage4ValidationAnalysisError, match="test 피처"):
        run_stage4_validation_analysis(
            baseline_result_path=baseline_path,
            score_artifact_path=artifact_path,
            output_path=output_path,
            report_path=tmp_path / "analysis.md",
            figures_dir=tmp_path / "figures",
        )

    assert not output_path.exists()
    assert not (tmp_path / "figures").exists()


def test_runner_rejects_customer_id_field_before_outputs(tmp_path: Path) -> None:
    baseline_path, artifact_path = _write_inputs(
        tmp_path,
        artifact_extra={"customer_ids": np.arange(100)},
    )

    with pytest.raises(Stage4ValidationAnalysisError, match="추가 필드"):
        run_stage4_validation_analysis(
            baseline_result_path=baseline_path,
            score_artifact_path=artifact_path,
            output_path=tmp_path / "analysis.json",
            report_path=tmp_path / "analysis.md",
            figures_dir=tmp_path / "figures",
        )

    assert not (tmp_path / "analysis.json").exists()
    assert not (tmp_path / "figures").exists()


def test_runner_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    baseline_path, artifact_path = _write_inputs(tmp_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["local_prediction_artifact"]["sha256"] = "0" * 64
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(Stage4ValidationAnalysisError, match="SHA-256"):
        run_stage4_validation_analysis(
            baseline_result_path=baseline_path,
            score_artifact_path=artifact_path,
            output_path=tmp_path / "analysis.json",
            report_path=tmp_path / "analysis.md",
            figures_dir=tmp_path / "figures",
        )

    assert not (tmp_path / "analysis.json").exists()


def test_runner_rejects_output_that_overwrites_input(tmp_path: Path) -> None:
    baseline_path, artifact_path = _write_inputs(tmp_path)

    with pytest.raises(Stage4ValidationAnalysisError, match="경로와 겹칩니다"):
        run_stage4_validation_analysis(
            baseline_result_path=baseline_path,
            score_artifact_path=artifact_path,
            output_path=baseline_path,
            report_path=tmp_path / "analysis.md",
            figures_dir=tmp_path / "figures",
        )

    assert (
        json.loads(baseline_path.read_text(encoding="utf-8"))["run_status"]
        == "complete"
    )


def test_git_safety_rejects_tracked_artifact() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    with pytest.raises(Stage4ValidationAnalysisError, match="Git에서 추적"):
        _git_artifact_safety(readme)


def test_runner_rejects_unexpected_model_contract(tmp_path: Path) -> None:
    baseline_path, artifact_path = _write_inputs(tmp_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["experiments"] = baseline["experiments"][:-1]
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(Stage4ValidationAnalysisError, match="실험 순서"):
        run_stage4_validation_analysis(
            baseline_result_path=baseline_path,
            score_artifact_path=artifact_path,
            output_path=tmp_path / "analysis.json",
            report_path=tmp_path / "analysis.md",
            figures_dir=tmp_path / "figures",
        )


def test_runner_rejects_unknown_baseline_schema(tmp_path: Path) -> None:
    baseline_path, artifact_path = _write_inputs(tmp_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["schema_version"] = "2.0"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(Stage4ValidationAnalysisError, match="schema_version"):
        run_stage4_validation_analysis(
            baseline_result_path=baseline_path,
            score_artifact_path=artifact_path,
            output_path=tmp_path / "analysis.json",
            report_path=tmp_path / "analysis.md",
            figures_dir=tmp_path / "figures",
        )


def test_runner_rejects_unknown_score_schema(tmp_path: Path) -> None:
    baseline_path, artifact_path = _write_inputs(
        tmp_path,
        artifact_extra={"schema_version": "2.0"},
    )

    with pytest.raises(Stage4ValidationAnalysisError, match="schema_version"):
        run_stage4_validation_analysis(
            baseline_result_path=baseline_path,
            score_artifact_path=artifact_path,
            output_path=tmp_path / "analysis.json",
            report_path=tmp_path / "analysis.md",
            figures_dir=tmp_path / "figures",
        )


@pytest.mark.parametrize(
    "fractions",
    [(), (0.0,), (1.1,), (True,), (0.1, 0.1)],
)
def test_invalid_top_k_scenarios_raise(fractions: tuple[object, ...]) -> None:
    with pytest.raises(MetricInputError, match="Top-K"):
        build_top_k_scenarios(
            [0, 1],
            [0.1, 0.9],
            fractions=fractions,  # type: ignore[arg-type]
        )
