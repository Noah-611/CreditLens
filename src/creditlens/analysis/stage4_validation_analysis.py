"""Stage 4 기준 모델의 validation 예측점수를 상세 진단한다.

이 모듈은 Git에서 제외된 ``validation_scores.joblib``과 고객 ID가 없는 기준
모델 집계 JSON만 읽는다. 원본 마트·학습 모델·test 데이터는 읽지 않으며,
공유 산출물에는 행별 정답·예측점수와 원시 ROC·PR 좌표를 기록하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "creditlens-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve

from creditlens.evaluation import evaluate_binary_metrics
from creditlens.evaluation.metrics import MetricInputError, _as_binary_arrays


SCHEMA_VERSION = "1.0"
RUN_VERSION = "stage4-validation-analysis-v1"
DEFAULT_BASELINE_RESULT = Path("reports/stage4_baseline_results.json")
DEFAULT_SCORE_ARTIFACT = Path("models/stage4/validation_scores.joblib")
DEFAULT_OUTPUT = Path("reports/stage4_validation_analysis.json")
DEFAULT_REPORT = Path("docs/Stage4_Validation_Analysis_Report.md")
DEFAULT_FIGURES_DIR = Path("reports/figures")
CALIBRATION_BINS = 10
RISK_BANDS = 10
TOP_FRACTIONS = (0.05, 0.10, 0.20)
METRIC_TOLERANCE = 1e-6
EXPECTED_MODEL_KEYS = (
    "dummy_prior",
    "logistic_v1",
    "logistic_v2",
    "logistic_v3",
    "random_forest_v3",
)
FIGURE_FILENAMES = (
    "stage4_roc_curve.png",
    "stage4_pr_curve.png",
    "stage4_calibration_curve.png",
    "stage4_risk_deciles.png",
    "stage4_topk_scenarios.png",
)
MODEL_COLORS = {
    "dummy_prior": "#777777",
    "logistic_v1": "#4C78A8",
    "logistic_v2": "#72B7B2",
    "logistic_v3": "#F58518",
    "random_forest_v3": "#E45756",
}


class Stage4ValidationAnalysisError(ValueError):
    """Stage 4 validation 분석 계약이 깨졌을 때 발생한다."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _git_artifact_safety(path: Path) -> dict[str, Any]:
    """현재 저장소 안의 로컬 산출물이 미추적·Git 제외 상태인지 확인한다."""

    resolved_path = path.resolve()
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=resolved_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        return {
            "inside_current_repository": False,
            "git_tracked": None,
            "git_ignored": None,
        }
    repository_root = Path(root_result.stdout.strip()).resolve()
    try:
        relative = resolved_path.relative_to(repository_root)
    except ValueError:
        return {
            "inside_current_repository": False,
            "git_tracked": False,
            "git_ignored": None,
        }

    relative_text = relative.as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_text],
        cwd=repository_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    ignored_result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative_text],
        cwd=repository_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if tracked:
        raise Stage4ValidationAnalysisError(
            "validation 점수 파일이 Git에서 추적되고 있습니다."
        )
    if ignored_result.returncode != 0:
        raise Stage4ValidationAnalysisError(
            "저장소 안의 validation 점수 파일이 Git 제외 상태가 아닙니다."
        )
    return {
        "inside_current_repository": True,
        "git_tracked": False,
        "git_ignored": True,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    _atomic_write_text(path, serialized + "\n")


def _atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        figure.savefig(temporary, dpi=150, bbox_inches="tight", format="png")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def _validate_paths(
    *,
    baseline_path: Path,
    scores_path: Path,
    output_path: Path,
    report_path: Path,
    figures_dir: Path,
) -> None:
    if output_path.suffix.lower() != ".json":
        raise Stage4ValidationAnalysisError("분석 집계 출력은 JSON 파일이어야 합니다.")
    if report_path.suffix.lower() != ".md":
        raise Stage4ValidationAnalysisError("분석 보고서 출력은 Markdown 파일이어야 합니다.")

    named_paths = {
        "기준 모델 결과": baseline_path,
        "validation 점수": scores_path,
        "분석 집계 출력": output_path,
        "분석 보고서 출력": report_path,
        **{
            f"그림 출력 {filename}": figures_dir / filename
            for filename in FIGURE_FILENAMES
        },
    }
    seen: dict[Path, str] = {}
    for name, path in named_paths.items():
        resolved = path.resolve()
        if resolved in seen:
            raise Stage4ValidationAnalysisError(
                f"{name}이(가) {seen[resolved]} 경로와 겹칩니다."
            )
        seen[resolved] = name


def _validate_group_count(value: int, *, name: str, sample_count: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise MetricInputError(f"{name}은 정수여야 합니다.")
    count = int(value)
    if count < 2 or count > sample_count:
        raise MetricInputError(f"{name}은 2 이상 표본 수 이하여야 합니다.")
    return count


def build_calibration_summary(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    n_bins: int = CALIBRATION_BINS,
) -> dict[str, Any]:
    """같은 점수를 쪼개지 않는 quantile calibration 진단을 만든다."""

    labels, scores = _as_binary_arrays(y_true, y_score)
    requested_bins = _validate_group_count(
        n_bins,
        name="n_bins",
        sample_count=labels.size,
    )
    quantiles = np.linspace(0.0, 1.0, requested_bins + 1)
    unique_edges = np.unique(np.quantile(scores, quantiles))
    if unique_edges.size == 1:
        bin_ids = np.zeros(labels.size, dtype=np.int64)
    else:
        bin_ids = np.searchsorted(unique_edges[1:-1], scores, side="right")

    bins: list[dict[str, Any]] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for raw_bin in np.unique(bin_ids):
        mask = bin_ids == raw_bin
        count = int(np.count_nonzero(mask))
        positives = int(labels[mask].sum(dtype=np.int64))
        mean_score = float(scores[mask].mean())
        observed_rate = positives / count
        absolute_gap = abs(observed_rate - mean_score)
        weighted_gap += count / labels.size * absolute_gap
        maximum_gap = max(maximum_gap, absolute_gap)
        bins.append(
            {
                "bin": len(bins) + 1,
                "direction": "low_to_high_predicted_probability",
                "customer_count": count,
                "positive_count": positives,
                "score_min": float(scores[mask].min()),
                "score_max": float(scores[mask].max()),
                "mean_predicted_probability": mean_score,
                "observed_positive_rate": observed_rate,
                "absolute_gap": absolute_gap,
            }
        )

    prevalence = float(labels.mean())
    mean_score = float(scores.mean())
    return {
        "strategy": "quantile_without_splitting_equal_scores",
        "requested_bin_count": requested_bins,
        "effective_bin_count": len(bins),
        "sample_count": int(labels.size),
        "prevalence": prevalence,
        "mean_predicted_probability": mean_score,
        "mean_probability_bias": mean_score - prevalence,
        "expected_calibration_error": weighted_gap,
        "maximum_calibration_error": maximum_gap,
        "brier_score": float(np.mean(np.square(scores - labels))),
        "bins": bins,
    }


def _fractional_selected_score_sum(
    scores: np.ndarray,
    *,
    cutoff_score: float,
    boundary_weight: float,
) -> float:
    above = scores > cutoff_score
    at = scores == cutoff_score
    return float(
        scores[above].sum(dtype=np.float64)
        + boundary_weight * scores[at].sum(dtype=np.float64)
    )


def build_risk_deciles(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    n_bands: int = RISK_BANDS,
) -> dict[str, Any]:
    """누적 fractional Top-K 차이로 고위험부터 위험구간을 만든다."""

    labels, scores = _as_binary_arrays(y_true, y_score)
    bands = _validate_group_count(
        n_bands,
        name="n_bands",
        sample_count=labels.size,
    )
    positive_count = int(labels.sum(dtype=np.int64))
    negative_count = int(labels.size - positive_count)
    if not positive_count or not negative_count:
        raise MetricInputError("위험구간 분석에는 TARGET 두 클래스가 모두 필요합니다.")

    prevalence = positive_count / labels.size
    previous_k = 0
    previous_positive_weight = 0.0
    previous_score_sum = 0.0
    previous_cutoff = float(scores.max())
    result_bands: list[dict[str, Any]] = []

    for band_number in range(1, bands + 1):
        cumulative_k = math.ceil(labels.size * band_number / bands)
        top_k = evaluate_binary_metrics(
            labels,
            scores,
            top_fraction=None,
            top_k=cumulative_k,
        )["top_k_metrics"]
        cutoff = float(top_k["cutoff_score"])
        boundary_weight = float(top_k["boundary_selected_weight"])
        cumulative_score_sum = _fractional_selected_score_sum(
            scores,
            cutoff_score=cutoff,
            boundary_weight=boundary_weight,
        )
        cumulative_positive_weight = float(top_k["true_positive_weight"])
        customer_count = cumulative_k - previous_k
        positive_weight = cumulative_positive_weight - previous_positive_weight
        score_sum = cumulative_score_sum - previous_score_sum
        observed_rate = positive_weight / customer_count

        result_bands.append(
            {
                "decile": band_number,
                "direction": "1_is_highest_risk",
                "customer_count": customer_count,
                "population_fraction": customer_count / labels.size,
                "positive_weight": positive_weight,
                "negative_weight": customer_count - positive_weight,
                "score_upper_bound": previous_cutoff,
                "score_lower_bound": cutoff,
                "mean_predicted_probability": score_sum / customer_count,
                "observed_positive_rate": observed_rate,
                "lift": observed_rate / prevalence,
                "cumulative_customer_count": cumulative_k,
                "cumulative_population_fraction": cumulative_k / labels.size,
                "cumulative_positive_weight": cumulative_positive_weight,
                "cumulative_recall": cumulative_positive_weight / positive_count,
                "boundary_tie_count": int(top_k["boundary_tie_count"]),
                "boundary_selected_weight": boundary_weight,
            }
        )
        previous_k = cumulative_k
        previous_positive_weight = cumulative_positive_weight
        previous_score_sum = cumulative_score_sum
        previous_cutoff = cutoff

    return {
        "method": "cumulative_fractional_top_k_differences",
        "band_count": bands,
        "direction": "decile_1_is_highest_risk",
        "sample_count": int(labels.size),
        "positive_count": positive_count,
        "prevalence": prevalence,
        "bands": result_bands,
    }


def build_top_k_scenarios(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    fractions: Sequence[Real] = TOP_FRACTIONS,
) -> list[dict[str, Any]]:
    """기존 공통 평가 계약으로 여러 우선검토 용량을 계산한다."""

    if not fractions:
        raise MetricInputError("Top-K 비율은 하나 이상 필요합니다.")
    resolved: list[float] = []
    for value in fractions:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise MetricInputError("Top-K 비율은 수치여야 합니다.")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0 or numeric > 1.0:
            raise MetricInputError("Top-K 비율은 (0, 1] 범위여야 합니다.")
        if numeric in resolved:
            raise MetricInputError("Top-K 비율은 중복될 수 없습니다.")
        resolved.append(numeric)

    scenarios: list[dict[str, Any]] = []
    for fraction in sorted(resolved):
        top_k = evaluate_binary_metrics(
            y_true,
            y_score,
            top_fraction=fraction,
        )["top_k_metrics"]
        scenarios.append(
            {
                "review_fraction": fraction,
                "selected_customer_count": int(top_k["k"]),
                "actual_review_fraction": float(top_k["actual_fraction"]),
                "observed_cutoff_score": float(top_k["cutoff_score"]),
                "boundary_tie_count": int(top_k["boundary_tie_count"]),
                "boundary_selected_weight": float(
                    top_k["boundary_selected_weight"]
                ),
                "true_positive_weight": float(top_k["true_positive_weight"]),
                "precision": float(top_k["precision"]),
                "recall": None
                if top_k["recall"] is None
                else float(top_k["recall"]),
                "lift": None
                if top_k["lift"] is None
                else float(top_k["lift"]),
            }
        )
    return scenarios


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage4ValidationAnalysisError(f"입력 파일을 찾을 수 없습니다: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage4ValidationAnalysisError(
            f"기준 모델 JSON을 읽을 수 없습니다: {path.name}"
        ) from error
    if not isinstance(payload, dict):
        raise Stage4ValidationAnalysisError("기준 모델 결과는 JSON 객체여야 합니다.")
    return payload


def _validate_baseline_result(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise Stage4ValidationAnalysisError("지원하지 않는 기준 모델 schema_version입니다.")
    if payload.get("run_status") != "complete":
        raise Stage4ValidationAnalysisError("완료된 기준 모델 결과가 아닙니다.")
    scope = payload.get("data_scope")
    if not isinstance(scope, Mapping):
        raise Stage4ValidationAnalysisError("기준 모델 데이터 사용 감사가 없습니다.")
    if scope.get("test_feature_rows_used") != 0:
        raise Stage4ValidationAnalysisError("기준 모델 실행에서 test 피처가 사용됐습니다.")
    if scope.get("test_predictions_created") is not False:
        raise Stage4ValidationAnalysisError("기준 모델 실행에서 test 예측이 생성됐습니다.")
    if scope.get("customer_ids_in_shared_outputs") is not False:
        raise Stage4ValidationAnalysisError("공유 기준 모델 결과에 고객 ID가 포함됐습니다.")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise Stage4ValidationAnalysisError("기준 모델 실험 목록이 없습니다.")
    keys = [item.get("key") for item in experiments if isinstance(item, Mapping)]
    if keys != list(EXPECTED_MODEL_KEYS):
        raise Stage4ValidationAnalysisError(
            "기준 모델 실험 순서가 Stage 4 계약과 다릅니다."
        )
    return [dict(item) for item in experiments]


def _load_and_validate_scores(
    artifact_path: Path,
    *,
    baseline: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, str],
    dict[str, Any],
]:
    if not artifact_path.is_file():
        raise Stage4ValidationAnalysisError(
            f"validation 점수 파일을 찾을 수 없습니다: {artifact_path.name}"
        )
    artifact_meta = baseline.get("local_prediction_artifact")
    if not isinstance(artifact_meta, Mapping):
        raise Stage4ValidationAnalysisError("validation 점수 파일 메타데이터가 없습니다.")
    if artifact_meta.get("git_ignored") is not True:
        raise Stage4ValidationAnalysisError("validation 점수 파일의 Git 제외를 확인할 수 없습니다.")
    if artifact_path.stat().st_size != artifact_meta.get("bytes"):
        raise Stage4ValidationAnalysisError("validation 점수 파일 크기가 기준 기록과 다릅니다.")
    if _sha256(artifact_path) != artifact_meta.get("sha256"):
        raise Stage4ValidationAnalysisError("validation 점수 파일 SHA-256이 기준 기록과 다릅니다.")
    git_safety = _git_artifact_safety(artifact_path)

    try:
        artifact = joblib.load(artifact_path)
    except Exception as error:
        raise Stage4ValidationAnalysisError(
            "validation 점수 파일을 읽을 수 없습니다."
        ) from error
    if not isinstance(artifact, Mapping):
        raise Stage4ValidationAnalysisError("validation 점수 산출물은 객체여야 합니다.")
    expected_fields = {
        "schema_version",
        "split",
        "customer_ids_included",
        "y_true",
        "scores",
    }
    if set(artifact) != expected_fields:
        raise Stage4ValidationAnalysisError(
            "validation 점수 산출물 필드가 계약과 다릅니다. 고객 ID 등 추가 필드는 허용하지 않습니다."
        )
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise Stage4ValidationAnalysisError(
            "지원하지 않는 validation 점수 schema_version입니다."
        )
    if artifact.get("split") != "validation":
        raise Stage4ValidationAnalysisError("validation 점수만 분석할 수 있습니다.")
    if artifact.get("customer_ids_included") is not False:
        raise Stage4ValidationAnalysisError("validation 점수에 고객 ID를 포함할 수 없습니다.")
    raw_scores = artifact.get("scores")
    if not isinstance(raw_scores, Mapping):
        raise Stage4ValidationAnalysisError("모델별 validation 점수가 없습니다.")

    expected_keys = [str(item["key"]) for item in experiments]
    if set(raw_scores) != set(expected_keys):
        raise Stage4ValidationAnalysisError("validation 점수의 모델 key가 실험 목록과 다릅니다.")

    labels_raw = np.asarray(artifact.get("y_true"))
    scores: dict[str, np.ndarray] = {}
    metric_deltas: dict[str, float] = {}
    artifact_score_dtypes: dict[str, str] = {}
    validated_labels: np.ndarray | None = None
    for experiment in experiments:
        key = str(experiment["key"])
        artifact_score_dtypes[key] = str(np.asarray(raw_scores[key]).dtype)
        try:
            labels, model_scores = _as_binary_arrays(labels_raw, raw_scores[key])
        except MetricInputError as error:
            raise Stage4ValidationAnalysisError(
                f"{key} validation 점수 계약이 깨졌습니다: {error}"
            ) from error
        if validated_labels is None:
            validated_labels = labels
        scores[key] = model_scores
        recomputed = evaluate_binary_metrics(labels, model_scores, top_fraction=0.1)
        recorded = experiment.get("metrics")
        if not isinstance(recorded, Mapping):
            raise Stage4ValidationAnalysisError(f"{key} 기준 지표가 없습니다.")
        for metric in ("roc_auc", "pr_auc", "ks", "gini", "brier_score"):
            expected = recorded.get(metric)
            actual = recomputed.get(metric)
            if expected is None or actual is None:
                raise Stage4ValidationAnalysisError(f"{key} {metric}을 비교할 수 없습니다.")
            delta = abs(float(actual) - float(expected))
            metric_deltas[f"{key}.{metric}"] = delta
            if delta > METRIC_TOLERANCE:
                raise Stage4ValidationAnalysisError(
                    f"{key} {metric}이 기준 결과와 일치하지 않습니다."
                )

    if validated_labels is None:
        raise Stage4ValidationAnalysisError("validation 정답이 없습니다.")
    positive_count = int(validated_labels.sum(dtype=np.int64))
    if not positive_count or positive_count == validated_labels.size:
        raise Stage4ValidationAnalysisError("validation에는 TARGET 두 클래스가 필요합니다.")
    scope = baseline["data_scope"]
    if validated_labels.size != scope.get("validation_rows"):
        raise Stage4ValidationAnalysisError("validation 행 수가 기준 결과와 다릅니다.")
    if not math.isclose(
        float(validated_labels.mean()),
        float(scope.get("validation_positive_rate")),
        rel_tol=0.0,
        abs_tol=METRIC_TOLERANCE,
    ):
        raise Stage4ValidationAnalysisError("validation 양성 비율이 기준 결과와 다릅니다.")
    return (
        validated_labels,
        scores,
        metric_deltas,
        artifact_score_dtypes,
        git_safety,
    )


def _curve_coordinates(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, np.ndarray]:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "precision": precision,
        "recall": recall,
    }


def _model_lookup(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["key"]): item for item in result["models"]}


def _scenario_by_fraction(
    model: Mapping[str, Any],
    fraction: float,
) -> Mapping[str, Any]:
    return next(
        item
        for item in model["top_k_scenarios"]
        if math.isclose(float(item["review_fraction"]), fraction)
    )


def _build_summary(models: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best_roc = max(models, key=lambda item: float(item["performance"]["roc_auc"]))
    best_pr = max(models, key=lambda item: float(item["performance"]["pr_auc"]))
    best_ks = max(models, key=lambda item: float(item["performance"]["ks"]))
    best_brier = min(
        models,
        key=lambda item: float(item["performance"]["brier_score"]),
    )
    by_key = {str(item["key"]): item for item in models}
    logistic = by_key["logistic_v3"]
    forest = by_key["random_forest_v3"]

    top_k_best: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        winner = max(
            models,
            key=lambda item: float(_scenario_by_fraction(item, fraction)["lift"]),
        )
        scenario = _scenario_by_fraction(winner, fraction)
        top_k_best.append(
            {
                "review_fraction": fraction,
                "model_key": winner["key"],
                "display_name": winner["display_name"],
                "recall": scenario["recall"],
                "precision": scenario["precision"],
                "lift": scenario["lift"],
            }
        )

    return {
        "best_point_estimates": {
            "roc_auc": {"model_key": best_roc["key"], "value": best_roc["performance"]["roc_auc"]},
            "pr_auc": {"model_key": best_pr["key"], "value": best_pr["performance"]["pr_auc"]},
            "ks": {"model_key": best_ks["key"], "value": best_ks["performance"]["ks"]},
            "brier_score": {"model_key": best_brier["key"], "value": best_brier["performance"]["brier_score"]},
        },
        "v3_logistic_minus_random_forest": {
            metric: float(logistic["performance"][metric])
            - float(forest["performance"][metric])
            for metric in ("roc_auc", "pr_auc", "ks", "gini", "brier_score")
        },
        "top_k_best_point_estimates": top_k_best,
        "stage4_reference_baseline": "logistic_v3",
        "reference_is_final_model": False,
        "probability_calibrator_fitted": False,
        "operating_cutoff_selected": False,
        "comparison_interpretation": "same_validation_point_estimates_without_significance_test",
        "next_stage": "LightGBM and TensorFlow MLP comparison with train-only tuning and calibration design",
    }


def _plot_roc(
    models: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for model in models:
        key = str(model["key"])
        curve = curves[key]
        axis.plot(
            curve["false_positive_rate"],
            curve["true_positive_rate"],
            color=MODEL_COLORS.get(key),
            linewidth=2,
            label=f"{model['display_name']} (AUC={model['performance']['roc_auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6, label="Random")
    axis.set(title="Validation ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    _atomic_save_figure(figure, output_path)


def _plot_pr(
    models: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, np.ndarray]],
    *,
    prevalence: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for model in models:
        key = str(model["key"])
        curve = curves[key]
        axis.step(
            curve["recall"],
            curve["precision"],
            where="post",
            color=MODEL_COLORS.get(key),
            linewidth=2,
            label=f"{model['display_name']} (AP={model['performance']['pr_auc']:.3f})",
        )
    axis.axhline(prevalence, linestyle="--", color="black", alpha=0.6, label="Prevalence")
    axis.set(title="Validation precision-recall curves", xlabel="Recall", ylabel="Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    _atomic_save_figure(figure, output_path)


def _plot_calibration(
    models: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for model in models:
        key = str(model["key"])
        bins = model["calibration"]["bins"]
        axis.plot(
            [item["mean_predicted_probability"] for item in bins],
            [item["observed_positive_rate"] for item in bins],
            marker="o",
            markersize=4,
            color=MODEL_COLORS.get(key),
            linewidth=1.8,
            label=f"{model['display_name']} (ECE={model['calibration']['expected_calibration_error']:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6, label="Ideal")
    axis.set(
        title="Validation calibration diagnostics (quantile bins)",
        xlabel="Mean predicted probability",
        ylabel="Observed difficulty rate",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    _atomic_save_figure(figure, output_path)


def _plot_deciles(
    models: Sequence[Mapping[str, Any]],
    *,
    prevalence: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    for model in models:
        if model["key"] == "dummy_prior":
            continue
        key = str(model["key"])
        bands = model["risk_deciles"]["bands"]
        axis.plot(
            [item["decile"] for item in bands],
            [item["observed_positive_rate"] for item in bands],
            marker="o",
            color=MODEL_COLORS.get(key),
            linewidth=2,
            label=model["display_name"],
        )
    axis.axhline(prevalence, linestyle="--", color="black", alpha=0.6, label="Overall prevalence")
    axis.set(
        title="Observed difficulty rate by validation risk decile",
        xlabel="Risk decile (1 = highest risk)",
        ylabel="Observed difficulty rate",
        xticks=range(1, RISK_BANDS + 1),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _atomic_save_figure(figure, output_path)


def _plot_top_k(
    models: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for model in models:
        key = str(model["key"])
        scenarios = model["top_k_scenarios"]
        fractions = [item["review_fraction"] * 100 for item in scenarios]
        axes[0].plot(
            fractions,
            [item["recall"] for item in scenarios],
            marker="o",
            color=MODEL_COLORS.get(key),
            linewidth=2,
            label=model["display_name"],
        )
        axes[1].plot(
            fractions,
            [item["lift"] for item in scenarios],
            marker="o",
            color=MODEL_COLORS.get(key),
            linewidth=2,
            label=model["display_name"],
        )
    axes[0].set(title="Recall by review capacity", xlabel="Reviewed customers (%)", ylabel="Recall")
    axes[1].set(title="Lift by review capacity", xlabel="Reviewed customers (%)", ylabel="Lift")
    axes[1].axhline(1.0, linestyle="--", color="black", alpha=0.6)
    for axis in axes:
        axis.set_xticks([value * 100 for value in TOP_FRACTIONS])
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    _atomic_save_figure(figure, output_path)


def _write_figures(
    result: Mapping[str, Any],
    curves: Mapping[str, Mapping[str, np.ndarray]],
    figures_dir: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    models = result["models"]
    prevalence = float(result["analysis_scope"]["validation_positive_rate"])
    paths = [figures_dir / filename for filename in FIGURE_FILENAMES]
    _plot_roc(models, curves, paths[0])
    _plot_pr(models, curves, prevalence=prevalence, output_path=paths[1])
    _plot_calibration(models, paths[2])
    _plot_deciles(models, prevalence=prevalence, output_path=paths[3])
    _plot_top_k(models, paths[4])
    return [
        {
            "display_path": _display_path(path),
            "report_relative_path": Path(
                os.path.relpath(path.resolve(), start=report_path.parent.resolve())
            ).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def render_markdown_report(result: Mapping[str, Any]) -> str:
    """집계 JSON과 같은 내용을 한국어 Markdown 보고서로 변환한다."""

    models = result["models"]
    by_key = _model_lookup(result)
    scope = result["analysis_scope"]
    reference = by_key[result["summary"]["stage4_reference_baseline"]]
    logistic = by_key["logistic_v3"]
    forest = by_key["random_forest_v3"]
    logistic_top10 = _scenario_by_fraction(logistic, 0.10)
    forest_top10 = _scenario_by_fraction(forest, 0.10)
    logistic_deciles = logistic["risk_deciles"]["bands"]
    forest_deciles = forest["risk_deciles"]["bands"]
    figure_links = {
        Path(item["display_path"]).name: item["report_relative_path"]
        for item in result["figures"]
    }

    lines = [
        "# Stage 4 Validation 상세 분석 보고서",
        "",
        "> 기준 모델 학습에서 저장한 validation 예측점수만 사용한 진단 결과입니다. 새 모델·확률 보정기를 학습하지 않았고 test 피처·예측·평가는 사용하지 않았습니다.",
        "",
        "## 분석 목적과 범위",
        "",
        f"- validation 고객: **{int(scope['validation_rows']):,}명**",
        f"- TARGET=1 고객: **{int(scope['validation_positive_count']):,}명 ({float(scope['validation_positive_rate']):.2%})**",
        f"- 비교 모델: **{int(scope['models_analyzed'])}개**",
        "- Calibration은 확률 보정 필요성을 진단한 것이며 보정기를 학습한 결과가 아닙니다.",
        "- Top-K 경계점수는 validation 관측값이며 운영 cutoff로 확정하지 않습니다.",
        "- 모델 차이는 같은 validation의 점추정치이며 통계적 유의성을 검정한 결론이 아닙니다.",
        "",
        "## 핵심 결론",
        "",
        f"- 현재 Stage 4 기준선은 **{reference['display_name']}**입니다. ROC-AUC·PR-AUC·KS가 현재 후보 중 가장 높지만 최종 모델은 아닙니다.",
        f"- V3 Logistic은 상위 10% 검토에서 위험고객의 **{float(logistic_top10['recall']):.2%}**를 포착했고 Lift는 **{float(logistic_top10['lift']):.3f}**입니다.",
        f"- V3 Random Forest의 상위 10% Recall은 **{float(forest_top10['recall']):.2%}**로 순위 성능은 유효하지만, 평균 점수 **{float(forest['calibration']['mean_predicted_probability']):.2%}**가 실제 비율 **{float(scope['validation_positive_rate']):.2%}**보다 크게 높습니다.",
        "- Random Forest는 `balanced_subsample`을 사용했으므로 현재 점수를 실제 상환곤란 확률이나 Logistic의 같은 숫자와 직접 비교하면 안 됩니다.",
        "",
        "## 모델별 구분력과 확률 진단",
        "",
        "| 모델 | ROC-AUC | PR-AUC(AP) | KS | Gini | Brier | 평균 예측점수 | ECE(q10) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        performance = model["performance"]
        calibration = model["calibration"]
        lines.append(
            f"| {model['display_name']} | {_format_number(performance['roc_auc'])} | "
            f"{_format_number(performance['pr_auc'])} | {_format_number(performance['ks'])} | "
            f"{_format_number(performance['gini'])} | {_format_number(performance['brier_score'])} | "
            f"{float(calibration['mean_predicted_probability']):.2%} | "
            f"{_format_number(calibration['expected_calibration_error'])} |"
        )
    lines.extend(
        [
            "",
            "PR-AUC는 사다리꼴 면적이 아니라 Average Precision입니다. 무작위 기준선은 validation 양성 비율이며, ROC-AUC가 불균형 데이터에서 낙관적으로 보일 수 있어 PR-AUC와 Top-K를 함께 봅니다.",
            "",
            f"![Validation ROC 곡선]({figure_links[FIGURE_FILENAMES[0]]})",
            "",
            f"![Validation PR 곡선]({figure_links[FIGURE_FILENAMES[1]]})",
            "",
            "## Calibration 진단",
            "",
            "같은 점수는 서로 다른 구간으로 나누지 않는 validation 10분위 진단입니다. ECE는 이 구간 정의에 의존하므로 절대적인 모델 품질 하나로 해석하지 않습니다.",
            "",
            "| 모델 | 요청 구간 | 실제 구간 | 실제 비율 | 평균 예측점수 | 평균 편향 | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        calibration = model["calibration"]
        lines.append(
            f"| {model['display_name']} | {int(calibration['requested_bin_count'])} | "
            f"{int(calibration['effective_bin_count'])} | {float(calibration['prevalence']):.2%} | "
            f"{float(calibration['mean_predicted_probability']):.2%} | "
            f"{float(calibration['mean_probability_bias']):+.2%} | "
            f"{_format_number(calibration['expected_calibration_error'])} |"
        )
    lines.extend(
        [
            "",
            f"![Validation Calibration 곡선]({figure_links[FIGURE_FILENAMES[2]]})",
            "",
            "## 위험도 Decile",
            "",
            "그림은 네 기준 모델의 위험구간을 비교하고, 표는 같은 V3 데이터를 사용한 Logistic Regression과 Random Forest를 상세 비교합니다. Decile 1이 예측위험 상위 10%, Decile 10이 하위 10%입니다. 경계 동점은 분수 가중치로 배분해 행 순서가 결과를 바꾸지 않습니다.",
            "",
            "| Decile | 고객 수 | Logistic 실제 위험률 | Logistic Lift | Logistic 누적 Recall | RF 실제 위험률 | RF Lift | RF 누적 Recall |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for logistic_band, forest_band in zip(logistic_deciles, forest_deciles, strict=True):
        lines.append(
            f"| {int(logistic_band['decile'])} | {int(logistic_band['customer_count']):,} | "
            f"{float(logistic_band['observed_positive_rate']):.2%} | "
            f"{_format_number(logistic_band['lift'], 3)} | "
            f"{float(logistic_band['cumulative_recall']):.2%} | "
            f"{float(forest_band['observed_positive_rate']):.2%} | "
            f"{_format_number(forest_band['lift'], 3)} | "
            f"{float(forest_band['cumulative_recall']):.2%} |"
        )
    lines.extend(
        [
            "",
            f"V3 Logistic의 실제 위험률은 최상위 decile **{float(logistic_deciles[0]['observed_positive_rate']):.2%}**에서 최하위 decile **{float(logistic_deciles[-1]['observed_positive_rate']):.2%}**로 낮아졌습니다. 이는 점수가 위험 순위를 실질적으로 구분한다는 validation 관찰입니다.",
            "",
            f"![Validation 위험도 Decile]({figure_links[FIGURE_FILENAMES[3]]})",
            "",
            "## 우선검토 Top-K 시나리오",
            "",
            "| 모델 | 검토 비율 | 선택 고객 | Precision | Recall | Lift |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        for scenario in model["top_k_scenarios"]:
            lines.append(
                f"| {model['display_name']} | {float(scenario['review_fraction']):.0%} | "
                f"{int(scenario['selected_customer_count']):,} | "
                f"{float(scenario['precision']):.2%} | {float(scenario['recall']):.2%} | "
                f"{_format_number(scenario['lift'], 3)} |"
            )
    lines.extend(
        [
            "",
            "`true_positive_weight`는 경계 동점이 있을 때 분수일 수 있습니다. 이 분석은 심사 가능 인원별 비교 시나리오이며 운영 정책이나 자동 승인·거절 기준이 아닙니다.",
            "",
            f"![Validation Top-K 시나리오]({figure_links[FIGURE_FILENAMES[4]]})",
            "",
            "## 데이터 사용 감사",
            "",
            f"- validation 점수 행: **{int(scope['validation_rows']):,}행**",
            f"- test 피처 사용: **{int(scope['test_feature_rows_used'])}행**",
            f"- test 예측 생성: **{'예' if scope['test_predictions_created'] else '아니요'}**",
            "- 입력 산출물과 공유 JSON·Markdown에 고객 ID가 없습니다.",
            "- 공유 JSON에는 행별 정답·예측점수와 원시 ROC·PR 좌표가 없습니다.",
            "",
            "## Stage 4 결론과 다음 단계",
            "",
            "V3 Logistic Regression을 Stage 5 모델 비교의 현재 기준선으로 사용합니다. 이는 최종 모델 확정이 아닙니다. Stage 5에서 LightGBM·TensorFlow MLP, train 내부 튜닝과 확률 보정 설계를 비교하고, 최종 모델과 운영 cutoff는 이후 단계에서 고정합니다. test는 Stage 8까지 봉인합니다.",
            "",
            "재현 명령:",
            "",
            "```bash",
            "PYTHONPATH=src .venv/bin/python -m creditlens.analysis.stage4_validation_analysis",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_stage4_validation_analysis(
    *,
    baseline_result_path: str | Path = DEFAULT_BASELINE_RESULT,
    score_artifact_path: str | Path = DEFAULT_SCORE_ARTIFACT,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Any]:
    """기준 모델 validation 점수를 검증·분석하고 공유 산출물을 만든다."""

    baseline_path = Path(baseline_result_path)
    scores_path = Path(score_artifact_path)
    output = Path(output_path)
    report = Path(report_path)
    figure_output = Path(figures_dir)

    _validate_paths(
        baseline_path=baseline_path,
        scores_path=scores_path,
        output_path=output,
        report_path=report,
        figures_dir=figure_output,
    )

    baseline = _load_json_object(baseline_path)
    experiments = _validate_baseline_result(baseline)
    (
        labels,
        scores_by_key,
        metric_deltas,
        artifact_score_dtypes,
        artifact_git_safety,
    ) = _load_and_validate_scores(
        scores_path,
        baseline=baseline,
        experiments=experiments,
    )

    models: list[dict[str, Any]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    for experiment in experiments:
        key = str(experiment["key"])
        scores = scores_by_key[key]
        metrics = evaluate_binary_metrics(labels, scores, top_fraction=0.1)
        curves[key] = _curve_coordinates(labels, scores)
        models.append(
            {
                "key": key,
                "display_name": str(experiment["display_name"]),
                "data_version": experiment.get("data_version"),
                "estimator": experiment.get("estimator"),
                "performance": {
                    metric: metrics[metric]
                    for metric in ("roc_auc", "pr_auc", "ks", "gini", "brier_score")
                },
                "curve_summary": {
                    "roc_point_count": int(curves[key]["false_positive_rate"].size),
                    "pr_point_count": int(curves[key]["precision"].size),
                    "raw_curve_coordinates_in_shared_json": False,
                },
                "calibration": build_calibration_summary(labels, scores),
                "risk_deciles": build_risk_deciles(labels, scores),
                "top_k_scenarios": build_top_k_scenarios(labels, scores),
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "run_status": "complete",
        "generated_at_utc": _utc_now(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "calibration_strategy": "quantile_without_splitting_equal_scores",
            "calibration_requested_bins": CALIBRATION_BINS,
            "risk_deciles": RISK_BANDS,
            "top_fractions": list(TOP_FRACTIONS),
            "top_k_boundary_ties": "fractional_weight",
            "metric_match_absolute_tolerance": METRIC_TOLERANCE,
        },
        "inputs": {
            "baseline_result": {
                "display_path": _display_path(baseline_path),
                "bytes": baseline_path.stat().st_size,
                "sha256": _sha256(baseline_path),
                "run_version": baseline.get("run_version"),
                "run_status": baseline.get("run_status"),
            },
            "validation_score_artifact": {
                "display_path": _display_path(scores_path),
                "bytes": scores_path.stat().st_size,
                "sha256": _sha256(scores_path),
                "split": "validation",
                "customer_ids_included": False,
                "repository_safety": artifact_git_safety,
            },
        },
        "analysis_scope": {
            "validation_rows": int(labels.size),
            "validation_positive_count": int(labels.sum(dtype=np.int64)),
            "validation_negative_count": int(labels.size - labels.sum(dtype=np.int64)),
            "validation_positive_rate": float(labels.mean()),
            "models_analyzed": len(models),
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_input": False,
            "customer_ids_in_shared_outputs": False,
            "row_level_scores_in_shared_outputs": False,
            "raw_curve_coordinates_in_shared_json": False,
        },
        "baseline_metric_recalculation_audit": {
            "status": "pass",
            "maximum_absolute_delta": max(metric_deltas.values(), default=0.0),
            "checked_metric_count": len(metric_deltas),
            "artifact_score_dtypes": artifact_score_dtypes,
        },
        "models": models,
        "summary": _build_summary(models),
        "figures": [],
    }
    result["figures"] = _write_figures(
        result,
        curves,
        figure_output,
        report,
    )
    _atomic_write_json(output, result)
    _atomic_write_text(report, render_markdown_report(result))
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4 기준 모델 validation 점수를 상세 분석합니다."
    )
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--score-artifact", type=Path, default=DEFAULT_SCORE_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_stage4_validation_analysis(
        baseline_result_path=args.baseline_result,
        score_artifact_path=args.score_artifact,
        output_path=args.output,
        report_path=args.report,
        figures_dir=args.figures_dir,
    )
    print(
        "Stage 4 validation 상세 분석 완료: "
        f"{result['analysis_scope']['models_analyzed']}개 모델, "
        f"test 피처 {result['analysis_scope']['test_feature_rows_used']}행 사용",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
