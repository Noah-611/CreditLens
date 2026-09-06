"""Stage 6 2/3 위험구간·Top-K·하위그룹 validation 분석.

Stage 5에서 잠근 V3 LightGBM과 보정기를 다시 학습하지 않는다. 공식
validation에서 위험구간과 심사 용량 시나리오를 만들고, 금융정보 가용성과
정책 제외 속성별 성능·확률 품질을 집계한다. test는 계속 봉인한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "creditlens-matplotlib"),
)

import duckdb
import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from creditlens.analysis.stage4_validation_analysis import (
    build_calibration_summary,
    build_risk_deciles,
    build_top_k_scenarios,
)
from creditlens.analysis.stage6_shap_analysis import (
    _artifact_metadata,
    _atomic_save_figure,
    _atomic_write_json,
    _atomic_write_text,
    _display_path,
    _extract_pipeline_contract,
    _read_json,
    _sha256,
    _validate_stage5_reference,
)
from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.data import DEFAULT_MART_PATHS, ModelSplit, load_model_split


SCHEMA_VERSION = "1.0"
RUN_VERSION = "stage6-strategy-analysis-v1"
STAGE_PART = "2/3"
TOP_K_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
HIGH_RISK_FRACTION = 0.10
MEDIUM_RISK_CUMULATIVE_FRACTION = 0.30
CALIBRATION_BINS = 10
SCORE_ATOL = 1e-6
MIN_PUBLISH_ROWS = 20
MIN_RELIABLE_ROWS = 1_000
MIN_RELIABLE_POSITIVES = 50
MIN_RELIABLE_NEGATIVES = 50
ROC_AUC_ALERT_DROP = 0.05
BRIER_ALERT_INCREASE = 0.02
CALIBRATION_BIAS_ALERT = 0.03
RECALL_ALERT_DROP = 0.10

DEFAULT_STAGE5_RESULT = Path("reports/stage5_final_results.json")
DEFAULT_STAGE6_SHAP_RESULT = Path("reports/stage6_shap_analysis.json")
DEFAULT_MODEL = Path("models/stage5/stage5_v3_lightgbm_candidate.joblib")
DEFAULT_CALIBRATOR = Path(
    "models/stage5/stage5_v3_probability_calibrator.joblib"
)
DEFAULT_LOCK_MANIFEST = Path(
    "models/stage5/stage5_v3_candidate_lock_manifest.json"
)
DEFAULT_VALIDATION_SCORES = Path(
    "models/stage5/stage5_final_validation_scores.joblib"
)
DEFAULT_OUTPUT = Path("reports/stage6_strategy_analysis.json")
DEFAULT_REPORT = Path("docs/Stage6_Risk_Strategy_and_Subgroup_Report.md")
DEFAULT_RISK_FIGURE = Path("reports/figures/stage6_risk_deciles.png")
DEFAULT_TOPK_FIGURE = Path("reports/figures/stage6_topk_scenarios.png")
DEFAULT_SUBGROUP_FIGURE = Path("reports/figures/stage6_subgroup_diagnostics.png")

FAMILY_LABELS = {
    "financial_history_coverage": "금융이력 가용성",
    "external_score_coverage": "외부 신용평가값 가용성",
    "age_band": "연령대",
    "gender_audit": "성별 기록 감사",
}

GROUP_LABELS = {
    "both_histories": "외부 신용·납부이력 모두 있음",
    "installments_only": "납부이력만 있음",
    "bureau_only": "외부 신용이력만 있음",
    "neither_history": "두 이력 모두 관측되지 않음",
    "zero_or_one_score": "외부 신용평가값 0~1개 관측",
    "two_scores": "외부 신용평가값 2개 관측",
    "three_scores": "외부 신용평가값 3개 관측",
    "under_30": "30세 미만",
    "age_30_to_49": "30~49세",
    "age_50_plus": "50세 이상",
    "age_missing": "연령 결측",
    "female_recorded": "여성으로 기록",
    "male_recorded": "남성으로 기록",
    "other_or_unknown": "기타·미상 기록",
}

GROUP_ORDER = {
    "financial_history_coverage": (
        "both_histories",
        "installments_only",
        "bureau_only",
        "neither_history",
    ),
    "external_score_coverage": (
        "zero_or_one_score",
        "two_scores",
        "three_scores",
    ),
    "age_band": ("under_30", "age_30_to_49", "age_50_plus", "age_missing"),
    "gender_audit": ("female_recorded", "male_recorded", "other_or_unknown"),
}


class Stage6StrategyError(RuntimeError):
    """Stage 6 위험전략 분석 계약이 깨졌을 때 발생한다."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_paths(
    *,
    references: Sequence[Path],
    output: Path,
    report: Path,
    figures: Sequence[Path],
) -> None:
    if output.suffix.lower() != ".json":
        raise Stage6StrategyError("공유 집계 결과는 JSON 파일이어야 합니다.")
    if report.suffix.lower() != ".md":
        raise Stage6StrategyError("분석 보고서는 Markdown 파일이어야 합니다.")
    if any(path.suffix.lower() != ".png" for path in figures):
        raise Stage6StrategyError("분석 그림은 PNG 파일이어야 합니다.")
    paths = (*references, output, report, *figures)
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise Stage6StrategyError("입력과 출력 경로는 서로 달라야 합니다.")


def _validate_stage6_shap_reference(
    payload: Mapping[str, Any],
    *,
    stage5_path: Path,
    verified_stage5: Mapping[str, Mapping[str, Any]],
) -> None:
    if payload.get("run_status") != "complete" or payload.get("stage_part") != "1/3":
        raise Stage6StrategyError("완료된 Stage 6 1/3 결과가 필요합니다.")
    scope = payload.get("data_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("test_feature_rows_used") != 0
        or scope.get("test_predictions_created") is not False
        or scope.get("customer_ids_in_shared_outputs") is not False
        or scope.get("row_level_values_in_shared_outputs") is not False
    ):
        raise Stage6StrategyError("Stage 6 1/3 데이터 보호 계약이 올바르지 않습니다.")
    settings = payload.get("settings")
    if (
        not isinstance(settings, Mapping)
        or settings.get("model_or_preprocessor_refit") is not False
        or settings.get("operating_cutoff_finalized") is not False
    ):
        raise Stage6StrategyError("Stage 6 1/3 모델·cutoff 상태가 올바르지 않습니다.")
    references = payload.get("references")
    if not isinstance(references, Mapping):
        raise Stage6StrategyError("Stage 6 1/3 입력 참조 기록이 없습니다.")
    stage5_reference = references.get("stage5_result")
    if (
        not isinstance(stage5_reference, Mapping)
        or stage5_reference.get("sha256") != _sha256(stage5_path)
    ):
        raise Stage6StrategyError("Stage 6 1/3이 참조한 Stage 5 결과가 다릅니다.")
    for key in ("model", "calibrator", "lock_manifest", "validation_scores"):
        reference = references.get(key)
        verified = verified_stage5.get(key)
        if (
            not isinstance(reference, Mapping)
            or not isinstance(verified, Mapping)
            or reference.get("sha256") != verified.get("sha256")
        ):
            raise Stage6StrategyError(f"Stage 6 1/3 {key} 참조 해시가 다릅니다.")


def _load_validation_gender(
    mart_path: Path,
    expected_customer_ids: pd.Index,
) -> pd.Series:
    """validation의 정책 제외 성별 기록만 로컬에서 정렬·검증해 읽는다."""

    if not mart_path.is_file():
        raise Stage6StrategyError(f"V3 분석 마트를 찾을 수 없습니다: {mart_path}")
    with duckdb.connect(database=":memory:") as connection:
        frame = connection.execute(
            """
            SELECT SK_ID_CURR, CODE_GENDER
            FROM read_parquet(?)
            WHERE SPLIT = 'validation'
            ORDER BY SK_ID_CURR
            """,
            [str(mart_path)],
        ).fetchdf()
    observed_ids = pd.Index(frame["SK_ID_CURR"].astype("int64"), name="SK_ID_CURR")
    if not observed_ids.equals(expected_customer_ids):
        raise Stage6StrategyError("성별 감사 데이터와 validation 고객 순서가 다릅니다.")
    values = frame["CODE_GENDER"].astype("string")
    values.index = expected_customer_ids
    return values


def _build_risk_bands(
    labels: np.ndarray,
    scores: np.ndarray,
    scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Top 10%와 Top 30% 점수 경계로 잠정 3개 위험구간을 만든다."""

    scenario_by_fraction = {
        round(float(item["review_fraction"]), 8): item for item in scenarios
    }
    try:
        high_cutoff = float(scenario_by_fraction[HIGH_RISK_FRACTION]["observed_cutoff_score"])
        medium_cutoff = float(
            scenario_by_fraction[MEDIUM_RISK_CUMULATIVE_FRACTION][
                "observed_cutoff_score"
            ]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Stage6StrategyError("위험구간에 필요한 Top 10%·30% 경계가 없습니다.") from error
    if not 0.0 <= medium_cutoff < high_cutoff <= 1.0:
        raise Stage6StrategyError("고·중·저 위험구간 점수 경계가 올바르지 않습니다.")

    masks = {
        "high": scores >= high_cutoff,
        "medium": (scores >= medium_cutoff) & (scores < high_cutoff),
        "low": scores < medium_cutoff,
    }
    if sum(int(mask.sum()) for mask in masks.values()) != len(labels):
        raise Stage6StrategyError("위험구간이 validation 전체를 정확히 나누지 못했습니다.")
    total_positive = int(labels.sum(dtype=np.int64))
    prevalence = float(labels.mean())
    bands: list[dict[str, Any]] = []
    for key, mask in masks.items():
        rows = int(mask.sum())
        positives = int(labels[mask].sum(dtype=np.int64))
        observed_rate = positives / rows
        mean_score = float(scores[mask].mean())
        bands.append(
            {
                "band": key,
                "rows": rows,
                "population_fraction": rows / len(labels),
                "positive_count": positives,
                "positive_recall_share": positives / total_positive,
                "mean_predicted_probability": mean_score,
                "observed_positive_rate": observed_rate,
                "calibration_gap": mean_score - observed_rate,
                "lift": observed_rate / prevalence,
                "score_min": float(scores[mask].min()),
                "score_max": float(scores[mask].max()),
            }
        )
    observed_rates = [item["observed_positive_rate"] for item in bands]
    predicted_rates = [item["mean_predicted_probability"] for item in bands]
    return {
        "status": "provisional_until_stage6_part3_lock",
        "policy": {
            "high": "score >= validation Top 10% cutoff",
            "medium": "validation Top 30% cutoff <= score < Top 10% cutoff",
            "low": "score < validation Top 30% cutoff",
            "high_cutoff": high_cutoff,
            "medium_cutoff": medium_cutoff,
            "equal_score_policy": "inclusive_threshold_for_reusable_score_bands",
        },
        "bands": bands,
        "observed_risk_strictly_descends": bool(
            observed_rates[0] > observed_rates[1] > observed_rates[2]
        ),
        "mean_score_strictly_descends": bool(
            predicted_rates[0] > predicted_rates[1] > predicted_rates[2]
        ),
        "high_to_low_observed_risk_ratio": None
        if observed_rates[2] == 0.0
        else observed_rates[0] / observed_rates[2],
    }


def _enrich_top_k_scenarios(
    labels: np.ndarray,
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    positives = int(labels.sum(dtype=np.int64))
    previous_rows = 0
    previous_true_positive = 0.0
    enriched: list[dict[str, Any]] = []
    for scenario in scenarios:
        item = dict(scenario)
        selected = int(item["selected_customer_count"])
        true_positive = float(item["true_positive_weight"])
        incremental_rows = selected - previous_rows
        incremental_positive = true_positive - previous_true_positive
        item.update(
            {
                "false_positive_weight": selected - true_positive,
                "missed_positive_weight": positives - true_positive,
                "incremental_review_rows": incremental_rows,
                "incremental_true_positive_weight": incremental_positive,
                "incremental_precision": incremental_positive / incremental_rows,
            }
        )
        enriched.append(item)
        previous_rows = selected
        previous_true_positive = true_positive
    return enriched


def _define_subgroup_families(
    features: pd.DataFrame,
    gender: pd.Series,
) -> dict[str, pd.Series]:
    required = {
        "BUREAU_HAS_HISTORY",
        "INST_HAS_HISTORY",
        "APP_EXT_SOURCE_OBSERVED_COUNT",
        "APP_AGE_YEARS",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise Stage6StrategyError(f"하위그룹 정의 피처가 없습니다: {missing}")
    if not gender.index.equals(features.index):
        raise Stage6StrategyError("성별 감사 값과 validation 피처 순서가 다릅니다.")

    bureau = pd.to_numeric(features["BUREAU_HAS_HISTORY"], errors="coerce")
    installments = pd.to_numeric(features["INST_HAS_HISTORY"], errors="coerce")
    if bureau.isna().any() or installments.isna().any():
        raise Stage6StrategyError("금융이력 존재 피처에 결측값이 있습니다.")
    history = pd.Series(index=features.index, dtype="string")
    history[(bureau == 1) & (installments == 1)] = "both_histories"
    history[(bureau == 0) & (installments == 1)] = "installments_only"
    history[(bureau == 1) & (installments == 0)] = "bureau_only"
    history[(bureau == 0) & (installments == 0)] = "neither_history"
    if history.isna().any():
        raise Stage6StrategyError("금융이력 존재 피처가 0/1 계약을 벗어났습니다.")

    observed = pd.to_numeric(
        features["APP_EXT_SOURCE_OBSERVED_COUNT"], errors="coerce"
    )
    if observed.isna().any() or not observed.isin((0, 1, 2, 3)).all():
        raise Stage6StrategyError("외부 신용평가값 관측 개수가 0~3 범위를 벗어났습니다.")
    external = pd.Series(index=features.index, dtype="string")
    external[observed <= 1] = "zero_or_one_score"
    external[observed == 2] = "two_scores"
    external[observed == 3] = "three_scores"

    age = pd.to_numeric(features["APP_AGE_YEARS"], errors="coerce")
    age_band = pd.Series("age_missing", index=features.index, dtype="string")
    age_band[age < 30] = "under_30"
    age_band[(age >= 30) & (age < 50)] = "age_30_to_49"
    age_band[age >= 50] = "age_50_plus"

    gender_group = pd.Series("other_or_unknown", index=features.index, dtype="string")
    gender_group[gender.eq("F")] = "female_recorded"
    gender_group[gender.eq("M")] = "male_recorded"
    return {
        "financial_history_coverage": history,
        "external_score_coverage": external,
        "age_band": age_band,
        "gender_audit": gender_group,
    }


def _subgroup_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    *,
    global_cutoff: float,
    overall: Mapping[str, float],
) -> dict[str, Any]:
    group_labels = labels[mask]
    group_scores = scores[mask]
    rows = int(mask.sum())
    positives = int(group_labels.sum(dtype=np.int64))
    negatives = rows - positives
    if rows < MIN_PUBLISH_ROWS:
        return {
            "rows": rows,
            "positive_count": None,
            "detail_status": "suppressed_small_group",
            "reliable_for_alerts": False,
            "suppression_reason": f"fewer_than_{MIN_PUBLISH_ROWS}_rows",
        }

    metrics = evaluate_binary_metrics(
        group_labels,
        group_scores,
        threshold=global_cutoff,
        top_fraction=HIGH_RISK_FRACTION,
    )
    threshold = metrics["threshold_metrics"]
    calibration = build_calibration_summary(
        group_labels,
        group_scores,
        n_bins=min(CALIBRATION_BINS, rows),
    )
    reliable = (
        rows >= MIN_RELIABLE_ROWS
        and positives >= MIN_RELIABLE_POSITIVES
        and negatives >= MIN_RELIABLE_NEGATIVES
    )
    prevalence = float(metrics["prevalence"])
    selection_rate = int(threshold["predicted_positive_count"]) / rows
    recall = threshold["recall"]
    roc_auc = metrics["roc_auc"]
    pr_auc = metrics["pr_auc"]
    brier = float(metrics["brier_score"])
    mean_bias = float(calibration["mean_probability_bias"])
    alerts: list[str] = []
    if reliable:
        if roc_auc is not None and roc_auc < overall["roc_auc"] - ROC_AUC_ALERT_DROP:
            alerts.append("roc_auc_drop")
        if brier > overall["brier_score"] + BRIER_ALERT_INCREASE:
            alerts.append("brier_increase")
        if abs(mean_bias) > CALIBRATION_BIAS_ALERT:
            alerts.append("absolute_calibration_bias")
        if recall is not None and recall < overall["global_cutoff_recall"] - RECALL_ALERT_DROP:
            alerts.append("global_cutoff_recall_drop")
    return {
        "rows": rows,
        "positive_count": positives,
        "negative_count": negatives,
        "detail_status": "reported",
        "reliable_for_alerts": reliable,
        "support_note": "reliable" if reliable else "exploratory_small_support",
        "positive_rate": prevalence,
        "mean_predicted_probability": float(group_scores.mean()),
        "mean_probability_bias": mean_bias,
        "expected_calibration_error": float(
            calibration["expected_calibration_error"]
        ),
        "brier_score": brier,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pr_auc_over_prevalence": None
        if pr_auc is None or prevalence == 0.0
        else float(pr_auc / prevalence),
        "ks": metrics["ks"],
        "global_cutoff": {
            "score": global_cutoff,
            "selected_count": int(threshold["predicted_positive_count"]),
            "selection_rate": selection_rate,
            "true_positive": int(threshold["true_positive"]),
            "false_positive": int(threshold["false_positive"]),
            "recall": recall,
            "precision": threshold["precision"],
        },
        "delta_vs_overall": {
            "roc_auc": None if roc_auc is None else float(roc_auc - overall["roc_auc"]),
            "brier_score": brier - overall["brier_score"],
            "global_cutoff_recall": None
            if recall is None
            else float(recall - overall["global_cutoff_recall"]),
            "selection_rate": selection_rate - overall["global_cutoff_selection_rate"],
        },
        "diagnostic_alerts": alerts,
    }


def _build_subgroup_analysis(
    labels: np.ndarray,
    scores: np.ndarray,
    families: Mapping[str, pd.Series],
    *,
    global_cutoff: float,
) -> dict[str, Any]:
    overall_metrics = evaluate_binary_metrics(
        labels,
        scores,
        threshold=global_cutoff,
        top_fraction=HIGH_RISK_FRACTION,
    )
    overall_threshold = overall_metrics["threshold_metrics"]
    overall = {
        "roc_auc": float(overall_metrics["roc_auc"]),
        "brier_score": float(overall_metrics["brier_score"]),
        "global_cutoff_recall": float(overall_threshold["recall"]),
        "global_cutoff_selection_rate": int(
            overall_threshold["predicted_positive_count"]
        )
        / len(labels),
    }
    family_results: list[dict[str, Any]] = []
    alert_records: list[dict[str, Any]] = []
    for family_key, membership in families.items():
        if len(membership) != len(labels) or membership.isna().any():
            raise Stage6StrategyError(f"{family_key} 하위그룹 배정이 완전하지 않습니다.")
        groups: list[dict[str, Any]] = []
        observed_groups = set(membership.astype(str))
        for group_key in GROUP_ORDER[family_key]:
            if group_key not in observed_groups:
                continue
            mask = membership.eq(group_key).to_numpy(dtype=bool)
            metrics = _subgroup_metrics(
                labels,
                scores,
                mask,
                global_cutoff=global_cutoff,
                overall=overall,
            )
            item = {
                "group": str(group_key),
                "display_name": GROUP_LABELS.get(str(group_key), str(group_key)),
                **metrics,
            }
            groups.append(item)
            for alert in item.get("diagnostic_alerts", []):
                if alert == "roc_auc_drop":
                    observed_value = item["roc_auc"]
                    overall_reference = overall["roc_auc"]
                    difference = item["delta_vs_overall"]["roc_auc"]
                    trigger_threshold = -ROC_AUC_ALERT_DROP
                elif alert == "brier_increase":
                    observed_value = item["brier_score"]
                    overall_reference = overall["brier_score"]
                    difference = item["delta_vs_overall"]["brier_score"]
                    trigger_threshold = BRIER_ALERT_INCREASE
                elif alert == "absolute_calibration_bias":
                    observed_value = item["mean_probability_bias"]
                    overall_reference = 0.0
                    difference = item["mean_probability_bias"]
                    trigger_threshold = CALIBRATION_BIAS_ALERT
                else:
                    observed_value = item["global_cutoff"]["recall"]
                    overall_reference = overall["global_cutoff_recall"]
                    difference = item["delta_vs_overall"]["global_cutoff_recall"]
                    trigger_threshold = -RECALL_ALERT_DROP
                alert_records.append(
                    {
                        "family": family_key,
                        "group": str(group_key),
                        "alert": str(alert),
                        "observed_value": observed_value,
                        "overall_reference": overall_reference,
                        "difference_from_overall": difference,
                        "trigger_threshold": trigger_threshold,
                    }
                )
        if sum(int(item["rows"]) for item in groups) != len(labels):
            raise Stage6StrategyError(f"{family_key} 하위그룹 행 수가 보존되지 않았습니다.")
        family_results.append(
            {
                "family": family_key,
                "display_name": FAMILY_LABELS[family_key],
                "groups": groups,
            }
        )
    return {
        "purpose": "aggregate_diagnostic_not_group_specific_credit_policy",
        "global_cutoff": global_cutoff,
        "overall_at_global_cutoff": overall,
        "publication_minimum_rows": MIN_PUBLISH_ROWS,
        "reliable_alert_minimums": {
            "rows": MIN_RELIABLE_ROWS,
            "positives": MIN_RELIABLE_POSITIVES,
            "negatives": MIN_RELIABLE_NEGATIVES,
        },
        "alert_thresholds": {
            "roc_auc_drop": ROC_AUC_ALERT_DROP,
            "brier_increase": BRIER_ALERT_INCREASE,
            "absolute_calibration_bias": CALIBRATION_BIAS_ALERT,
            "global_cutoff_recall_drop": RECALL_ALERT_DROP,
        },
        "families": family_results,
        "reliable_group_alerts": alert_records,
        "reliable_group_alert_count": len(alert_records),
        "group_specific_cutoffs_allowed": False,
        "causal_or_fairness_conclusion_allowed": False,
    }


def _plot_risk_deciles(deciles: Mapping[str, Any], path: Path) -> dict[str, Any]:
    bands = deciles["bands"]
    x = np.arange(1, len(bands) + 1)
    observed = np.asarray([item["observed_positive_rate"] for item in bands])
    predicted = np.asarray([item["mean_predicted_probability"] for item in bands])
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.bar(x, observed, color="#E45756", alpha=0.82, label="Observed difficulty rate")
    axis.plot(
        x,
        predicted,
        color="#4C78A8",
        marker="o",
        linewidth=2.2,
        label="Mean predicted probability",
    )
    axis.set_title("Validation risk deciles (1 = highest model risk)")
    axis.set_xlabel("Risk decile")
    axis.set_ylabel("Rate")
    axis.set_xticks(x)
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return _atomic_save_figure(figure, path)


def _plot_top_k(scenarios: Sequence[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    fractions = np.asarray([item["review_fraction"] for item in scenarios])
    recall = np.asarray([item["recall"] for item in scenarios])
    precision = np.asarray([item["precision"] for item in scenarios])
    lift = np.asarray([item["lift"] for item in scenarios])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(fractions * 100, recall * 100, marker="o", label="Recall")
    axes[0].plot(fractions * 100, precision * 100, marker="o", label="Precision")
    axes[0].set_title("Capture versus review capacity")
    axes[0].set_xlabel("Reviewed population (%)")
    axes[0].set_ylabel("Metric (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].bar(fractions * 100, lift, width=3.5, color="#59A14F")
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_title("Lift by review capacity")
    axes[1].set_xlabel("Reviewed population (%)")
    axes[1].set_ylabel("Lift")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _atomic_save_figure(figure, path)


def _plot_subgroups(analysis: Mapping[str, Any], path: Path) -> dict[str, Any]:
    labels: list[str] = []
    auc_deltas: list[float] = []
    recall_deltas: list[float] = []
    colors: list[str] = []
    palette = {
        "financial_history_coverage": "#F28E2B",
        "external_score_coverage": "#4E79A7",
        "age_band": "#59A14F",
        "gender_audit": "#B07AA1",
    }
    for family in analysis["families"]:
        for group in family["groups"]:
            if not group.get("reliable_for_alerts") or group.get("roc_auc") is None:
                continue
            labels.append(group["group"])
            auc_deltas.append(group["delta_vs_overall"]["roc_auc"])
            recall_deltas.append(group["delta_vs_overall"]["global_cutoff_recall"])
            colors.append(palette[family["family"]])
    y = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(14, max(6.0, 0.45 * len(labels))))
    axes[0].barh(y, auc_deltas, color=colors)
    axes[0].axvline(0.0, color="#333333", linewidth=1)
    axes[0].axvline(-ROC_AUC_ALERT_DROP, color="#E15759", linestyle="--")
    axes[0].set_title("ROC-AUC difference from overall")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Subgroup - overall")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(y, recall_deltas, color=colors)
    axes[1].axvline(0.0, color="#333333", linewidth=1)
    axes[1].axvline(-RECALL_ALERT_DROP, color="#E15759", linestyle="--")
    axes[1].set_title("Global-cutoff recall difference")
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Subgroup - overall")
    axes[1].grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    figure.suptitle("Reliable subgroup diagnostics (validation only)")
    figure.tight_layout()
    return _atomic_save_figure(figure, path)


def _relative(report: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), report.parent.resolve())).as_posix()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_markdown_report(
    payload: Mapping[str, Any],
    *,
    report_path: Path,
    risk_figure_path: Path,
    topk_figure_path: Path,
    subgroup_figure_path: Path,
) -> str:
    validation = payload["validation_replay"]
    risk = payload["risk_strategy"]
    subgroup = payload["subgroup_analysis"]
    lines = [
        "# Stage 6 2/3 위험전략·하위그룹 분석 보고서",
        "",
        "> Stage 5에서 고정한 V3 LightGBM을 다시 학습하지 않고 validation에서 위험구간, 심사 용량과 하위그룹을 분석했습니다. test는 사용하지 않았습니다.",
        "",
        "## 왜 이 분석을 했는가",
        "",
        "전체 ROC-AUC와 PR-AUC만으로는 실제 검토 인원에 따라 위험고객을 얼마나 찾는지, 점수 구간이 실제 위험률을 구분하는지, 정보가 부족한 고객이나 주요 하위그룹에서 성능이 달라지는지 알 수 없습니다. 따라서 하나의 점수를 업무 참고 정보로 쓰기 전에 용량·구간·하위그룹별 한계를 validation에서 확인했습니다.",
        "",
        "## 분석 계약",
        "",
        f"- 분석 데이터: validation {payload['data_scope']['validation_rows']:,}명",
        "- 모델·전처리 재학습 또는 변경: 없음",
        "- 고위험 잠정 구간: validation 위험점수 상위 10% 경계 이상",
        "- 중위험 잠정 구간: 상위 10% 다음부터 누적 상위 30% 경계 이상",
        "- 하위그룹별 별도 cutoff 사용: 금지",
        f"- test 피처·예측 사용: {payload['data_scope']['test_feature_rows_used']}행",
        "",
        "## 고정 모델 재현 확인",
        "",
        f"- ROC-AUC / PR-AUC: `{_fmt(validation['metrics']['roc_auc'])}` / `{_fmt(validation['metrics']['pr_auc'])}`",
        f"- 저장 점수와 현재 예측 최대 차이: `{_fmt(validation['max_abs_probability_difference'], 10)}`",
        f"- 선택된 확률 보정: `{validation['calibration_method']}`",
        "",
        "## 위험구간",
        "",
        "아래 구간은 Stage 6 3/3에서 잠그기 전의 잠정 기준입니다. 실제 상환곤란 비율이 고위험→중위험→저위험 순으로 낮아지는지를 확인합니다.",
        "",
        "| 구간 | 고객 수 | 인구 비중 | 평균 예측확률 | 실제 상환곤란 비율 | 전체 위험고객 중 포함 비중 | Lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    band_names = {"high": "고위험", "medium": "중위험", "low": "저위험"}
    for item in risk["risk_bands"]["bands"]:
        lines.append(
            f"| {band_names[item['band']]} | {item['rows']:,} | "
            f"{_fmt(100 * item['population_fraction'], 2)}% | "
            f"{_fmt(100 * item['mean_predicted_probability'], 2)}% | "
            f"{_fmt(100 * item['observed_positive_rate'], 2)}% | "
            f"{_fmt(100 * item['positive_recall_share'], 2)}% | "
            f"{_fmt(item['lift'], 2)} |"
        )
    lines.extend(
        [
            "",
            f"- 고위험/저위험 실제 위험률 배수: `{_fmt(risk['risk_bands']['high_to_low_observed_risk_ratio'], 2)}`배",
            f"- 실제 위험률 순서 단조성: `{'PASS' if risk['risk_bands']['observed_risk_strictly_descends'] else 'FAIL'}`",
            "",
            f"![위험도 decile]({_relative(report_path, risk_figure_path)})",
            "",
            "### 위험도 decile",
            "",
            "| Decile | 고객 수 | 평균 예측확률 | 실제 상환곤란 비율 | Lift | 누적 Recall |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in risk["deciles"]["bands"]:
        lines.append(
            f"| {item['decile']} | {item['customer_count']:,} | "
            f"{_fmt(100 * item['mean_predicted_probability'], 2)}% | "
            f"{_fmt(100 * item['observed_positive_rate'], 2)}% | "
            f"{_fmt(item['lift'], 2)} | {_fmt(100 * item['cumulative_recall'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "## 심사 용량별 Top-K 시나리오",
            "",
            "Top-K는 점수가 높은 순서로 정해진 수만큼 검토한다고 가정한 분석입니다. 검토 범위를 넓히면 Recall은 늘지만 Precision과 Lift는 낮아지는 것이 일반적인 교환관계입니다.",
            "",
            "| 검토 비중 | 검토 인원 | cutoff | 포착 위험고객 | Recall | Precision | Lift | 추가 구간 Precision |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in risk["top_k_scenarios"]:
        lines.append(
            f"| {_fmt(100 * item['review_fraction'], 0)}% | "
            f"{item['selected_customer_count']:,} | {_fmt(item['observed_cutoff_score'], 6)} | "
            f"{_fmt(item['true_positive_weight'], 0)} | {_fmt(100 * item['recall'], 2)}% | "
            f"{_fmt(100 * item['precision'], 2)}% | {_fmt(item['lift'], 2)} | "
            f"{_fmt(100 * item['incremental_precision'], 2)}% |"
        )
    lines.extend(
        [
            "",
            f"![Top-K 시나리오]({_relative(report_path, topk_figure_path)})",
            "",
            "기존 핵심 평가 기준인 Top 10%는 이번 단계에서도 비교 기준으로 유지하지만 아직 운영 cutoff로 확정하지 않습니다.",
            "",
            "## 하위그룹 진단",
            "",
            "하위그룹에는 전체와 같은 고위험 cutoff를 적용했습니다. 표본 1,000명·양성 50명·음성 50명 이상인 그룹만 자동 경고 판단에 사용하고, 20명 미만 그룹은 세부 지표를 공개하지 않았습니다.",
            "",
        ]
    )
    for family in subgroup["families"]:
        lines.extend(
            [
                f"### {family['display_name']}",
                "",
                "| 그룹 | 고객 수 | 위험률 | ROC-AUC | PR-AUC | Brier | 확률편향 | cutoff Recall | 선택률 | 상태 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in family["groups"]:
            if item["detail_status"] != "reported":
                lines.append(
                    f"| {item['display_name']} | {item['rows']:,} | 비공개 | 비공개 | 비공개 | 비공개 | 비공개 | 비공개 | 비공개 | 소표본 억제 |"
                )
                continue
            status = "경고 판단 가능" if item["reliable_for_alerts"] else "탐색적 소표본"
            lines.append(
                f"| {item['display_name']} | {item['rows']:,} | "
                f"{_fmt(100 * item['positive_rate'], 2)}% | {_fmt(item['roc_auc'])} | "
                f"{_fmt(item['pr_auc'])} | {_fmt(item['brier_score'])} | "
                f"{_fmt(100 * item['mean_probability_bias'], 2)}%p | "
                f"{_fmt(100 * item['global_cutoff']['recall'], 2)}% | "
                f"{_fmt(100 * item['global_cutoff']['selection_rate'], 2)}% | {status} |"
            )
        lines.append("")
    lines.extend(
        [
            f"![하위그룹 진단]({_relative(report_path, subgroup_figure_path)})",
            "",
            f"- 신뢰 가능한 그룹에서 발생한 진단 경고 수: `{subgroup['reliable_group_alert_count']}`건",
            "- 경고는 개선 검토 신호이며 차별이나 인과관계를 확정하는 판정이 아닙니다.",
            "- 성별 기록은 모델 피처에서 제외됐고, 여기서는 집계 감사에만 사용했습니다.",
            "- 그룹마다 위험률이 다르므로 PR-AUC·Brier·선택률을 단순 숫자 하나로만 비교하지 않습니다.",
            "- 하위그룹별 별도 cutoff는 만들지 않습니다.",
            "",
            "### 진단 경고 상세",
            "",
        ]
    )
    if not subgroup["reliable_group_alerts"]:
        lines.append("- 사전 정의한 기준을 넘은 신뢰 가능한 하위그룹 경고가 없습니다.")
    for alert in subgroup["reliable_group_alerts"]:
        family_name = FAMILY_LABELS[alert["family"]]
        group_name = GROUP_LABELS[alert["group"]]
        if alert["alert"] == "brier_increase":
            detail = (
                f"Brier `{_fmt(alert['observed_value'])}` vs 전체 "
                f"`{_fmt(alert['overall_reference'])}` "
                f"(차이 `+{_fmt(alert['difference_from_overall'])}`)"
            )
        elif alert["alert"] == "global_cutoff_recall_drop":
            detail = (
                f"공통 cutoff Recall `{_fmt(100 * alert['observed_value'], 2)}%` vs 전체 "
                f"`{_fmt(100 * alert['overall_reference'], 2)}%` "
                f"(차이 `{_fmt(100 * alert['difference_from_overall'], 2)}%p`)"
            )
        elif alert["alert"] == "roc_auc_drop":
            detail = (
                f"ROC-AUC `{_fmt(alert['observed_value'])}` vs 전체 "
                f"`{_fmt(alert['overall_reference'])}` "
                f"(차이 `{_fmt(alert['difference_from_overall'])}`)"
            )
        else:
            detail = (
                "평균 확률편향 "
                f"`{_fmt(100 * alert['observed_value'], 2)}%p`"
            )
        lines.append(f"- **{family_name} · {group_name}:** {detail}")
    lines.extend(
        [
            "",
            "Brier는 그룹의 원래 위험률에도 영향을 받으므로, Brier 경고 하나만으로 모델이 그 그룹에서 나쁘다고 단정하지 않습니다. Recall 경고도 전체 공통 cutoff에서의 선택률 차이와 함께 해석합니다.",
            "",
            "## Stage 6 3/3로 넘기는 판단 자료",
            "",
            f"- 위험구간 순서: `{'PASS' if risk['risk_bands']['observed_risk_strictly_descends'] else 'REVIEW'}`",
            f"- 신뢰 가능한 하위그룹 경고: `{subgroup['reliable_group_alert_count']}`건",
            "- Top 10%는 현재 비교 기준일 뿐 최종 cutoff가 아닙니다.",
            "- 다음 단계에서 경고 내용을 검토하고 개선 여부를 판단한 뒤 모델·보정기·위험구간·cutoff·checksum과 사용 한계를 고정합니다.",
            "- test는 Stage 8에서 고정된 산출물에 한 번만 사용하며, 그 결과로 모델을 다시 조정하지 않습니다.",
            "",
            "## 해석 한계",
            "",
            "- 모든 결과는 해외 과거 공개 데이터의 validation 분석이며 국내 실제 금융환경을 대표하지 않습니다.",
            "- 점수구간과 하위그룹 차이는 연관성 진단이며 개인의 원인이나 집단의 본질적 특성을 뜻하지 않습니다.",
            "- 이 결과만으로 대출을 자동 승인·거절하거나 공식 신용등급을 만들 수 없습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def run_stage6_strategy_analysis(
    *,
    stage5_result_path: str | Path = DEFAULT_STAGE5_RESULT,
    stage6_shap_result_path: str | Path = DEFAULT_STAGE6_SHAP_RESULT,
    model_path: str | Path = DEFAULT_MODEL,
    calibrator_path: str | Path = DEFAULT_CALIBRATOR,
    lock_manifest_path: str | Path = DEFAULT_LOCK_MANIFEST,
    validation_scores_path: str | Path = DEFAULT_VALIDATION_SCORES,
    mart_path: str | Path = DEFAULT_MART_PATHS["v3"],
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    risk_figure_path: str | Path = DEFAULT_RISK_FIGURE,
    topk_figure_path: str | Path = DEFAULT_TOPK_FIGURE,
    subgroup_figure_path: str | Path = DEFAULT_SUBGROUP_FIGURE,
    gender_loader: Callable[[Path, pd.Index], pd.Series] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """잠긴 모델을 validation에서 위험전략과 하위그룹 관점으로 진단한다."""

    stage5_path = Path(stage5_result_path)
    stage6_shap_path = Path(stage6_shap_result_path)
    model_file = Path(model_path)
    calibrator_file = Path(calibrator_path)
    lock_file = Path(lock_manifest_path)
    validation_scores_file = Path(validation_scores_path)
    mart_file = Path(mart_path)
    output = Path(output_path)
    report = Path(report_path)
    risk_figure = Path(risk_figure_path)
    topk_figure = Path(topk_figure_path)
    subgroup_figure = Path(subgroup_figure_path)
    _validate_paths(
        references=(
            stage5_path,
            stage6_shap_path,
            model_file,
            calibrator_file,
            lock_file,
            validation_scores_file,
            mart_file,
        ),
        output=output,
        report=report,
        figures=(risk_figure, topk_figure, subgroup_figure),
    )
    started = time.perf_counter()
    try:
        stage5 = _read_json(stage5_path, label="Stage 5 결과")
        stage6_shap = _read_json(stage6_shap_path, label="Stage 6 1/3 결과")
        verified = _validate_stage5_reference(
            stage5,
            model_path=model_file,
            calibrator_path=calibrator_file,
            lock_path=lock_file,
            validation_scores_path=validation_scores_file,
        )
        _validate_stage6_shap_reference(
            stage6_shap,
            stage5_path=stage5_path,
            verified_stage5=verified,
        )
    except Exception as error:
        if isinstance(error, Stage6StrategyError):
            raise
        raise Stage6StrategyError("Stage 5·Stage 6 1/3 입력 검증에 실패했습니다.") from error
    if progress is not None:
        progress("Stage 5 잠금 산출물과 Stage 6 1/3 참조 해시 검증 완료")

    try:
        pipeline = joblib.load(model_file)
        calibrator = joblib.load(calibrator_file)
        stored = joblib.load(validation_scores_file)
        pipeline, _, features, _, _ = _extract_pipeline_contract(pipeline)
    except Exception as error:
        raise Stage6StrategyError("고정 모델·보정기·validation 점수를 읽지 못했습니다.") from error
    if not isinstance(pipeline, Pipeline):
        raise Stage6StrategyError("고정 모델은 sklearn Pipeline이어야 합니다.")
    if getattr(calibrator, "method", None) != "identity":
        raise Stage6StrategyError("Stage 6 2/3은 잠긴 identity 보정기를 기대합니다.")

    validation: ModelSplit = load_model_split(mart_file, "v3", "validation")
    if validation.name != "validation" or tuple(validation.X.columns) != features:
        raise Stage6StrategyError("validation 모델 입력 계약이 고정 모델과 다릅니다.")
    labels = validation.y.to_numpy(dtype=np.int8, copy=False)
    raw_scores = np.asarray(
        pipeline.predict_proba(validation.X)[:, 1], dtype=np.float64
    )
    scores = np.asarray(calibrator.predict(raw_scores), dtype=np.float64)
    stored_labels = np.asarray(stored.get("y_true"))
    stored_scores = stored.get("scores", {})
    stored_raw = np.asarray(stored_scores.get("raw"))
    stored_calibrated = np.asarray(stored_scores.get("calibrated"))
    if (
        not np.array_equal(labels, stored_labels)
        or stored_raw.shape != scores.shape
        or stored_calibrated.shape != scores.shape
    ):
        raise Stage6StrategyError("저장 validation 점수의 행 계약이 다릅니다.")
    raw_difference = float(np.max(np.abs(raw_scores - stored_raw)))
    calibrated_difference = float(np.max(np.abs(scores - stored_calibrated)))
    if max(raw_difference, calibrated_difference) > SCORE_ATOL:
        raise Stage6StrategyError("현재 예측이 Stage 5 저장 점수와 다릅니다.")
    metrics = evaluate_binary_metrics(labels, scores, top_fraction=HIGH_RISK_FRACTION)
    stage6_metrics = stage6_shap["validation_replay"]["metrics"]
    for name in ("roc_auc", "pr_auc", "ks", "gini", "brier_score"):
        if not np.isclose(metrics[name], stage6_metrics[name], rtol=0.0, atol=SCORE_ATOL):
            raise Stage6StrategyError(f"현재 validation {name}이 Stage 6 1/3과 다릅니다.")
    if progress is not None:
        progress("공식 validation 예측 재현과 test 0행 계약 확인")

    base_scenarios = build_top_k_scenarios(
        labels,
        scores,
        fractions=TOP_K_FRACTIONS,
    )
    top_k_scenarios = _enrich_top_k_scenarios(labels, base_scenarios)
    risk_bands = _build_risk_bands(labels, scores, top_k_scenarios)
    deciles = build_risk_deciles(labels, scores, n_bands=10)
    if progress is not None:
        progress("위험구간·decile·5~30% Top-K 시나리오 계산 완료")

    loader = gender_loader or _load_validation_gender
    gender = loader(mart_file, validation.customer_ids)
    families = _define_subgroup_families(validation.X, gender)
    high_cutoff = float(risk_bands["policy"]["high_cutoff"])
    subgroup_analysis = _build_subgroup_analysis(
        labels,
        scores,
        families,
        global_cutoff=high_cutoff,
    )
    if progress is not None:
        progress("금융이력·외부점수·연령·성별 기록 하위그룹 집계 완료")

    figure_metadata = {
        "risk_deciles": _plot_risk_deciles(deciles, risk_figure),
        "top_k_scenarios": _plot_top_k(top_k_scenarios, topk_figure),
        "subgroup_diagnostics": _plot_subgroups(subgroup_analysis, subgroup_figure),
    }
    risk_status = risk_bands["observed_risk_strictly_descends"]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "stage_part": STAGE_PART,
        "run_status": "complete",
        "generated_at_utc": _utc_now(),
        "data_scope": {
            "validation_rows": len(labels),
            "validation_positive_count": int(labels.sum(dtype=np.int64)),
            "validation_positive_rate": float(labels.mean()),
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
            "row_level_values_in_shared_outputs": False,
            "policy_excluded_gender_used_for_aggregate_audit_only": True,
        },
        "references": {
            "stage5_result": {
                "display_path": _display_path(stage5_path),
                "sha256": _sha256(stage5_path),
                "run_version": stage5.get("run_version"),
            },
            "stage6_shap_result": {
                "display_path": _display_path(stage6_shap_path),
                "sha256": _sha256(stage6_shap_path),
                "run_version": stage6_shap.get("run_version"),
            },
            **verified,
        },
        "model_contract": {
            "data_version": "v3",
            "model_family": "LightGBM",
            "source_feature_columns": len(features),
            "calibration_method": getattr(calibrator, "method"),
            "model_or_preprocessor_refit": False,
            "group_specific_cutoffs_allowed": False,
            "automatic_credit_decision_allowed": False,
        },
        "validation_replay": {
            "metrics": metrics,
            "max_abs_probability_difference": max(
                raw_difference, calibrated_difference
            ),
            "calibration_method": getattr(calibrator, "method"),
            "stage6_part1_metrics_matched": True,
        },
        "risk_strategy": {
            "deciles": deciles,
            "risk_bands": risk_bands,
            "top_k_scenarios": top_k_scenarios,
            "top10_status": "comparison_reference_not_final_operating_cutoff",
        },
        "subgroup_analysis": subgroup_analysis,
        "diagnostic_summary": {
            "risk_band_order_passed": risk_status,
            "reliable_subgroup_alert_count": subgroup_analysis[
                "reliable_group_alert_count"
            ],
            "model_improvement_decision": "pending_stage6_part3",
            "operating_cutoff_finalized": False,
            "test_remains_sealed": True,
        },
        "figures": figure_metadata,
        "resources": {
            "total_seconds": round(time.perf_counter() - started, 3),
            "process_peak_rss_mb": _peak_rss_mb(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "duckdb": duckdb.__version__,
        },
        "stage6_next": {
            "part": "3/3",
            "work": "improvement_decision_and_final_artifact_lock",
            "test_remains_sealed": True,
            "test_first_use_stage": 8,
        },
    }
    report_text = render_markdown_report(
        payload,
        report_path=report,
        risk_figure_path=risk_figure,
        topk_figure_path=topk_figure,
        subgroup_figure_path=subgroup_figure,
    )
    _atomic_write_text(report, report_text)
    _atomic_write_json(output, payload)
    if progress is not None:
        progress(
            f"Stage 6 2/3 완료: validation {len(labels):,}명, "
            f"하위그룹 경고 {subgroup_analysis['reliable_group_alert_count']}건, "
            "test 0행 사용"
        )
    return payload


def _peak_rss_mb() -> float:
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if platform.system() != "Darwin" else 1024.0 * 1024.0
    return round(value / divisor, 3)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 6 2/3 위험구간·Top-K·하위그룹 분석을 실행합니다."
    )
    parser.add_argument("--stage5-result", type=Path, default=DEFAULT_STAGE5_RESULT)
    parser.add_argument(
        "--stage6-shap-result", type=Path, default=DEFAULT_STAGE6_SHAP_RESULT
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--lock-manifest", type=Path, default=DEFAULT_LOCK_MANIFEST)
    parser.add_argument(
        "--validation-scores", type=Path, default=DEFAULT_VALIDATION_SCORES
    )
    parser.add_argument("--mart", type=Path, default=DEFAULT_MART_PATHS["v3"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--risk-figure", type=Path, default=DEFAULT_RISK_FIGURE)
    parser.add_argument("--topk-figure", type=Path, default=DEFAULT_TOPK_FIGURE)
    parser.add_argument(
        "--subgroup-figure", type=Path, default=DEFAULT_SUBGROUP_FIGURE
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_stage6_strategy_analysis(
        stage5_result_path=args.stage5_result,
        stage6_shap_result_path=args.stage6_shap_result,
        model_path=args.model,
        calibrator_path=args.calibrator,
        lock_manifest_path=args.lock_manifest,
        validation_scores_path=args.validation_scores,
        mart_path=args.mart,
        output_path=args.output,
        report_path=args.report,
        risk_figure_path=args.risk_figure,
        topk_figure_path=args.topk_figure,
        subgroup_figure_path=args.subgroup_figure,
        progress=print,
    )
    print(
        "Stage 6 2/3 완료: "
        f"validation {result['data_scope']['validation_rows']:,}명, "
        f"test {result['data_scope']['test_feature_rows_used']}행 사용"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
