"""Stage 6 3/3: 경고 검토, 프로토타입 정책 결정, 변경 검출 가능한 모델 동결.

실행은 validation만 사용한다. 이미 생성한 동결 버전은 덮어쓰지 않는다.
--verify는 데이터 조회 없이 동결 파일·추론 코드·환경을 점검한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from creditlens.analysis.stage6_shap_analysis import (
    _extract_pipeline_contract, _resolve_source_groups, _validate_stage5_reference,
)
from creditlens.analysis.stage6_strategy_analysis import _validate_stage6_shap_reference
from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.data import DEFAULT_MART_PATHS, load_model_split
from creditlens.modeling.frozen_model import (
    DEFAULT_BUNDLE, DEFAULT_RESULT, RUNTIME_SOURCES, FrozenModelError,
    apply_risk_policy, file_digest, runtime_versions, verify_bundle,
)
from creditlens.modeling.preprocessing import transformed_feature_names
from creditlens.modeling.train_lightgbm import (
    _atomic_write_json, _atomic_write_text, _git_ignored,
)

RELEASE = "creditlens-v3-lightgbm-v1"
REPORT = Path("docs/Stage6_Finalization_Report.md")
MODEL_CARD = Path("docs/Model_Card.md")
REFERENCES = {
    "stage5_result": Path("reports/stage5_final_results.json"),
    "stage6_shap_result": Path("reports/stage6_shap_analysis.json"),
    "strategy": Path("reports/stage6_strategy_analysis.json"),
}
SOURCE_ARTIFACTS = {
    "model": Path("models/stage5/stage5_v3_lightgbm_candidate.joblib"),
    "calibrator": Path("models/stage5/stage5_v3_probability_calibrator.joblib"),
    "lock_manifest": Path("models/stage5/stage5_v3_candidate_lock_manifest.json"),
    "validation_scores": Path("models/stage5/stage5_final_validation_scores.joblib"),
}
EXPECTED_ALERTS = {
    ("financial_history_coverage", "bureau_only", "global_cutoff_recall_drop"),
    ("age_band", "under_30", "brier_increase"),
    ("age_band", "age_50_plus", "global_cutoff_recall_drop"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": file_digest(path)}


def check_close(actual: Any, expected: Any, label: str, atol: float = 1e-6) -> None:
    if (np.shape(actual) != np.shape(expected)
            or not np.allclose(actual, expected, rtol=0.0, atol=atol, equal_nan=False)):
        raise FrozenModelError(f"기존 결과와 재현값 불일치: {label}")


def audit_alerts(strategy: dict[str, Any], X: Any, y: np.ndarray,
                 scores: np.ndarray, cutoff: float) -> list[dict[str, Any]]:
    """예고된 세 경고만 검토한다. 새 경고가 생기면 자동 동결을 거부한다."""
    observed = {(a["family"], a["group"], a["alert"])
                for a in strategy["subgroup_analysis"]["reliable_group_alerts"]}
    if observed != EXPECTED_ALERTS:
        raise FrozenModelError("검토 대상 경고가 변경됐습니다. 새 판단 기록이 필요합니다.")
    masks = {
        "bureau_only": (X.BUREAU_HAS_HISTORY.eq(1) & X.INST_HAS_HISTORY.eq(0)),
        "under_30": X.APP_AGE_YEARS.lt(30),
        "age_50_plus": X.APP_AGE_YEARS.ge(50),
    }
    explanations = {
        "bureau_only": "공통 cutoff 선택률이 낮지만 집단 내 순위 성능은 유효하다. 포착률 격차는 잔여 한계로 유지한다.",
        "under_30": "위험률을 반영한 상수 예측보다 Brier가 낮다. 전체 대비 Brier 증가만으로 재보정 필요성을 확정하지 않는다.",
        "age_50_plus": "평균 확률편향은 작지만 공통 cutoff의 낮은 포착률과 상대적으로 낮은 ROC-AUC를 개선 과제로 남긴다.",
    }
    result = []
    for family in strategy["subgroup_analysis"]["families"]:
        for group in family["groups"]:
            key = group["group"]
            if key not in masks:
                continue
            mask = masks[key].to_numpy(dtype=bool)
            metrics = evaluate_binary_metrics(y[mask], scores[mask], threshold=cutoff)
            if metrics["sample_count"] != group["rows"]:
                raise FrozenModelError("하위그룹 표본 수 불일치")
            for metric in ("roc_auc", "pr_auc", "brier_score", "ks"):
                check_close(metrics[metric], group[metric], f"{key}.{metric}")
            check_close(metrics["threshold_metrics"]["recall"],
                        group["global_cutoff"]["recall"], f"{key}.recall")
            prevalence = metrics["prevalence"]
            # 해당 집단의 관측 위험률을 모든 행에 예측하는 사후 기술통계 기준.
            constant_brier = prevalence * (1 - prevalence)
            skill = 1 - metrics["brier_score"] / constant_brier
            bias = float(scores[mask].mean() - prevalence)
            if skill <= 0 or abs(bias) >= 0.03:
                raise FrozenModelError(f"추가 개선 검토 필요: {key}")
            result.append({
                "group": key, "display_name": group["display_name"],
                "rows": metrics["sample_count"], "positive_count": metrics["positive_count"],
                "positive_rate": prevalence, "roc_auc": metrics["roc_auc"],
                "model_brier": metrics["brier_score"], "constant_prevalence_brier": constant_brier,
                "descriptive_brier_skill": skill, "mean_probability_bias": bias,
                "common_cutoff_recall": metrics["threshold_metrics"]["recall"],
                "common_cutoff_selected_fraction": metrics["threshold_metrics"]["predicted_positive_count"] / mask.sum(),
                "within_group_top10_recall_diagnostic_only": metrics["top_k_metrics"]["recall"],
                "disposition": "accepted_limitation_for_public_data_prototype",
                "reason": explanations[key],
            })
    return result


def write_bundle(bundle: Path, model: Path, calibrator: Path,
                 policy: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """기존 버전은 보존한다. 중단된 디렉터리도 명시적 검토 없이 덮어쓰지 않는다."""
    if bundle.exists():
        raise FrozenModelError("동결 디렉터리가 이미 있습니다. --verify로 확인하세요.")
    # Git 제외 검사는 부모가 있어야 저장소를 찾을 수 있다.
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if _git_ignored(bundle.parent / (bundle.name + ".joblib")) is False:
        raise FrozenModelError("모델 출력은 Git 제외 경로여야 합니다.")
    bundle.mkdir()
    for filename in ("model.joblib", "calibrator.joblib", "policy.json", "manifest.json"):
        if _git_ignored(bundle / filename) is False:
            raise FrozenModelError("동결 산출물 전체가 Git 제외되어야 합니다.")
    shutil.copyfile(model, bundle / "model.joblib")
    shutil.copyfile(calibrator, bundle / "calibrator.joblib")
    _atomic_write_json(bundle / "policy.json", policy)
    artifacts = {name: metadata(bundle / name)
                 for name in ("model.joblib", "calibrator.joblib", "policy.json")}
    for name, source in (("model.joblib", model), ("calibrator.joblib", calibrator)):
        if artifacts[name] != metadata(source):
            raise FrozenModelError("모델 복사 중 원본과 불일치")
    manifest = {**manifest, "artifacts": artifacts}
    _atomic_write_json(bundle / "manifest.json", manifest)
    return metadata(bundle / "manifest.json")


def render_report(result: dict[str, Any]) -> str:
    policy = result["policy"]
    m = result["validation_metrics"]
    t = m["threshold_metrics"]
    lines = [
        "# Stage 6 3/3 최종 모델·정책 고정 보고서", "",
        "## 결정", "",
        "V3 LightGBM(규제·행/열 표본추출)과 identity 확률 출력을 공개 데이터 프로토타입의 최종 버전으로 고정했다. "
        "Stage 5의 train 내부 튜닝·보정 비교, Stage 6의 SHAP·위험구간·하위그룹 진단을 근거로 이번 버전에는 추가 학습을 수행하지 않았다. "
        "하위그룹 격차는 해결된 것이 아니며 검토한 한계로 유지한다.", "",
        "## 경고 3건 재검토", "",
        "상수 Brier는 각 집단의 실제 위험률 p를 모두에게 예측했을 때의 p(1-p)다. "
        "Brier Skill = 1 − 모델 Brier / 상수 Brier로 비교한다. 이는 같은 validation의 사후 기술통계이며 "
        "새로운 독립 검증이나 튜닝 기준이 아니다. 집단 내 Top10%도 진단에만 사용하며 집단별 cutoff를 배포하지 않는다.", "",
        "| 집단 | 모델 Brier | 상수 Brier | Skill | 공통 cutoff Recall | 집단 내 Top10% Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in result["alert_review"]:
        lines.append(f"| {a['display_name']} | {a['model_brier']:.4f} | {a['constant_prevalence_brier']:.4f} | "
                     f"{a['descriptive_brier_skill']:.2%} | {a['common_cutoff_recall']:.2%} | "
                     f"{a['within_group_top10_recall_diagnostic_only']:.2%} |")
    lines += [""] + [f"- {a['display_name']}: {a['reason']}" for a in result["alert_review"]]
    lines += [
        "", "표본 수 기준 충족은 통계적 유의성·공정성 보장이 아니다. 신뢰구간이나 다중비교 검정을 수행하지 않았고 "
        "경고가 없는 집단까지 문제가 없다고 판정하지 않는다. 두 이력이 모두 없는 356명 등 소표본의 불확실성도 남는다.",
        "", "## 확정한 시연 정책", "",
        f"- 고위험: 점수 ≥ `{policy['high_cutoff']!r}`",
        f"- 중위험: `{policy['medium_cutoff']!r}` ≤ 점수 < `{policy['high_cutoff']!r}`",
        f"- 저위험: 점수 < `{policy['medium_cutoff']!r}`",
        "- 우선검토 여부는 고위험 경계 이상의 점수로 결정한다. 경계와 같은 점수는 모두 포함한다.",
        "- 정밀도를 유지한 점수로 판정하고 화면 표시 시에만 반올림한다.",
        "- Top 10%는 기존 비교 기준과 시연 용량 가정이다. 비용·인력 자료가 없어 사업적으로 최적이라고 주장하지 않는다.",
        "- 고정 cutoff는 새로운 배치에서 정확히 10%를 선택한다는 뜻이 아니다. "
        "Top-K는 배치 내 순위로 정해진 인원을 고르는 별도 평가 지표이며 경계 동점은 공통 평가 함수의 분수 가중치를 따른다.",
        "- 10% 검토를 20%로 넓히면 validation Recall은 35.61%→54.89%, Precision은 28.74%→22.15%다. "
        "운영 용량 최적화의 근거가 없으므로 이번 시연은 기존 10% 기준을 유지한다.",
        "", "## 고정 cutoff validation 성능", "",
        f"- ROC-AUC `{m['roc_auc']:.4f}`, PR-AUC(AP) `{m['pr_auc']:.4f}`, KS `{m['ks']:.4f}`, Gini `{m['gini']:.4f}`",
        f"- Brier `{m['brier_score']:.4f}`, F1 `{t['f1']:.4f}`",
        f"- 우선검토 {t['predicted_positive_count']:,}명, TP {t['true_positive']:,}, FP {t['false_positive']:,}, "
        f"FN {t['false_negative']:,}, TN {t['true_negative']:,}",
        f"- Recall `{t['recall']:.2%}`, Precision `{t['precision']:.2%}`",
        "- 검증 데이터는 개발 중 여러 번 분석되었으므로 독립 최종 성능으로 해석하지 않는다.",
        "", "## 동결 산출물과 재현", "",
        "`models/stage6/frozen_v1/`에 모델(학습된 전처리 포함), 보정기, 정책 JSON, manifest를 저장했다. "
        "원본 모델을 바이트 단위로 복사했고 validation에서 동일 예측과 경계 판정을 확인했다. "
        "manifest는 입력 피처 순서, 420개 변환 피처, 데이터 해시, 모델 설정, 실행환경과 추론 코드 해시를 기록한다.",
        "", "```bash", "PYTHONPATH=src .venv/bin/python -m creditlens.modeling.finalize_stage6 --verify", "```", "",
        "검증은 Git 관리 결과 JSON의 manifest 해시에서 시작하며 모델 역직렬화보다 먼저 수행한다. "
        "기존 동결 디렉터리를 덮어쓰지 않는다. 해시/추론 코드/버전이 달라지면 검증 실패로 중단한다. "
        "검증은 파일 변경 검출이며 디지털 서명이나 OS 쓰기 방지 장치는 아니다.",
        "", "## 다음 단계 및 개선 정책", "",
        "- Stage 7: 공통 동결 모델 로더와 점수 정책을 이용해 Streamlit·FastAPI를 구현한다.",
        "- Stage 8 구현 항목 3: 동일 동결 산출물에 내부 holdout test를 한 번 평가한다. "
        "이는 정답이 없는 Kaggle application_test.csv가 아니라 application_train에서 분리한 46,126명이다.",
        "- test 결과로 재튜닝하지 않는다. 이후 개선은 별도 버전으로 관리하고 새 독립 평가가 필요하다.",
        "- 이번 버전의 다음 개선 후보는 50세 이상·납부이력 부재 고객의 누락 패턴과 추가 피처다. "
        "효과가 입증되었다고 보지 않으며 train 내부 검증으로 먼저 비교해야 한다.", "",
        "[모델 카드](Model_Card.md) · [집계 JSON](../reports/stage6_final_results.json)", "",
    ]
    return "\n".join(lines)


def render_model_card(result: dict[str, Any]) -> str:
    m, p = result["validation_metrics"], result["policy"]
    return f"""# CreditLens 모델 카드

## 모델과 목적

- 버전: `{RELEASE}`
- 모델: V3 LightGBM 이진분류, 500 trees, 학습된 전처리 포함
- 데이터: Kaggle Home Credit Default Risk 공개 익명 금융 데이터
- 예측 대상: 대회 정의의 TARGET=1(상환곤란) 확률 추정
- 용도: 공개 데이터 분석과 우선검토 순서 시연
- 최종 상태: 프로토타입용 모델·정책 고정, 독립 test 평가는 아직 미실시

## 입력과 학습

신청정보·외부 신용·과거 납부이력을 고객 단위로 집계한 V3 피처 198개(수치 184, 범주 14)를 사용한다.
고객 ID, TARGET, SPLIT, CODE_GENDER는 모델 입력에서 제외한다. 누락된 이력의 0과 결측을 구분한다.
필드 의미와 산식은 [피처 사전](Feature_Dictionary.md), 학습·전처리는 [Stage 4 명세](Stage4_Preprocessing_and_Evaluation_Spec.md)를 따른다.

seed 42의 고객 단위 층화 70/15/15 분할이다. train 215,258명으로 전처리와 모델을 학습했다.
제한 튜닝과 확률 보정 방법 비교는 train 내부에서 수행했다. identity가 선택되어 추가 확률 변환 없이 원 출력을 유지한다.
validation 46,127명은 모델 비교·설명·정책 선택에 사용했다. test 46,126명은 Stage 8까지 봉인한다.
train+validation 합본 재학습은 하지 않는다.

## validation 결과와 정책

| ROC-AUC | PR-AUC(AP) | KS | Gini | Brier | Recall@10% | Lift@10% |
|---:|---:|---:|---:|---:|---:|---:|
| {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['ks']:.4f} | {m['gini']:.4f} | {m['brier_score']:.4f} | {m['top_k_metrics']['recall']:.4f} | {m['top_k_metrics']['lift']:.4f} |

고위험 및 우선검토 경계는 `{p['high_cutoff']!r}`, 중위험 하한은 `{p['medium_cutoff']!r}`다.
경계는 이상(≥) 비교로 포함하며 표시 반올림 전 점수를 사용한다. 고정 cutoff의 선택 비율은 배치마다 달라진다.
고·중·저 구간의 validation 상환곤란 비율은 28.74%·12.96%·3.72%다.
저위험은 무위험을 뜻하지 않는다. 우선검토 여부는 승인·거절 여부가 아니다.

## 설명과 한계

SHAP은 모델 raw log-odds를 설명한다. 양수는 모델 점수를 높인 기여이며 확률의 %p나 인과효과가 아니다.
외부 신용평가값 평균이 가장 큰 전역 중요도를 보였다. 상관된 피처들은 중요도를 나눠 갖는다.
고객별 설명은 로컬에서만 보관하고 공유 보고서에는 집계만 둔다.

외부 신용이력만 있는 고객과 50세 이상 고객의 공통 cutoff Recall이 낮다.
30세 미만의 Brier는 전체보다 크지만 집단 위험률 상수 기준보다 작다.
세 경고는 해결된 것으로 표시하지 않고 프로토타입의 잔여 한계로 수용했다.
성별 기록 제외만으로 공정성이 보장되지 않는다. 표본 수 기준은 유의성 검정이 아니며 소표본은 불확실하다.
validation을 반복 사용했으므로 이 수치는 독립 최종 성능이 아니다.
해외 과거 자료여서 국내 금융환경과 미래 시점 성능을 보장하지 못한다.
신뢰할 수 있는 신청 기준일 부재로 시점 외 검증도 수행하지 못했다.

## 사용 범위와 관리

실제 금융 의사결정, 자동 대출 승인·거절, 공식 신용등급 산출에는 사용할 수 없다.
프로토타입에서 심사자가 검토 순서를 이해하는 보조 정보로 사용한다. 집단별 별도 cutoff는 적용하지 않는다.
Stage 7에서 입력 결측·오류, 점수 분포, PSI·CSI, 정답 확보 후 성능 모니터링을 구현·문서화할 예정이다.
Stage 8에서 동결 파일에 내부 test를 한 번 평가하며 결과로 재튜닝하지 않는다.

## 재현과 근거

동결 디렉터리: `models/stage6/frozen_v1/` (Git 제외)

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.modeling.finalize_stage6 --verify
```

[최종 판단 보고서](Stage6_Finalization_Report.md) · [동결 결과](../reports/stage6_final_results.json) ·
[모델 선정](Stage5_Final_Model_Selection_Report.md) · [SHAP](Stage6_SHAP_Analysis_Report.md) ·
[위험전략·하위그룹](Stage6_Risk_Strategy_and_Subgroup_Report.md)
"""


def finalize_stage6() -> dict[str, Any]:
    for path in (DEFAULT_BUNDLE, DEFAULT_RESULT, REPORT, MODEL_CARD):
        if path.exists():
            raise FrozenModelError(f"기존 결과를 덮어쓰지 않습니다: {path}. --verify를 사용하세요.")
    stage5, shap, strategy = [read_json(REFERENCES[k])
                              for k in ("stage5_result", "stage6_shap_result", "strategy")]
    verified = _validate_stage5_reference(
        stage5, model_path=SOURCE_ARTIFACTS["model"],
        calibrator_path=SOURCE_ARTIFACTS["calibrator"],
        lock_path=SOURCE_ARTIFACTS["lock_manifest"],
        validation_scores_path=SOURCE_ARTIFACTS["validation_scores"],
    )
    _validate_stage6_shap_reference(shap, stage5_path=REFERENCES["stage5_result"],
                                    verified_stage5=verified)
    if strategy.get("run_status") != "complete" or strategy.get("stage_part") != "2/3":
        raise FrozenModelError("완료된 2/3 결과가 필요합니다.")
    for payload in (stage5, shap, strategy):
        scope = payload["data_scope"]
        if scope["test_feature_rows_used"] != 0 or scope["test_predictions_created"] is not False:
            raise FrozenModelError("test 봉인 계약 위반")
    for key in ("stage5_result", "stage6_shap_result"):
        if strategy["references"][key]["sha256"] != file_digest(REFERENCES[key]):
            raise FrozenModelError("분석 보고서 참조 변경")
    for key in SOURCE_ARTIFACTS:
        if strategy["references"][key]["sha256"] != verified[key]["sha256"]:
            raise FrozenModelError("분석 모델 참조 변경")
    pipeline = joblib.load(SOURCE_ARTIFACTS["model"])
    calibrator = joblib.load(SOURCE_ARTIFACTS["calibrator"])
    pipeline, model, features, numeric, categorical = _extract_pipeline_contract(pipeline)
    _resolve_source_groups(features, stage5["data_version"]["feature_groups"])
    if calibrator.method != "identity" or stage5["calibration"]["decision"]["selected_method"] != "identity":
        raise FrozenModelError("보정기 계약 변경")
    if file_digest(DEFAULT_MART_PATHS["v3"]) != stage5["data_version"]["parquet_sha256"]:
        raise FrozenModelError("Stage 5 데이터 checksum 불일치")
    print("기존 모델·분석 참조 검증 완료. validation 재현 확인", flush=True)
    validation = load_model_split(DEFAULT_MART_PATHS["v3"], "v3", "validation")
    if tuple(validation.X.columns) != features:
        raise FrozenModelError("피처 순서 불일치")
    y = validation.y.to_numpy(dtype=np.int8)
    scores = calibrator.predict(pipeline.predict_proba(validation.X)[:, 1])
    stored = joblib.load(SOURCE_ARTIFACTS["validation_scores"])
    if (stored.get("split") != "validation" or stored.get("customer_ids_included") is not False
            or not np.array_equal(y, stored["y_true"])):
        raise FrozenModelError("validation 저장 행 계약 불일치")
    for score_type in ("raw", "calibrated"):
        check_close(scores, stored["scores"][score_type], "저장 점수")
    for prior in (stage5, shap, strategy):
        if len(y) != prior["data_scope"]["validation_rows"]:
            raise FrozenModelError("validation 행 수 불일치")
    boundaries = strategy["risk_strategy"]["risk_bands"]["policy"]
    policy = {
        "schema_version": "1.0", "status": "frozen", "release": RELEASE,
        "medium_cutoff": boundaries["medium_cutoff"],
        "high_cutoff": boundaries["high_cutoff"],
        "review_cutoff": boundaries["high_cutoff"], "comparison": ">=",
        "calibration_method": "identity", "group_specific_cutoffs": False,
        "automatic_credit_decision": False,
        "purpose": "public_data_review_priority_prototype",
        "source": "validation_top10_and_top30_score_boundaries",
        "new_batch_selection_fraction_is_fixed": False,
        "round_before_comparison": False,
    }
    classified = apply_risk_policy(scores, policy)
    metrics = evaluate_binary_metrics(y, scores, threshold=policy["review_cutoff"])
    for metric in ("roc_auc", "pr_auc", "ks", "gini", "brier_score"):
        check_close(metrics[metric], strategy["validation_replay"]["metrics"][metric], metric)
    for fraction, key in ((0.1, "high_cutoff"), (0.3, "medium_cutoff")):
        boundary = evaluate_binary_metrics(y, scores, top_fraction=fraction)["top_k_metrics"]["cutoff_score"]
        check_close(boundary, policy[key], key, atol=0)
    rates = []
    for band in strategy["risk_strategy"]["risk_bands"]["bands"]:
        mask = classified["risk_band"] == band["band"]
        if mask.sum() != band["rows"] or y[mask].sum() != band["positive_count"]:
            raise FrozenModelError("위험구간 집계 재현 불일치")
        rates.append(float(y[mask].mean()))
    if not rates[0] > rates[1] > rates[2]:
        raise FrozenModelError("위험구간 순서 재검토 필요")
    review = audit_alerts(strategy, validation.X, y, scores, policy["review_cutoff"])
    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": "1.0", "release": RELEASE, "status": "frozen", "frozen_at_utc": now,
        "runtime": runtime_versions(),
        "runtime_sources": {path: file_digest(Path(path)) for path in RUNTIME_SOURCES},
        "model_settings": model.get_params(),
        "input_contract": {"features": list(features), "numeric": list(numeric),
                           "categorical": list(categorical),
                           "transformed_features": list(transformed_feature_names(pipeline.named_steps["preprocessor"]))},
        "data_version": stage5["data_version"],
        "test_feature_rows_used": 0,
        "references": {key: {"path": str(path), **metadata(path)} for key, path in REFERENCES.items()},
    }
    print("경고 재검토 완료. 기존 모델과 프로토타입 cutoff 동결", flush=True)
    manifest_metadata = write_bundle(DEFAULT_BUNDLE, SOURCE_ARTIFACTS["model"],
                                     SOURCE_ARTIFACTS["calibrator"], policy, manifest)
    result = {
        "schema_version": "1.0", "run_status": "complete", "stage_part": "3/3",
        "release": RELEASE, "frozen_at_utc": now,
        "data_scope": {"validation_rows": len(y), "test_feature_rows_used": 0,
                       "test_predictions_created": False, "model_refit": False,
                       "customer_ids_in_shared_outputs": False, "row_level_values_in_shared_outputs": False},
        "decision": {"model": "retain_stage5_lightgbm", "calibration": "identity",
                     "new_training": False, "alerts_resolved": False,
                     "alert_disposition": "reviewed_and_accepted_for_prototype_only",
                     "cost_optimal_policy_claimed": False},
        "policy": policy, "alert_review": review, "validation_metrics": metrics,
        "risk_bands": strategy["risk_strategy"]["risk_bands"]["bands"],
        "bundle_manifest": {"path": str(DEFAULT_BUNDLE / "manifest.json"), **manifest_metadata},
        "runtime": manifest["runtime"], "references": manifest["references"],
        "artifacts": read_json(DEFAULT_BUNDLE / "manifest.json")["artifacts"],
        "next_stage": 7, "test_evaluation_stage": 8,
    }
    # 공유 결과는 완료 마커다. 먼저 복사한 모델의 예측을 확인한다.
    copied_model = joblib.load(DEFAULT_BUNDLE / "model.joblib")
    copied_calibrator = joblib.load(DEFAULT_BUNDLE / "calibrator.joblib")
    copy_scores = copied_calibrator.predict(copied_model.predict_proba(validation.X)[:, 1])
    check_close(copy_scores, scores, "복사 후 validation 점수", atol=0)
    result["verification"] = {"copied_validation_predictions_identical": True,
                              "test_loaded": False, "validation_metric_replay": True}
    _atomic_write_text(REPORT, render_report(result))
    _atomic_write_text(MODEL_CARD, render_model_card(result))
    result["documents"] = {str(p): metadata(p) for p in (REPORT, MODEL_CARD)}
    _atomic_write_json(DEFAULT_RESULT, result)
    verify_bundle()
    print("Stage 6 완료: 동결 검증 통과, test 0행", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="데이터 조회 없이 기존 동결 검증")
    args = parser.parse_args()
    if args.verify:
        verify_bundle()
        print("동결 모델·정책·추론 코드·환경 검증 PASS (데이터 조회 없음)")
    else:
        finalize_stage6()


if __name__ == "__main__":
    main()
