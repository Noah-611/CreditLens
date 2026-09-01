# Stage 4 기준 모델 학습 보고서

> train으로만 모델과 전처리를 학습하고 validation으로 비교한 실제 실행 결과입니다. test 피처·예측·평가는 사용하지 않았습니다.

## 실험 목적

- Dummy Prior로 아무 피처도 사용하지 않는 최저 기준을 확인합니다.
- V1·V2·V3 Logistic Regression을 비교해 외부 신용이력과 납부이력의 추가 가치를 측정합니다.
- V3 Random Forest를 제한된 복잡도의 비선형 기준 모델로 비교합니다.
- 이번 단계에서는 하이퍼파라미터 탐색, cutoff 선택, 확률 보정과 test 평가를 수행하지 않습니다.

## Validation 결과

| 실험 | ROC-AUC | PR-AUC(AP) | KS | Gini | Brier | Recall@10% | Lift@10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy Prior | 0.5000 | 0.0807 | 0.0000 | 0.0000 | 0.0742 | 0.1000 | 1.0000 |
| V1 Logistic Regression | 0.7436 | 0.2269 | 0.3656 | 0.4871 | 0.0686 | 0.3161 | 3.1604 |
| V2 Logistic Regression | 0.7490 | 0.2324 | 0.3750 | 0.4980 | 0.0684 | 0.3257 | 3.2570 |
| V3 Logistic Regression | 0.7585 | 0.2424 | 0.3908 | 0.5170 | 0.0678 | 0.3375 | 3.3752 |
| V3 Random Forest | 0.7507 | 0.2333 | 0.3806 | 0.5013 | 0.1820 | 0.3276 | 3.2758 |

PR-AUC의 무작위 기준선은 validation의 TARGET=1 비율입니다. Gini는 `2 × ROC-AUC - 1`이며 독립 지표로 과장하지 않습니다.

## 데이터 버전별 추가 가치

| 비교 | Δ ROC-AUC | Δ PR-AUC | Δ KS | Δ Brier |
|---|---:|---:|---:|---:|
| V2 Logistic - V1 Logistic | 0.0054 | 0.0055 | 0.0094 | -0.0003 |
| V3 Logistic - V2 Logistic | 0.0095 | 0.0100 | 0.0157 | -0.0006 |

Brier Score는 낮을수록 좋으므로 음수 변화가 개선입니다. 위 차이는 동일한 Logistic Regression에서 데이터 원천만 추가한 결과입니다.

## 고정한 학습 조건

- 재현 seed: `42`
- 분류 임계값: `0.5`
- Top-K 비율: `0.1`
- Logistic Regression은 확률 기준선을 확인하기 위해 class weight 없이 학습했습니다.
- Random Forest는 제한된 깊이와 트리 수, `balanced_subsample`을 사용했습니다.
- 각 데이터 버전의 전처리기는 해당 train에만 fit했습니다.
- 선형 수치 피처는 `StandardScaler(with_mean=False)`로 크기를 맞췄고 결측 indicator는 0/1을 유지했습니다.

## 수렴과 실행 자원

| 실험 | 반복 횟수 | 최대 반복 | 수렴 | 학습시간(초) |
|---|---:|---:|---|---:|
| Dummy Prior | - | - | 해당 없음 | 0.020 |
| V1 Logistic Regression | 255 | 600 | 예 | 46.746 |
| V2 Logistic Regression | 283 | 600 | 예 | 66.220 |
| V3 Logistic Regression | 421 | 600 | 예 | 112.749 |
| V3 Random Forest | - | - | 해당 없음 | 251.544 |

- 전체 실행시간: 494.009초
- 프로세스 최대 RSS: 3124.168 MiB

초기 검증에서는 사분위 범위가 0인 희소 금액 피처가 `RobustScaler`에서 수천만 단위로 남아 V2·V3 Logistic 최적화를 방해했습니다. 모든 비상수 수치 피처를 표준편차 기준으로 맞추는 현재 방식으로 수정한 뒤 V1→V2→V3의 일관된 비교 결과를 얻었습니다.

## 데이터 사용 감사

- train 고객: 215,258명
- validation 고객: 46,127명
- test 피처 사용: 0행
- test 예측·평가: 없음
- 고객 ID와 행별 예측값은 공유용 JSON·Markdown에 저장하지 않았습니다.

## 해석 범위

이 결과는 튜닝 전 기준 성능입니다. ROC·PR·Calibration 곡선, 위험도 decile, Top-K 상세 시나리오와 모델 선택 해석은 다음 분석 단계에서 작성합니다. Random Forest는 class weight를 사용했으므로 Brier Score를 보정된 확률 품질로 해석하지 않습니다.
