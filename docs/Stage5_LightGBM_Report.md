# Stage 5 1/3 LightGBM 비교 보고서

> 같은 고정 설정의 LightGBM을 V1·V2·V3 train으로 학습하고 같은 validation에서 평가한 결과입니다. test는 사용하지 않았습니다.
>
> 이 문서는 Stage 5 1/3 당시의 비교 기록입니다. 후속 튜닝과 최종 후보 선정 결과는 [Stage 5 3/3 보고서](Stage5_Final_Model_Selection_Report.md)에 정리했습니다.

## 왜 이 실험을 했는가

LightGBM은 표 형태 데이터의 비선형 관계와 변수 사이 상호작용을 학습하는 트리 기반 모델입니다. 같은 LightGBM에 아래 데이터를 차례로 추가해 모델 변화와 데이터 추가 효과를 분리했습니다.

- V1: 대출 신청정보
- V2: V1 + 외부 신용이력
- V3: V2 + 과거 납부이력

이번 결과는 튜닝 전 기준 성능입니다. 세 버전 모두 같은 설정을 사용했고 validation을 학습, early stopping 또는 설정 선택에 사용하지 않았습니다.

## LightGBM validation 결과

| 데이터 | ROC-AUC | PR-AUC(AP) | KS | Gini | Brier | Recall@10% | Lift@10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0.7621 | 0.2448 | 0.3938 | 0.5242 | 0.0677 | 0.3410 | 3.4101 |
| V2 | 0.7680 | 0.2584 | 0.4030 | 0.5359 | 0.0671 | 0.3510 | 3.5094 |
| V3 | 0.7752 | 0.2665 | 0.4181 | 0.5504 | 0.0667 | 0.3596 | 3.5954 |

PR-AUC의 무작위 기준은 validation의 상환곤란 비율입니다. Brier Score는 낮을수록 좋고 나머지 표의 주요 지표는 높을수록 좋습니다.

## 같은 LightGBM에서 데이터 추가 효과

| 비교 | Δ ROC-AUC | Δ PR-AUC | Δ KS | Δ Brier | Δ Recall@10% |
|---|---:|---:|---:|---:|---:|
| V2 - V1: 외부 신용이력 추가 | +0.0059 | +0.0136 | +0.0092 | -0.0005 | +0.0099 |
| V3 - V2: 납부이력 추가 | +0.0072 | +0.0081 | +0.0151 | -0.0005 | +0.0086 |
| V3 - V1: 두 정보 원천 추가 | +0.0131 | +0.0217 | +0.0243 | -0.0010 | +0.0185 |

## 기존 모델과의 같은 데이터 비교

| 비교 | Δ ROC-AUC | Δ PR-AUC | Δ KS | Δ Brier | Δ Recall@10% |
|---|---:|---:|---:|---:|---:|
| V1 LightGBM - V1 Logistic | +0.0185 | +0.0179 | +0.0282 | -0.0010 | +0.0250 |
| V2 LightGBM - V2 Logistic | +0.0190 | +0.0261 | +0.0280 | -0.0012 | +0.0252 |
| V3 LightGBM - V3 Logistic | +0.0167 | +0.0241 | +0.0273 | -0.0011 | +0.0220 |
| V3 LightGBM - V3 Random Forest | +0.0245 | +0.0332 | +0.0375 | 비교 불가 | +0.0320 |

V3 Random Forest는 `balanced_subsample` 가중치를 사용해 현재 점수를 보정된 실제 확률로 해석할 수 없습니다. 따라서 LightGBM과 Random Forest의 Brier 차이는 확률 품질의 공정 비교에서 제외했습니다.

## 현재 해석

- LightGBM 세 버전 중 validation PR-AUC가 가장 높은 데이터는 V3이며 값은 0.2665입니다.
- V3 LightGBM의 ROC-AUC/PR-AUC는 0.7752/0.2665, V3 Logistic은 0.7585/0.2424입니다.
- 이것은 1/3 시점의 고정 설정 validation 비교 결과이며, 이 결과만으로 최종 후보를 선정하지 않았습니다.
- validation 결과를 보고 LightGBM 설정을 다시 바꾸지 않았습니다. 제한 튜닝은 Stage 5 3/3에서 train 내부 데이터로만 수행했습니다.

## 고정한 조건과 데이터 감사

- LightGBM: 4.7.0
- 트리 수: 500
- learning rate: 0.05
- 재현 seed: 42
- 불균형 가중치: 사용하지 않음
- early stopping: 사용하지 않음
- 전처리: train에서만 median 대치·결측 플래그·희소범주·원-핫 인코딩 학습
- train 고객: 215,258명
- validation 고객: 46,127명
- test 피처 사용: 0행
- test 예측·평가: 없음
- 공유 JSON·Markdown에 고객 ID와 행별 예측값을 저장하지 않음

## 후속 결과

Stage 5 2/3의 V3 TensorFlow MLP 비교와 3/3의 train 내부 제한 튜닝·확률 보정·피처군 분석을 완료했습니다. 최종 비교와 Stage 6 전달 후보는 [Stage 5 3/3 보고서](Stage5_Final_Model_Selection_Report.md)에서 확인할 수 있으며, test는 계속 봉인했습니다.
