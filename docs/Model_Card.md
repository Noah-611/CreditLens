# CreditLens 모델 카드

## 모델과 목적

- 버전: `creditlens-v3-lightgbm-v1`
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
| 0.7765 | 0.2699 | 0.4174 | 0.5530 | 0.0665 | 0.3561 | 3.5605 |

고위험 및 우선검토 경계는 `0.18607729049496233`, 중위험 하한은 `0.08251815922901691`다.
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
