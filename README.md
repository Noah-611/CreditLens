# CreditLens

CreditLens는 Kaggle의 공개 익명 금융 데이터를 활용하여 대출 신청자의 상환곤란 위험을 분석하고 예측하는 머신러닝 프로젝트입니다.

대출 신청정보, 외부 신용거래와 과거 납부기록을 고객 단위로 통합하고 금융 파생변수를 생성합니다. Logistic Regression, Random Forest, LightGBM과 TensorFlow MLP를 동일한 데이터 분할과 평가 기준으로 비교하고, 최종 모델의 주요 예측요인을 SHAP으로 분석합니다. 학습이 완료된 모델은 Streamlit 화면과 FastAPI 예측 API에서 사용할 수 있도록 구성합니다.

## 현재 상태

- Stage 0: 프로젝트 초기 환경 구성 완료
- Stage 1: 데이터 확보 및 무결성 확인 완료 (`PASS`, 오류 0건)
- Stage 2: 고객 분할·train-only EDA·전처리 및 누수 정책 설계 완료
- Stage 3: SQL·Python 고객 분석 마트와 V1·V2·V3 구축 완료
- Stage 4: 기준 모델 5개 학습과 ROC·PR·Calibration·Decile·Top-K validation 분석 완료
- 다음 작업: Stage 5 LightGBM·TensorFlow MLP 학습과 데이터·모델 비교
- 모델은 train으로만 학습했고 test 데이터는 계속 봉인합니다.

전체 범위와 Stage별 완료 조건은 [프로젝트 계획서](docs/Project_Plan.md)를 참고하세요.
Stage 4의 전처리·평가 기준은 [Stage 4 전처리·평가 명세](docs/Stage4_Preprocessing_and_Evaluation_Spec.md)에 기록되어 있습니다.
기준 모델의 실제 실행 결과는 [Stage 4 기준 모델 학습 보고서](docs/Stage4_Baseline_Model_Report.md)에서 확인할 수 있습니다.
곡선·위험구간·심사 용량별 해석은 [Stage 4 Validation 상세 분석 보고서](docs/Stage4_Validation_Analysis_Report.md)에 기록되어 있습니다.

## 핵심 분석 목표

- 약 1,560만 건의 관계형 금융 데이터 품질·분포 분석
- SQL·Python 기반 고객 단위 V1·V2·V3 분석 마트 구축
- Logistic Regression·Random Forest·LightGBM·TensorFlow MLP 모델 학습과 비교
- 외부 신용정보와 납부이력의 추가 예측 가치 분석
- KS·Gini·Lift·Calibration과 위험구간별 상환곤란 비율 검증
- SHAP 기반 주요 위험요인과 우선검토 시나리오 분석
- 데이터 계약, 배치 스코어링(batch scoring), PSI·CSI와 품질 모니터링 설계
- 고정된 모델을 사용하는 Streamlit 프로그램과 FastAPI `/predict` API

## 사용 범위

CreditLens는 공개 데이터 기반의 분석 프로젝트입니다. 모델 결과는 실제 대출 승인·거절이나 공식 신용평가에 사용하지 않습니다.

## 저장소 구조

```text
CreditLens/
├── data/
│   ├── raw/                  # Kaggle 원본 데이터 (Git 제외)
│   ├── interim/              # 정제·집계 중간 데이터 (Git 제외)
│   └── processed/            # 고객 단위 가공 데이터 (Git 제외)
├── docs/
│   ├── Data_Download_Guide.md
│   ├── Data_Dictionary.md
│   ├── Data_Split_Spec.md
│   ├── Data_Validation_Report.md
│   ├── Analysis_Mart_Spec.md
│   ├── Feature_Dictionary.md
│   ├── Preprocessing_and_Leakage_Spec.md
│   ├── Stage4_Preprocessing_and_Evaluation_Spec.md
│   ├── Stage4_Baseline_Model_Report.md
│   ├── Stage4_Validation_Analysis_Report.md
│   ├── Stage2_EDA_Report.md
│   ├── Stage3_Build_Report.md
│   └── Project_Plan.md
├── models/                   # 학습 모델과 전처리 산출물 (Git 제외)
├── notebooks/
│   └── 01_stage2_eda.ipynb   # train-only EDA 실행 진입점
├── reports/
│   ├── figures/              # Stage 2 EDA·Stage 4 모델 평가 시각화
│   ├── data_validation.json
│   ├── stage2_eda.json
│   ├── stage2_split_summary.json
│   ├── stage3_build_summary.json
│   ├── stage3_feature_profile.json
│   ├── stage4_baseline_results.json
│   └── stage4_validation_analysis.json
├── sql/
│   └── stage3/              # V1·bureau·V2·installments·V3 DuckDB SQL
├── src/
│   └── creditlens/
│       ├── analysis/         # EDA·피처 분석·validation 모델 진단
│       ├── data/             # 원본 검증·분할·고객 마트 구축
│       ├── evaluation/       # 공통 이진분류 평가 지표
│       └── modeling/         # 모델 입력 계약·로더·전처리 파이프라인
├── tests/                    # 합성 데이터 기반 자동 테스트
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt          # 데이터 처리·전통적 ML 핵심 의존성
└── requirements-stage5-7.txt # Stage 5·7에서 설치할 MLP·시연 의존성
```

빈 디렉터리를 저장소에 유지하기 위해 `.gitkeep` 파일만 추적합니다.

## 로컬 환경 설정

WSL2 Ubuntu 기준 예시입니다.

```bash
cd CreditLens
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow MLP를 학습하는 Stage 5와 Streamlit·FastAPI를 구현하는 Stage 7에서는 후반 의존성도 설치합니다.

```bash
python -m pip install -r requirements-stage5-7.txt
```

설치 확인:

```bash
python --version
python -m pip check
```

`requirements-stage5-7.txt`에는 설치 용량이 큰 TensorFlow와 Streamlit·FastAPI 관련 의존성을 분리해 두었습니다.

## 데이터 준비

데이터는 Kaggle의 [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)를 사용합니다. Kaggle 대회 약관에 동의한 뒤 다음 핵심 파일을 `data/raw/`에 둡니다.

- `application_train.csv`
- `bureau.csv`
- `installments_payments.csv`
- `HomeCredit_columns_description.csv`

Kaggle CLI를 사용할 경우 인증정보인 `kaggle.json`은 저장소에 복사하거나 커밋하지 마세요. 전체 절차와 고정 원본 checksum은 [데이터 다운로드 가이드](docs/Data_Download_Guide.md)에 기록되어 있습니다.

## Stage 1 원본 데이터 검증

청크 기반 전체 검증을 재실행합니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.data.validate_raw \
  --raw-dir data/raw
```

현재 검증 결과:

| 테이블 | 데이터 행 수 | 컬럼 수 | 핵심 키 결과 |
|---|---:|---:|---|
| `application_train.csv` | 307,511 | 122 | `SK_ID_CURR` 결측·중복 0 |
| `bureau.csv` | 1,716,428 | 17 | `SK_ID_BUREAU` 결측·중복 0 |
| `installments_payments.csv` | 13,605,401 | 8 | `SK_ID_PREV → SK_ID_CURR` 위반 0 |

- 최종 상태: `PASS`
- 오류: 0건
- 경고: 43건(고결측 컬럼 등 Stage 2 검토 대상)
- `TARGET=1`: 24,825건(8.0729%)

세부 결과는 [데이터 검증 보고서](docs/Data_Validation_Report.md), 전체 컬럼 정보는 [데이터 사전](docs/Data_Dictionary.md)에서 확인할 수 있습니다. 기계 판독용 원본 결과는 `reports/data_validation.json`에 저장됩니다.

자동 테스트:

```bash
.venv/bin/python -m pytest -q
```

Stage 1 범위의 합성 데이터 기반 회귀 테스트 11개가 통과합니다.

## Stage 2 고객 분할과 EDA

고객을 `TARGET` 비율이 유지되도록 seed `42`로 분할합니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.data.split_data
```

| split | 고객 수 | TARGET=1 | TARGET=1 비율 |
|---|---:|---:|---:|
| train | 215,258 | 17,377 | 8.0726% |
| validation | 46,127 | 3,724 | 8.0734% |
| test | 46,126 | 3,724 | 8.0735% |

고객별 배정표 `data/interim/customer_splits.csv`는 Git에서 제외됩니다. 고객 ID가 없는 집계 결과만 `reports/stage2_split_summary.json`으로 공유합니다. 자세한 규칙은 [데이터 분할 명세](docs/Data_Split_Spec.md)를 참고하세요.

학습 파티션만 사용하는 EDA를 재실행합니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.analysis.stage2_eda
```

주요 관찰 결과:

- train 215,258명, 피처 120개
- 수치형 104개, 범주형 16개
- 결측이 있는 피처 67개, 결측률 40% 이상 피처 49개
- `DAYS_EMPLOYED=365243` sentinel 38,564건(17.92%)
- `EXT_SOURCE_1~3`이 `TARGET`과 가장 큰 단변량 수치 상관을 보이지만 인과관계로 해석하지 않음
- 검증·테스트 고객의 피처 사용 0건, 고객 ID 출력 0건

상세 결과는 [Stage 2 EDA 보고서](docs/Stage2_EDA_Report.md), 처리 원칙은 [전처리·누수 점검 명세](docs/Preprocessing_and_Leakage_Spec.md)에 기록되어 있습니다. Stage 2에서는 실제 결측 대치, 인코딩, 피처 생성이나 모델 학습을 수행하지 않았습니다.

전체 자동 테스트:

```bash
.venv/bin/python -m pytest -q
```

Stage 2 완료 시점까지 합성 회귀 테스트 34개가 통과했습니다.

## Stage 3 고객 분석 마트

DuckDB SQL로 일대다 이력을 먼저 집계한 뒤 고객 신청정보와 결합합니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.data.build_feature_mart
```

| 버전 | 구성 | 고객 수 | 전체 컬럼 | 후보 피처 | 모델 입력 후보 |
|---|---|---:|---:|---:|---:|
| V1 | 신청정보 + 신청 파생값 | 307,511 | 136 | 133 | 132 |
| V2 | V1 + 외부 신용이력 | 307,511 | 173 | 170 | 169 |
| V3 | V2 + 과거 납부행동 | 307,511 | 202 | 199 | 198 |

`SK_ID_CURR`, `TARGET`, `SPLIT`은 후보 피처 수에서 제외합니다. `CODE_GENDER`는 분석·공정성 점검용으로 마트에 보존하되 모델 입력에서는 제외하므로 모델 입력 후보는 한 개 더 적습니다.

주요 구축 결과:

- bureau 신청 이후 갱신 행 17건 제외, 263,490명의 1,465,308행 집계
- 납부 이벤트 11,591,592건을 납부회차 11,026,627건으로 먼저 합친 뒤 291,643명 단위로 집계
- V1→V2와 V2→V3 공통 컬럼 불일치 0건
- 고객 키·`TARGET`·split 보존, 조인 증식과 피처 계약 위반 0건
- 전체 빌드 87.684초, 프로세스 최대 RSS 약 2,300MiB

새 파생 피처의 통계는 train 고객만 사용해 생성합니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.analysis.stage3_feature_profile
```

신청 파생 13개, bureau 37개, installments 29개 등 총 79개 피처를 분석하며 validation·test 피처 행은 사용하지 않습니다. 상세한 구조와 산식은 [분석 마트 명세](docs/Analysis_Mart_Spec.md), [피처 사전](docs/Feature_Dictionary.md), 실제 결과는 [Stage 3 구축 보고서](docs/Stage3_Build_Report.md)를 참고하세요.

전체 자동 테스트:

```bash
.venv/bin/python -m pytest -q
```

Stage 3 완료 시점에는 합성 데이터 기반 회귀 테스트 48개가 통과했습니다.

## Stage 4 기준 모델 비교

동일한 train·validation 분할과 평가 기준으로 5개 기준 모델을 비교했습니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.modeling.train_baselines \
  --random-forest-jobs 2
```

| 모델 | ROC-AUC | PR-AUC | KS | Brier | Recall@Top10% | Lift@Top10% |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Prior | 0.5000 | 0.0807 | 0.0000 | 0.0742 | 0.1000 | 1.0000 |
| V1 Logistic Regression | 0.7436 | 0.2269 | 0.3656 | 0.0686 | 0.3161 | 3.1604 |
| V2 Logistic Regression | 0.7490 | 0.2324 | 0.3750 | 0.0684 | 0.3257 | 3.2570 |
| V3 Logistic Regression | 0.7585 | 0.2424 | 0.3908 | 0.0678 | 0.3375 | 3.3752 |
| V3 Random Forest | 0.7507 | 0.2333 | 0.3806 | 0.1820 | 0.3276 | 3.2758 |

동일한 Logistic Regression에서 V1→V2→V3로 데이터 원천을 추가할수록 validation 성능이 개선됐습니다. 현재 튜닝 전 기준 모델 중 V3 Logistic Regression이 가장 좋지만, 최종 모델 선택은 후속 모델 비교와 확률 보정을 마친 뒤 진행합니다. Random Forest는 클래스 가중치를 사용했으므로 현재 Brier Score를 보정된 확률 품질로 해석하지 않습니다.

상세 설정, 실행 자원, 데이터 사용 감사와 전체 수치는 [Stage 4 기준 모델 학습 보고서](docs/Stage4_Baseline_Model_Report.md)와 [기계 판독용 결과](reports/stage4_baseline_results.json)에 기록했습니다. 학습 모델과 행별 validation 예측값은 `models/`에만 저장되며 Git에서 제외됩니다. test 피처·예측·평가는 사용하지 않았습니다.

저장된 validation 점수로 ROC·PR·Calibration 곡선, 위험도 decile과 Top 5%·10%·20% 우선검토 시나리오를 재현합니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.analysis.stage4_validation_analysis
```

주요 분석 결과:

- V3 Logistic은 위험도 상위 5%·10%·20%에서 실제 위험고객의 21.11%·33.75%·52.07%를 포착했습니다.
- V3 Logistic의 상위 decile 실제 상환곤란 비율은 27.25%, 하위 decile은 1.69%로 위험 순서가 뚜렷했습니다.
- V3 Random Forest도 순위 성능은 유효하지만 평균 예측점수 40.81%가 실제 위험률 8.07%보다 높아 현재 점수를 실제 확률로 해석할 수 없습니다.
- Calibration은 보정 필요성을 진단한 것이며 보정기를 학습하거나 운영 cutoff를 확정하지 않았습니다.
- V3 Logistic은 Stage 5 비교 기준선이며 아직 최종 모델이 아닙니다.

전체 표와 그림은 [Stage 4 Validation 상세 분석 보고서](docs/Stage4_Validation_Analysis_Report.md), 집계 결과는 [기계 판독용 분석 JSON](reports/stage4_validation_analysis.json)에 기록했습니다. test 데이터는 계속 봉인했습니다.

## Git에 올리지 않는 파일

- `data/raw/`, `data/interim/`, `data/processed/` 안의 모든 금융 데이터
- `models/` 안의 모든 학습 모델과 전처리 산출물
- 저장 위치와 무관한 일반 모델 형식(`.pkl`, `.joblib`, `.keras`, `.h5`, `.onnx`, `.pt`, `.pth`, `.ckpt` 등)
- `.env`, `kaggle.json`과 기타 인증정보
- 가상환경, Python 캐시, 노트북 체크포인트와 테스트 캐시
- 위치와 무관한 Parquet·DuckDB 파일(`*.parquet`, `*.duckdb`, `*.duckdb.wal`)

제외 규칙은 다음처럼 확인할 수 있습니다. 파일을 만들 필요 없이 경로만 검사합니다.

```bash
git check-ignore -v --no-index data/raw/application_train.csv
git check-ignore -v --no-index data/processed/features.parquet
git check-ignore -v --no-index models/creditlens.joblib
```

## 다음 작업

Stage 4를 완료했습니다. 다음 Stage에서는 현재 기준선인 V3 Logistic과 새로운 모델을 동일한 평가 체계로 비교합니다.

현재 Stage 1~4 범위의 자동 테스트 136개가 통과합니다.

1. LightGBM을 V1·V2·V3에서 학습해 데이터 원천별 추가 가치를 비교합니다.
2. V3 TensorFlow MLP를 학습해 전통적 머신러닝과 딥러닝을 비교합니다.
3. 하이퍼파라미터 탐색과 확률 보정은 train 내부에서만 수행합니다.
4. Logistic Regression·Random Forest·LightGBM·MLP를 같은 validation 지표로 비교합니다.
5. 최종 모델과 cutoff를 확정하기 전까지 test는 계속 봉인합니다.
