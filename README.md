# CreditLens

CreditLens는 Kaggle의 공개 익명 금융 데이터를 활용하여 대출 신청자의 상환곤란 위험을 분석하고 예측하는 머신러닝 프로젝트입니다.

대출 신청정보, 외부 신용거래와 과거 납부기록을 고객 단위로 통합하고 금융 파생변수를 생성합니다. Logistic Regression, Random Forest, LightGBM과 TensorFlow MLP를 동일한 데이터 분할과 평가 기준으로 비교하고, 최종 모델의 주요 예측요인을 SHAP으로 분석합니다. 학습이 완료된 모델은 Streamlit 화면과 FastAPI 예측 API에서 사용할 수 있도록 구성합니다.

## 현재 상태

- Stage 0: 프로젝트 초기 환경 구성 완료
- Stage 1: 데이터 확보 및 무결성 확인 완료 (`PASS`, 오류 0건)
- Stage 2: 고객 분할·train-only EDA·전처리 및 누수 정책 설계 완료
- 다음 작업: Stage 3 SQL·Python 고객 분석 마트와 V1·V2·V3 구축
- 아직 전처리 적용, 피처 생성과 모델 학습은 시작하지 않았습니다.

전체 범위와 Stage별 완료 조건은 [프로젝트 계획서](docs/Project_Plan.md)를 참고하세요.

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
│   ├── Preprocessing_and_Leakage_Spec.md
│   ├── Stage2_EDA_Report.md
│   └── Project_Plan.md
├── models/                   # 학습 모델과 전처리 산출물 (Git 제외)
├── notebooks/
│   └── 01_stage2_eda.ipynb   # train-only EDA 실행 진입점
├── reports/
│   ├── figures/              # Stage 2 핵심 시각화
│   ├── data_validation.json
│   ├── stage2_eda.json
│   └── stage2_split_summary.json
├── src/
│   └── creditlens/
│       ├── analysis/         # train-only EDA 모듈
│       └── data/             # 원본 검증·고객 분할 모듈
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

현재 합성 데이터 기반 회귀 테스트 11개가 통과합니다.

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

상세 결과는 [Stage 2 EDA 보고서](docs/Stage2_EDA_Report.md), 처리 원칙은 [전처리·누수 점검 명세](docs/Preprocessing_and_Leakage_Spec.md)에 기록되어 있습니다. 아직 실제 결측 대치, 인코딩, 피처 생성이나 모델 학습은 수행하지 않았습니다.

전체 자동 테스트:

```bash
.venv/bin/python -m pytest -q
```

현재 Stage 1·2 합성 회귀 테스트 34개가 통과합니다.

## Git에 올리지 않는 파일

- `data/raw/`, `data/interim/`, `data/processed/` 안의 모든 금융 데이터
- `models/` 안의 모든 학습 모델과 전처리 산출물
- 저장 위치와 무관한 일반 모델 형식(`.pkl`, `.joblib`, `.keras`, `.h5`, `.onnx`, `.pt`, `.pth`, `.ckpt` 등)
- `.env`, `kaggle.json`과 기타 인증정보
- 가상환경, Python 캐시, 노트북 체크포인트와 테스트 캐시

제외 규칙은 다음처럼 확인할 수 있습니다. 파일을 만들 필요 없이 경로만 검사합니다.

```bash
git check-ignore -v --no-index data/raw/application_train.csv
git check-ignore -v --no-index data/processed/features.parquet
git check-ignore -v --no-index models/creditlens.joblib
```

## 다음 작업

Stage 3에서 다음 작업을 진행합니다.

1. 분석 단위, 예측 시점과 입력·출력 계약을 정의합니다.
2. `application_train` 정제와 V1 피처를 구현합니다.
3. DuckDB SQL로 `bureau`와 `installments_payments`를 고객 단위 집계해 V2·V3를 만듭니다.
4. Python으로 전체 실행 흐름과 SQL 집계 결과를 검산합니다.
5. 고객 수·키·`TARGET`·split과 조인 전후 행 수 보존을 자동 검증합니다.
6. 피처 원천·산식·결측 의미·예측 시점 사용 가능 여부를 Markdown으로 기록합니다.
7. 원본 checksum, 실행 설정과 출력 스키마를 분석 마트 빌드 명세에 남깁니다.

모델 학습은 Stage 3 데이터 구축과 검증이 끝난 뒤 Stage 4에서 시작합니다.
