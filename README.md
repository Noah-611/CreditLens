# CreditLens

CreditLens는 대출 신청정보와 과거 신용·납부 이력을 이용해 상환곤란 위험을 예측하고, 금융기관의 우선심사를 지원하는 교육용 AI 프로젝트입니다.

모델이 대출을 자동으로 승인하거나 거절하지 않습니다. 예측 결과는 심사자가 검토 순서를 정할 때 참고하는 보조 정보로만 사용합니다.

## 현재 상태

- Stage 0: 프로젝트 초기 환경 구성 완료
- Stage 1: 데이터 확보 및 무결성 확인 완료 (`PASS`, 오류 0건)
- 다음 작업: Stage 2 분할 전략·EDA·전처리 설계
- 아직 데이터 전처리, 피처 생성, 모델 학습, API·화면 구현은 시작하지 않았습니다.

전체 범위와 Stage별 완료 조건은 [프로젝트 계획서](docs/Project_Plan.md)를 참고하세요.

## 예정 기능

- 상환곤란 예측확률과 위험등급 제공
- 고객별 주요 위험요인 설명
- 우선심사 대상 제안
- V1·V2·V3 데이터 및 모델 성능 비교
- FastAPI 예측 API와 Streamlit 대시보드

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
│   ├── Data_Validation_Report.md
│   └── Project_Plan.md
├── models/                   # 학습 모델과 전처리 산출물 (Git 제외)
├── notebooks/                # 탐색·실험 노트북
├── reports/
│   ├── figures/              # 결과 시각화
│   └── data_validation.json  # 기계 판독용 Stage 1 검증 결과
├── src/
│   └── creditlens/
│       └── data/             # 원본 데이터 검증 모듈
├── tests/                    # 합성 데이터 기반 자동 테스트
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
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

설치 확인:

```bash
python --version
python -m pip check
```

`requirements.txt`는 계획된 전체 기술 스택의 초기 호환 범위를 담고 있습니다. 첫 환경 검증 후 실제로 사용한 버전을 고정하고, 프로젝트가 커지면 개발·모델링·서비스 의존성을 분리합니다.

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

Stage 2에서 다음 작업을 진행합니다.

1. 고정 random seed와 층화 학습·검증·테스트 분할을 설계합니다.
2. 수치형·범주형 변수의 EDA와 시각화를 진행합니다.
3. 예측 시점 기준 데이터 누수 후보를 검토합니다.
4. 결측·이상치·범주 인코딩·스케일링 기준을 문서화합니다.

모델링과 피처 생성은 Stage 2의 분할·누수 방지 설계가 끝난 뒤 시작합니다.
