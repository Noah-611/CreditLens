# CreditLens

CreditLens는 대출 신청정보와 과거 신용·납부 이력을 이용해 상환곤란 위험을 예측하고, 금융기관의 우선심사를 지원하는 교육용 AI 프로젝트입니다.

모델이 대출을 자동으로 승인하거나 거절하지 않습니다. 예측 결과는 심사자가 검토 순서를 정할 때 참고하는 보조 정보로만 사용합니다.

## 현재 상태

- Stage 0: 프로젝트 초기 환경 구성 완료
- 다음 작업: Stage 1 데이터 확보 및 무결성 확인
- 아직 데이터 다운로드, 전처리, 모델 학습, API·화면 구현은 시작하지 않았습니다.

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
│   └── Project_Plan.md       # Stage별 프로젝트 계획
├── models/                   # 학습 모델과 전처리 산출물 (Git 제외)
├── notebooks/                # 탐색·실험 노트북
├── reports/
│   └── figures/              # 결과 시각화
├── src/
│   └── creditlens/           # 재사용 가능한 Python 패키지
├── tests/                    # 자동 테스트
├── .gitignore
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

Kaggle CLI를 사용할 경우 인증정보인 `kaggle.json`은 저장소에 복사하거나 커밋하지 마세요. 데이터 다운로드와 스키마 검증 절차는 Stage 1에서 구체화합니다.

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

Stage 1에서만 다음 작업을 진행합니다.

1. Kaggle 데이터 다운로드 절차를 확정합니다.
2. 파일·스키마·키 관계를 확인합니다.
3. 결측치, 중복, 자료형과 `TARGET` 분포를 점검합니다.
4. 데이터 사전 초안과 검증 보고서를 만듭니다.

모델링은 데이터 검증과 분할·누수 방지 설계가 끝난 뒤 시작합니다.
