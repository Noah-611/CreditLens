# CreditLens 데이터 다운로드 가이드

> Kaggle Home Credit Default Risk 원본을 `data/raw/`에 안전하게 내려받고 재검증하는 절차입니다.

## 데이터 출처

- 데이터셋: [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- 라이선스: Kaggle Competition Rules 적용
- 최초 확보일: 2026-08-24
- 다운로드 도구: Kaggle CLI 2.2.4

Kaggle 대회가 종료되어 화면에 `Late Submission`이 표시되더라도 데이터 접근 전 대회 규칙에 동의해야 합니다. 브라우저와 CLI는 같은 Kaggle 계정을 사용합니다.

## 사전 준비

프로젝트 가상환경을 만들고 Kaggle CLI를 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "kaggle>=2.2,<3.0"
```

OAuth 인증을 진행합니다.

```bash
kaggle auth login --no-launch-browser
```

브라우저에서 인증을 완료한 뒤 대회 참가 상태를 확인합니다.

```bash
kaggle competitions list --group entered --format json
```

출력에 `home-credit-default-risk`와 `"userHasEntered": true`가 있어야 합니다. 비밀번호, OAuth 코드, API 토큰과 `kaggle.json`은 프로젝트 문서나 Git에 기록하지 않습니다.

## 핵심 파일 다운로드

Stage 1에서는 현재 신청정보, 외부 신용이력, 납부이력과 공식 컬럼 설명만 받습니다.

```bash
kaggle competitions download home-credit-default-risk \
  -f application_train.csv \
  -p data/raw

kaggle competitions download home-credit-default-risk \
  -f bureau.csv \
  -p data/raw

kaggle competitions download home-credit-default-risk \
  -f installments_payments.csv \
  -p data/raw

kaggle competitions download home-credit-default-risk \
  -f HomeCredit_columns_description.csv \
  -p data/raw
```

CSV 압축본은 기존 파일을 덮어쓰지 않는 `-n` 옵션으로 해제합니다.

```bash
unzip -n data/raw/application_train.csv.zip -d data/raw
unzip -n data/raw/bureau.csv.zip -d data/raw
unzip -n data/raw/installments_payments.csv.zip -d data/raw
```

## 고정 원본 Manifest

| 파일 | 데이터 행 수 | 컬럼 수 | 크기(bytes) | SHA-256 |
|---|---:|---:|---:|---|
| `application_train.csv` | 307,511 | 122 | 166,133,370 | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |
| `bureau.csv` | 1,716,428 | 17 | 170,016,717 | `9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7` |
| `installments_payments.csv` | 13,605,401 | 8 | 723,118,349 | `428c2e2496e4d6d697ee8270e98497e5213c41be16d882eed1bc95b133726797` |
| `HomeCredit_columns_description.csv` | 219 | 5 | 37,383 | `eef7665398228a80f7367c9258220c5fbe1038f3f54094244f354d54e2d4fb03` |

행 수는 헤더를 제외한 값입니다.

### 압축본 Manifest

| 파일 | 크기(bytes) | SHA-256 |
|---|---:|---|
| `application_train.csv.zip` | 37,847,529 | `64289b17dd316a4106a2e3fe8d37f17ca6e16729e45989e86944049b7fb9050f` |
| `bureau.csv.zip` | 38,550,359 | `f99c38fad04437b1926675d079a821cd35e65fadbc39e7751c0e51222dba4e5f` |
| `installments_payments.csv.zip` | 284,147,164 | `978a18833c15bf537120d7e69026cf58357c087dee7fbbc0b0746a6d3afbc3f0` |

## 무결성 재검증

압축본 CRC를 확인합니다.

```bash
unzip -t data/raw/application_train.csv.zip
unzip -t data/raw/bureau.csv.zip
unzip -t data/raw/installments_payments.csv.zip
```

CSV 체크섬을 확인합니다.

```bash
sha256sum \
  data/raw/application_train.csv \
  data/raw/bureau.csv \
  data/raw/installments_payments.csv \
  data/raw/HomeCredit_columns_description.csv
```

Stage 1 전체 검증과 Markdown 문서 재생성:

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.data.validate_raw \
  --raw-dir data/raw
```

## 원본 관리 원칙

- `data/raw/`의 CSV와 ZIP은 내려받은 상태 그대로 보존하고 수정하지 않습니다.
- 정제·집계 결과는 각각 `data/interim/`, `data/processed/`에 저장합니다.
- 금융 데이터, 압축본과 생성 모델은 GitHub에 올리지 않습니다.
- 다운로드 파일을 교체했다면 행·열 수와 SHA-256 차이를 먼저 검토합니다.
- 원본 재배포 대신 이 문서와 재현 가능한 검증 코드를 Git에 기록합니다.

Git 제외 여부 확인:

```bash
git check-ignore -v --no-index data/raw/application_train.csv
git check-ignore -v --no-index data/raw/installments_payments.csv.zip
```
