# CreditLens 분석 마트 명세

> Stage 3에서 대출 신청정보, 외부 신용거래와 과거 납부기록을 고객당 한 행의 V1·V2·V3 분석 데이터로 만드는 기준을 정의합니다. 이 문서는 모델 학습 전 데이터 구축 계약이며, 학습 기반 결측 대치·인코딩·스케일링은 다루지 않습니다.

## 1. 목적

분석 마트는 서로 다른 행 단위의 세 원천 데이터를 같은 고객 기준으로 결합하여 다음 질문을 비교할 수 있게 합니다.

- 대출 신청정보만 사용했을 때 어느 정도의 상환곤란 위험을 구분할 수 있는가?
- 외부 신용거래를 추가하면 예측 정보가 얼마나 늘어나는가?
- 과거 납부행동까지 추가하면 위험 구분력이 얼마나 개선되는가?

이를 위해 V1에서 V3까지 공통 고객, `TARGET`과 `SPLIT`을 유지합니다. 보조 테이블은 먼저 고객 단위로 집계한 뒤 LEFT JOIN하여 일대다 조인에 따른 고객 행 증가를 방지합니다.

## 2. 분석 단위와 예측 시점

| 항목 | 계약 |
|---|---|
| 분석 단위 | 현재 대출 신청 고객 1명당 1행 |
| 고객 기준키 | `SK_ID_CURR` |
| 예측 대상 | `TARGET=1`, 상환곤란이 관측된 고객 |
| 예측 시점 | 현재 대출 신청을 심사하는 시점 |
| 허용 정보 | 신청 시점에 관측 가능한 신청정보와 신청 시점까지의 신용·납부이력 |
| 금지 정보 | 현재 신청 이후 생성된 정보, `TARGET`을 직접·간접적으로 나타내는 정보 |
| 분할 | Stage 2에서 고정한 train/validation/test 고객 배정 |

분석 마트는 전체 고객에 대해 고객 내부의 과거 이력을 결정론적으로 집계합니다. 고객 간 분포를 학습하는 중앙값, 분위수, 희소범주 기준과 스케일 값은 여기서 계산하지 않습니다. 이러한 전처리기는 Stage 4에서 train에만 `fit`합니다.

`SK_ID_CURR`, `TARGET`, `SPLIT`은 데이터 추적과 평가를 위한 메타데이터이며 모델 입력이 아닙니다.

## 3. 입력 계약

### 입력 파일

| 입력 | 원본 행 단위 | 핵심 계약 | 사용 방식 |
|---|---|---|---|
| `data/raw/application_train.csv` | 현재 대출 신청 1건 | `SK_ID_CURR` 유일·비결측, `TARGET`은 0 또는 1 | V1의 기준 고객과 신청 피처 |
| `data/raw/bureau.csv` | 외부 신용거래 1건 | `SK_ID_CURR`, `SK_ID_BUREAU` 비결측, `SK_ID_BUREAU` 유일 | 신청 고객만 선별한 뒤 고객별 집계 |
| `data/raw/installments_payments.csv` | 납부행위 1건 | 고객·이전대출 키 비결측, 납부일·납부금액 결측 상태 일치 | 할부 회차로 1차 집계한 뒤 고객별 2차 집계 |
| `data/interim/customer_splits.csv` | 현재 신청 고객 1명 | `SK_ID_CURR` 유일, `TARGET` 보존, `SPLIT`은 train/validation/test | 모든 마트에 동일한 분할 부여 |

입력 파일은 읽기 전용으로 사용합니다. 실행 전 SHA-256을 계산하고 파일 크기와 수정시각을 기록하며, 실행 후 해당 상태가 바뀌면 최종 출력을 교체하지 않습니다.

### 최신 입력 식별값

최신 전체 빌드에 사용한 입력은 다음과 같습니다.

| 입력 | 크기 | SHA-256 |
|---|---:|---|
| `application_train.csv` | 166,133,370 bytes | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |
| `bureau.csv` | 170,016,717 bytes | `9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7` |
| `installments_payments.csv` | 723,118,349 bytes | `428c2e2496e4d6d697ee8270e98497e5213c41be16d882eed1bc95b133726797` |
| `customer_splits.csv` | 4,797,198 bytes | `bb61706ea286086a726497ade1fa0aa3f1aad60ff505461d0612aca6222382af` |

## 4. 출력 계약과 최신 결과

### 데이터 버전

| 버전 | 구성 | 로컬 출력 |
|---|---|---|
| V1 | 신청 원본 피처 + 신청 파생 피처 | `data/processed/feature_mart_v1.parquet` |
| V2 | V1 + 고객별 외부 신용거래 피처 | `data/processed/feature_mart_v2.parquet` |
| V3 | V2 + 고객별 과거 납부행동 피처 | `data/processed/feature_mart_v3.parquet` |

모든 버전은 다음 조건을 만족해야 합니다.

- 고객 수는 307,511명이며 `SK_ID_CURR`는 유일하고 비어 있지 않습니다.
- `TARGET`과 `SPLIT`은 `customer_splits.csv`와 정확히 일치합니다.
- V2의 V1 공통 컬럼과 V3의 V2 공통 컬럼은 이름, 순서와 값이 바뀌지 않습니다.
- 컬럼명과 DuckDB 자료형을 함께 사용한 스키마 해시를 기록합니다.
- 최종 행은 `SK_ID_CURR` 오름차순으로 저장합니다.

### 최신 전체 빌드 결과

최신 결과는 `reports/stage3_build_summary.json`의 2026-08-31 12:42:07 UTC 빌드를 기준으로 합니다.

| 버전 | 행 | 전체 컬럼 | 후보 피처 | 정책 제외 | 모델 입력 가능 피처 | 이력 보유 고객 |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 307,511 | 136 | 133 | 1 | 132 | - |
| V2 | 307,511 | 173 | 170 | 1 | 169 | bureau 263,490명 |
| V3 | 307,511 | 202 | 199 | 1 | 198 | bureau 263,490명, installments 291,643명 |

후보 피처 수는 전체 컬럼에서 `SK_ID_CURR`, `TARGET`, `SPLIT`을 제외한 수입니다. 정책 제외 1개는 `CODE_GENDER`이며, 이후 분석에 따라 Stage 4에서 추가 피처를 제외할 수 있습니다.

| split | 전체 고객 | TARGET=0 | TARGET=1 |
|---|---:|---:|---:|
| train | 215,258 | 197,881 | 17,377 |
| validation | 46,127 | 42,403 | 3,724 |
| test | 46,126 | 42,402 | 3,724 |
| 전체 | 307,511 | 282,686 | 24,825 |

| 버전 | 스키마 SHA-256 | Parquet SHA-256 |
|---|---|---|
| V1 | `04bd3db5a23efb8bbbf85d079703ea5b475b95b1f7deecffa9aa5cdeb41a7549` | `b879f6607ca15be6225942ecc5d2dad6b8a3a8841597a89b1bb50a914a8f6cd4` |
| V2 | `49df806e0575d037eeaf49817d018c08d681272a2b7cd14df141357e51c54702` | `4d95d75b680b916f89a6641fd114d6b3da9ccd0c6f032c8d7cea4eda2833a02e` |
| V3 | `846fdfb5adb20774b9ff3c1bfd9de218c7063161eadcd87f9db1ce8bdb7dd5ee` | `94b761171131d1bb8fccc67fec68d30ff01e37404e052480ecf33758945c1732` |

## 5. V1 → V2 → V3 데이터 계보

```text
application_train + customer_splits
                │
                └── V1 신청정보 마트
                         │
bureau ── 신청 이후 갱신 제외·고객별 집계 ──┴── V2 외부 신용정보 마트
                                  │
installments ── 회차별 집계 ── 고객별 집계 ──┴── V3 납부행동 마트
```

### V1: 신청정보

`01_v1_application.sql`은 application과 고객 분할표를 `SK_ID_CURR`, `TARGET`으로 INNER JOIN합니다. 실행 전 두 입력의 고객과 TARGET 보존을 별도로 검사하므로 불일치가 있으면 빌드를 중단합니다.

원본 신청 피처를 유지하면서 다음과 같은 결정론적 파생값을 추가합니다.

- 소득 대비 대출금액과 연간 상환액
- 대출금액 대비 상품가격과 대출금액 대비 연간 상환액
- 가족 1인당 소득
- 나이, 재직기간과 나이 대비 재직기간
- 외부 신용평가값 관측 개수와 관측값 평균
- `DAYS_EMPLOYED=365243` sentinel 분리
- 차량 미보유에 따른 `OWN_CAR_AGE` 해당 없음과 차량 보유자의 실제 결측 구분

분모가 0이거나 필요한 값이 결측이면 비율을 임의의 0으로 만들지 않고 NULL로 둡니다.

### V2: 외부 신용거래

`02_bureau_features.sql`은 application 고객의 bureau 행만 사용하며 다음 순서로 처리합니다.

1. `SK_ID_CURR`로 application 고객 범위를 제한합니다.
2. `DAYS_CREDIT_UPDATE > 0`인 17행을 신청 이후 갱신 가능성이 있는 정보로 보고 제외합니다.
3. 남은 신용거래를 `SK_ID_CURR`로 집계합니다.
4. `03_v2_bureau.sql`에서 V1에 LEFT JOIN합니다.

최신 빌드에서는 application 고객에 속하고 예측 시점 조건을 통과한 bureau 1,465,308행을 집계했습니다. 외부 신용이력 유무는 필터 적용 후의 `BUREAU_HAS_HISTORY`로 표시합니다.

거래 상태·유형·연체 건수와 비율은 모든 통화를 대상으로 계산합니다. 금액 합산은 통화가 다른 값을 직접 더하지 않도록 `currency 1`만 사용합니다. 부채비율은 대출금액과 부채가 함께 관측된 동일 거래만 분자와 분모에 포함합니다.

### V3: 과거 납부행동

`installments_payments.csv`에는 같은 예정 할부금에 대한 분할납부가 여러 행으로 기록될 수 있습니다. 원본 행을 고객 단위로 바로 합산하지 않고 다음 두 단계로 집계합니다.

1. **할부 회차 집계**
   - 키: `SK_ID_CURR`, `SK_ID_PREV`, `NUM_INSTALMENT_VERSION`, `NUM_INSTALMENT_NUMBER`
   - 예정일과 예정금액이 회차 안에서 하나로 결정되는지 먼저 검증합니다.
   - 예정일은 `MIN(DAYS_INSTALMENT)`, 최종 납부일은 `MAX(DAYS_ENTRY_PAYMENT)`으로 정리합니다.
   - 예정금액은 회차 내 동일성을 확인한 뒤 `MAX(AMT_INSTALMENT)`를 사용합니다.
   - 분할 납부금액은 DECIMAL로 합산하여 중간 계산 오차를 줄입니다.
2. **고객 집계**
   - 회차 수, 이전 대출 수, 납부행 수를 계산합니다.
   - 지연 회차 수·비율, 평균·최대 지연일을 계산합니다.
   - 예정금액·납부금액·부족금액과 납부비율을 계산합니다.
   - 전체, 최근 365일과 최근 730일의 납부행동을 각각 집계합니다.

최신 빌드에서는 application 고객의 납부행 11,591,592건을 11,026,627개 할부 회차로 정리한 뒤 291,643명의 고객 피처로 집계했습니다. 최종 금액 피처는 Stage 4에서 바로 수치형으로 사용할 수 있도록 DOUBLE로 저장합니다.

실제 납부일과 납부금액은 함께 결측이어야 합니다. 한 회차의 납부행 중 하나라도 이 두 값이 결측이면 그 회차를 `INST_MISSING_PAYMENT_SCHEDULE_COUNT`에 포함하고, 해당 회차의 납부금액·지연·부족금액 계산은 NULL로 남깁니다. 결측을 납부금액 0으로 단정하지 않습니다.

## 6. 조인 grain과 행 수 보존

| 단계 | 조인·집계 키 | 조인 방식 | 보존 규칙 |
|---|---|---|---|
| application + split | `SK_ID_CURR`, `TARGET` | INNER JOIN | 두 입력이 완전히 일치해야 하며 307,511명 보존 |
| bureau 범위 제한 | `SK_ID_CURR` | INNER JOIN | application 밖 고객 제외 |
| bureau 고객 집계 | `SK_ID_CURR` | GROUP BY | 고객당 최대 1행 |
| V1 + bureau | `SK_ID_CURR` | LEFT JOIN | V1 고객 전체 보존 |
| installments 범위 제한 | `SK_ID_CURR` | INNER JOIN | application 밖 고객 제외 |
| installments 회차 집계 | 고객·이전대출·버전·할부번호 | GROUP BY | 분할납부를 예정 할부 회차 1행으로 축약 |
| installments 고객 집계 | `SK_ID_CURR` | GROUP BY | 고객당 최대 1행 |
| V2 + installments | `SK_ID_CURR` | LEFT JOIN | V2 고객 전체 보존 |

보조 테이블을 application에 원본 행 그대로 조인하는 것은 금지합니다. 그렇게 하면 한 고객이 여러 행으로 늘어나고 `TARGET`이 반복되어 모델 학습 단위가 깨집니다.

## 7. 0과 NULL 규약

0은 실제로 관측된 0 또는 존재하지 않는 건수를 뜻하고, NULL은 계산할 수 없거나 관측되지 않은 값을 뜻합니다.

| 상황 | 저장 규칙 |
|---|---|
| bureau 또는 installments 이력 없음 | `*_HAS_HISTORY=0` |
| 이력 없음의 건수 피처 | 0 |
| 이력 없음의 평균·최댓값·비율·금액 관측통계 | NULL |
| 이력은 있으나 해당 금액이 관측되지 않음 | 관측 건수 0, 금액 통계 NULL |
| 실제 납부금액이 관측된 0원 | 0으로 유지하고 underpaid 계산에 포함 |
| 실제 납부금액이 결측 | 0으로 대치하지 않고 해당 회차의 납부 파생값을 NULL 처리 |
| 비율의 분모가 0 또는 관측되지 않음 | `NULLIF`를 사용하여 NULL |
| `DAYS_EMPLOYED=365243` | `DAYS_EMPLOYED=NULL`, `DAYS_EMPLOYED_SENTINEL=1` |
| 차량 미보유자의 차량 연식 결측 | 해당 없음 플래그로 구분 |

`BUREAU_PROLONG_COUNT_SUM`과 각 보조 테이블의 건수 피처는 이력이 없을 때 0입니다. 반면 금액 합계는 원천 금액이 모두 결측인 경우와 이력 자체가 없는 경우를 임의의 0으로 합치지 않습니다.

납부비율의 분모에는 납부금액이 관측된 회차의 예정금액만 포함합니다. 결측 회차를 미납으로 간주해 비율을 낮추지 않습니다.

## 8. 모델 입력 제외 정책

다음 컬럼은 분석 마트에 존재하더라도 모델 입력에서 제외합니다.

| 컬럼 | 정책 |
|---|---|
| `SK_ID_CURR` | 조인·추적용 식별자이므로 제외 |
| `TARGET` | 정답 레이블이므로 제외 |
| `SPLIT` | 데이터 분할 메타데이터이므로 제외 |
| `CODE_GENDER` | 직접적인 성별 사용을 피하기 위해 정책상 제외 |

`CODE_GENDER`는 모델 학습에는 사용하지 않지만 공정성 점검을 위해 로컬 분석 마트에 보존합니다. 이 정책은 `reports/stage3_build_summary.json`의 `policy_excluded_columns`에 기록되며 모든 버전에서 동일하게 적용됩니다.

## 9. DuckDB SQL 실행 순서

Python 모듈 `creditlens.data.build_feature_mart`가 다음 순서를 제어합니다.

1. 입력·출력 경로, 메모리와 스레드 설정을 검증합니다.
2. 입력 파일과 SQL의 SHA-256을 계산합니다.
3. DuckDB에 application, bureau, installments와 split CSV view를 등록합니다.
4. 키, TARGET, split, 상대일수, 납부 결측 쌍과 할부 회차 계약을 검증합니다.
5. `01_v1_application.sql`로 임시 V1 Parquet을 만들고 `v1_source` view로 등록합니다.
6. `02_bureau_features.sql` 결과를 임시 테이블로 만들고 `03_v2_bureau.sql`로 임시 V2 Parquet을 만듭니다.
7. `04_installment_features.sql` 결과를 임시 테이블로 만들고 `05_v3_installments.sql`로 임시 V3 Parquet을 만듭니다.
8. 세 출력의 키·TARGET·split, 스키마·자료형, 건수·비율 계약을 검증합니다.
9. `EXCEPT ALL` 대사로 V1→V2와 V2→V3 공통 컬럼의 값 보존을 확인합니다.
10. 입력 파일이 실행 중 바뀌지 않았는지 확인하고 출력 해시와 공유용 요약을 만듭니다.
11. 모든 검증이 끝난 뒤 임시 파일을 최종 경로로 파일 단위 원자 교체하고 요약 JSON을 마지막에 교체합니다.

DuckDB 기본 실행 설정은 메모리 제한 3GB, 2개 스레드와 `preserve_insertion_order=false`입니다. 대용량 GROUP BY가 메모리를 넘으면 `data/interim/duckdb_tmp/`를 임시 저장공간으로 사용합니다. Parquet은 Zstandard 압축과 100,000행 row group으로 기록합니다.

## 10. 검증과 재현성

### 자동 검증

빌드는 다음 조건 중 하나라도 위반하면 실패합니다.

- application 또는 split의 고객 중복·누락, TARGET 불일치
- 허용되지 않은 split 값
- bureau 키 결측·중복 또는 `DAYS_CREDIT > 0`
- installments 키 결측, 신청 이후 예정·납부일
- 실제 납부일과 납부금액의 결측 상태 불일치
- 한 `SK_ID_PREV`가 여러 고객에 연결됨
- 동일 할부 회차의 예정일 또는 예정금액 불일치
- 고객 행 수 증가, TARGET·split 변경
- 이력 플래그와 건수 불일치, 비율 범위·유한성 위반
- V1→V2→V3 공통 데이터 변경
- installments 금액 피처가 DOUBLE이 아님

최신 빌드의 계약 위반은 모두 0건이며, V1→V2와 V2→V3 계보 불일치도 각각 0건입니다. 모든 `invariants`는 `true`입니다. 각 파생값의 정확한 산식은 [피처 사전](Feature_Dictionary.md)에 기록했습니다.

### 최신 실행 환경과 자원 사용

| 항목 | 결과 |
|---|---:|
| Python | 3.12.3 |
| DuckDB | 1.5.5 |
| 메모리 제한 | 3GB |
| 스레드 | 2 |
| 프로세스 최대 RSS | 2,300.25 MiB |
| 전체 실행시간 | 87.684초 |
| 입력 검증 | 26.616초 |
| V1 구축 | 8.814초 |
| V2 구축 | 11.692초 |
| V3 구축 | 18.173초 |
| 출력 검증 | 19.631초 |

최대 RSS는 현재 Python 프로세스 수명 전체에서 관측한 값입니다. 다른 코드가 같은 프로세스에서 먼저 실행되면 Stage 3만의 사용량보다 크게 기록될 수 있습니다.

### 재실행 명령

프로젝트 루트에서 기본 전체 빌드를 실행합니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.data.build_feature_mart
```

메모리와 스레드를 명시하려면 다음과 같이 실행합니다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.data.build_feature_mart \
  --memory-limit 3GB \
  --threads 2
```

합성 데이터 회귀 테스트는 원본 금융 데이터를 읽지 않습니다.

```bash
.venv/bin/python -m pytest -q tests/test_feature_mart.py
```

재실행 후에는 다음 항목을 확인합니다.

1. `reports/stage3_build_summary.json`의 입력 SHA-256이 의도한 원본과 같은지 확인합니다.
2. 모든 `invariants`가 `true`인지 확인합니다.
3. `lineage_mismatch_rows`의 두 값이 0인지 확인합니다.
4. 각 버전의 고객 수, TARGET·split 건수와 스키마가 이 명세와 같은지 확인합니다.
5. SQL 또는 실행환경이 바뀌었다면 출력 파일 SHA가 달라질 수 있으므로 summary에 기록된 SQL·환경·스키마 해시를 함께 비교합니다.

## 11. 로컬 보관과 Git 정책

고객별 원본·중간·가공 데이터는 모두 로컬 전용입니다.

- 원본 CSV: `data/raw/`
- DuckDB 임시 파일: `data/interim/duckdb_tmp/`
- 고객별 V1·V2·V3 Parquet: `data/processed/`

`data/raw/`, `data/interim/`, `data/processed/`, `*.parquet`, `*.duckdb`와 `*.duckdb.wal`은 `.gitignore`로 제외합니다. 따라서 고객 ID, 개별 금융정보와 분석 마트는 GitHub에 올리지 않습니다.

Git에는 다음과 같이 개인 행이 없는 재현 자료만 기록합니다.

- `src/creditlens/data/build_feature_mart.py`
- `sql/stage3/*.sql`
- 합성 데이터 기반 테스트
- 이 분석 마트 명세
- 고객 ID와 절대경로가 없는 집계 요약 `reports/stage3_build_summary.json`

공유용 요약에는 입력·출력 체크섬, 행·열 수, 집계 건수, 실행시간과 검증 결과만 포함합니다. 고객 ID 목록이나 행 단위 금융정보는 포함하지 않습니다.
