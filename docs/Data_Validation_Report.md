# CreditLens 원본 데이터 검증 보고서

> 최종 상태: **PASS** — 오류 0건, 경고 43건, 정보 3건

## 실행 정보

- 생성 시각(UTC): `2026-08-24T06:52:21.619819+00:00`
- 청크 크기: `200,000`행
- Python: `3.12.3`
- Pandas: `3.0.5`
- NumPy: `2.5.2`
- 검증 방식: 원본을 수정하지 않는 전체 청크 스캔

## 원본 파일 Manifest

| 파일 | 행 | 열 | 크기 | SHA-256 |
|---|---:|---:|---:|---|
| `data/raw/application_train.csv` | 307,511 | 122 | 158.4 MB | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |
| `data/raw/bureau.csv` | 1,716,428 | 17 | 162.1 MB | `9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7` |
| `data/raw/installments_payments.csv` | 13,605,401 | 8 | 689.6 MB | `428c2e2496e4d6d697ee8270e98497e5213c41be16d882eed1bc95b133726797` |
| `data/raw/HomeCredit_columns_description.csv` | 219 | 5 | 36.5 KB | `eef7665398228a80f7367c9258220c5fbe1038f3f54094244f354d54e2d4fb03` |

## 스키마·키·중복 검증

| 테이블 | 행 단위 | 기본키 | 키 결측 | 키 중복 | 동일 행 해시 반복 |
|---|---|---|---:|---:|---:|
| application | 현재 대출 신청 1건당 1행 | `SK_ID_CURR` | 0 | 0 | 0 |
| bureau | 타 금융기관 신용거래 1건당 1행 | `SK_ID_BUREAU` | 0 | 0 | 0 |
| installments | 예정 할부금에 대한 실제 납부행위 1건당 1행(분할납부 가능) | 보장된 단일 키 없음 | - | - | 0 |

- `SK_ID_PREV → SK_ID_CURR` 함수 종속 위반: **0개 키**
- 동일 행 검사는 64비트 행 해시 기반 후보 탐지이며, 반복이 발견되면 Stage 2에서 실제 행을 재확인합니다.

## 테이블 관계와 이력 커버리지

| 보조 테이블 | 전체 고객 | train 매칭 고객 | train 고객 커버리지 | train 밖 고객 | train 매칭 행 |
|---|---:|---:|---:|---:|---:|
| bureau | 305,811 | 263,491 | 85.69% | 42,320 | 1,465,325 (85.37%) |
| installments | 339,587 | 291,643 | 94.84% | 47,944 | 11,591,592 (85.20%) |

train 고객 이력 조합:

- 두 이력 모두 존재: 250,003명 (81.30%)
- bureau만 존재: 13,488명
- installments만 존재: 41,640명
- 두 이력 모두 없음: 2,380명 (0.77%)

보조 테이블에는 현재 신청 데이터에 없는 고객도 포함됩니다. Stage 3에서는 현재 신청 고객만 범위에 포함하고, 각 보조 테이블을 고객 단위로 먼저 집계한 뒤 `application_train`에 LEFT JOIN했습니다.

## TARGET 분포

| TARGET | 의미 | 건수 | 비율 |
|---:|---|---:|---:|
| 0 | 정상 상환 | 282,686 | 91.9271% |
| 1 | 상환곤란 | 24,825 | 8.0729% |

## 결측률 상위 컬럼

### application

| 컬럼 | 결측 건수 | 결측률 |
|---|---:|---:|
| `COMMONAREA_AVG` | 214,865 | 69.87% |
| `COMMONAREA_MODE` | 214,865 | 69.87% |
| `COMMONAREA_MEDI` | 214,865 | 69.87% |
| `NONLIVINGAPARTMENTS_AVG` | 213,514 | 69.43% |
| `NONLIVINGAPARTMENTS_MODE` | 213,514 | 69.43% |
| `NONLIVINGAPARTMENTS_MEDI` | 213,514 | 69.43% |
| `FONDKAPREMONT_MODE` | 210,295 | 68.39% |
| `LIVINGAPARTMENTS_AVG` | 210,199 | 68.35% |
| `LIVINGAPARTMENTS_MODE` | 210,199 | 68.35% |
| `LIVINGAPARTMENTS_MEDI` | 210,199 | 68.35% |

### bureau

| 컬럼 | 결측 건수 | 결측률 |
|---|---:|---:|
| `AMT_ANNUITY` | 1,226,791 | 71.47% |
| `AMT_CREDIT_MAX_OVERDUE` | 1,124,488 | 65.51% |
| `DAYS_ENDDATE_FACT` | 633,653 | 36.92% |
| `AMT_CREDIT_SUM_LIMIT` | 591,780 | 34.48% |
| `AMT_CREDIT_SUM_DEBT` | 257,669 | 15.01% |
| `DAYS_CREDIT_ENDDATE` | 105,553 | 6.15% |
| `AMT_CREDIT_SUM` | 13 | 0.00% |
| `SK_ID_CURR` | 0 | 0.00% |
| `SK_ID_BUREAU` | 0 | 0.00% |
| `CREDIT_ACTIVE` | 0 | 0.00% |

### installments

| 컬럼 | 결측 건수 | 결측률 |
|---|---:|---:|
| `DAYS_ENTRY_PAYMENT` | 2,905 | 0.02% |
| `AMT_PAYMENT` | 2,905 | 0.02% |
| `SK_ID_PREV` | 0 | 0.00% |
| `SK_ID_CURR` | 0 | 0.00% |
| `NUM_INSTALMENT_VERSION` | 0 | 0.00% |
| `NUM_INSTALMENT_NUMBER` | 0 | 0.00% |
| `DAYS_INSTALMENT` | 0 | 0.00% |
| `AMT_INSTALMENT` | 0 | 0.00% |

## 주요 도메인 관찰

### application

| 관찰 항목 | 건수 |
|---|---:|
| CODE_GENDER=XNA | 4 |
| DAYS_EMPLOYED=365243 sentinel | 55,374 |
| 가족상태 Unknown | 2 |
| 차량 보유 고객의 차량연식 결측 | 5 |

### bureau

| 관찰 항목 | 건수 |
|---|---:|
| Active이지만 실제 종료일 존재 | 1,969 |
| Closed이지만 실제 종료일 결측 | 125 |
| 음수 외부 대출잔액 | 8,418 |
| 음수 신용한도 | 351 |
| 양수 DAYS_CREDIT_UPDATE | 17 |

### installments

| 관찰 항목 | 건수 |
|---|---:|
| 예정금액보다 많은 납부 | 179,397 |
| 예정금액보다 적은 납부 | 1,295,493 |
| 예정일보다 이른 납부 | 9,309,477 |
| 예정금액과 같은 납부 | 12,127,606 |
| 실제 납부일·납부금액 동시 결측 | 2,905 |
| 실제 납부일·납부금액 결측 불일치 | 0 |
| 예정일보다 늦은 납부 | 1,146,669 |
| 예정일 당일 납부 | 3,146,350 |

## 발견사항

| 심각도 | 테이블 | 코드 | 내용 |
|---|---|---|---|
| WARNING | application | `high_missing_rate` | OWN_CAR_AGE 결측률이 65.99%입니다. |
| WARNING | application | `high_missing_rate` | EXT_SOURCE_1 결측률이 56.38%입니다. |
| WARNING | application | `high_missing_rate` | APARTMENTS_AVG 결측률이 50.75%입니다. |
| WARNING | application | `high_missing_rate` | BASEMENTAREA_AVG 결측률이 58.52%입니다. |
| WARNING | application | `high_missing_rate` | YEARS_BUILD_AVG 결측률이 66.50%입니다. |
| WARNING | application | `high_missing_rate` | COMMONAREA_AVG 결측률이 69.87%입니다. |
| WARNING | application | `high_missing_rate` | ELEVATORS_AVG 결측률이 53.30%입니다. |
| WARNING | application | `high_missing_rate` | ENTRANCES_AVG 결측률이 50.35%입니다. |
| WARNING | application | `high_missing_rate` | FLOORSMIN_AVG 결측률이 67.85%입니다. |
| WARNING | application | `high_missing_rate` | LANDAREA_AVG 결측률이 59.38%입니다. |
| WARNING | application | `high_missing_rate` | LIVINGAPARTMENTS_AVG 결측률이 68.35%입니다. |
| WARNING | application | `high_missing_rate` | LIVINGAREA_AVG 결측률이 50.19%입니다. |
| WARNING | application | `high_missing_rate` | NONLIVINGAPARTMENTS_AVG 결측률이 69.43%입니다. |
| WARNING | application | `high_missing_rate` | NONLIVINGAREA_AVG 결측률이 55.18%입니다. |
| WARNING | application | `high_missing_rate` | APARTMENTS_MODE 결측률이 50.75%입니다. |
| WARNING | application | `high_missing_rate` | BASEMENTAREA_MODE 결측률이 58.52%입니다. |
| WARNING | application | `high_missing_rate` | YEARS_BUILD_MODE 결측률이 66.50%입니다. |
| WARNING | application | `high_missing_rate` | COMMONAREA_MODE 결측률이 69.87%입니다. |
| WARNING | application | `high_missing_rate` | ELEVATORS_MODE 결측률이 53.30%입니다. |
| WARNING | application | `high_missing_rate` | ENTRANCES_MODE 결측률이 50.35%입니다. |
| WARNING | application | `high_missing_rate` | FLOORSMIN_MODE 결측률이 67.85%입니다. |
| WARNING | application | `high_missing_rate` | LANDAREA_MODE 결측률이 59.38%입니다. |
| WARNING | application | `high_missing_rate` | LIVINGAPARTMENTS_MODE 결측률이 68.35%입니다. |
| WARNING | application | `high_missing_rate` | LIVINGAREA_MODE 결측률이 50.19%입니다. |
| WARNING | application | `high_missing_rate` | NONLIVINGAPARTMENTS_MODE 결측률이 69.43%입니다. |
| WARNING | application | `high_missing_rate` | NONLIVINGAREA_MODE 결측률이 55.18%입니다. |
| WARNING | application | `high_missing_rate` | APARTMENTS_MEDI 결측률이 50.75%입니다. |
| WARNING | application | `high_missing_rate` | BASEMENTAREA_MEDI 결측률이 58.52%입니다. |
| WARNING | application | `high_missing_rate` | YEARS_BUILD_MEDI 결측률이 66.50%입니다. |
| WARNING | application | `high_missing_rate` | COMMONAREA_MEDI 결측률이 69.87%입니다. |
| WARNING | application | `high_missing_rate` | ELEVATORS_MEDI 결측률이 53.30%입니다. |
| WARNING | application | `high_missing_rate` | ENTRANCES_MEDI 결측률이 50.35%입니다. |
| WARNING | application | `high_missing_rate` | FLOORSMIN_MEDI 결측률이 67.85%입니다. |
| WARNING | application | `high_missing_rate` | LANDAREA_MEDI 결측률이 59.38%입니다. |
| WARNING | application | `high_missing_rate` | LIVINGAPARTMENTS_MEDI 결측률이 68.35%입니다. |
| WARNING | application | `high_missing_rate` | LIVINGAREA_MEDI 결측률이 50.19%입니다. |
| WARNING | application | `high_missing_rate` | NONLIVINGAPARTMENTS_MEDI 결측률이 69.43%입니다. |
| WARNING | application | `high_missing_rate` | NONLIVINGAREA_MEDI 결측률이 55.18%입니다. |
| WARNING | application | `high_missing_rate` | FONDKAPREMONT_MODE 결측률이 68.39%입니다. |
| WARNING | application | `high_missing_rate` | HOUSETYPE_MODE 결측률이 50.18%입니다. |
| WARNING | application | `high_missing_rate` | WALLSMATERIAL_MODE 결측률이 50.84%입니다. |
| WARNING | bureau | `high_missing_rate` | AMT_CREDIT_MAX_OVERDUE 결측률이 65.51%입니다. |
| WARNING | bureau | `high_missing_rate` | AMT_ANNUITY 결측률이 71.47%입니다. |
| INFO | application | `target_imbalance` | TARGET=1은 24,825건(8.07%)으로 불균형 데이터입니다. |
| INFO | bureau | `support_table_scope` | train 밖 고객 42,320명은 application_test 고객이 포함된 원본 구조로 해석합니다. |
| INFO | installments | `support_table_scope` | train 밖 고객 47,944명은 application_test 고객이 포함된 원본 구조로 해석합니다. |

## Stage 2 전달사항

- `DAYS_EMPLOYED=365243`은 오류가 아니라 알려진 sentinel이므로 결측 처리와 별도 플래그를 검토합니다.
- 음수 외부 대출잔액·신용한도는 업무 의미를 확인하기 전 삭제하지 않습니다.
- 실제 납부일·금액 동시 결측은 미납 또는 미기록 가능성이 있어 단순 0 대치하지 않습니다.
- 극단값은 삭제 근거가 아니며 분포와 업무 의미를 함께 검토합니다.
- 데이터 분할과 누수 검토는 Stage 2에서 시작합니다.

## 재실행 명령

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.data.validate_raw --raw-dir data/raw
```
