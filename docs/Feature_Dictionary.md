# CreditLens Stage 3 피처 사전

> Stage 3에서 생성하는 V1·V2·V3 고객 단위 분석 마트의 파생 피처 정의입니다.

## 1. 문서 범위

이 문서는 Stage 3에서 새로 생성한 파생 피처 79개를 설명합니다. 원본 신청정보와 원본 보조 테이블 컬럼의 의미·자료형·결측 현황은 [Data_Dictionary.md](./Data_Dictionary.md)를 참고합니다.

| 데이터 버전 | 구성 | 이 단계에서 추가되는 피처 |
|---|---|---:|
| V1 | 대출 신청정보와 신청정보 파생값 | 13개 |
| V2 | V1 + 외부 신용이력 고객 집계 | 37개 |
| V3 | V2 + 과거 납부행동 고객 집계 | 29개 |
| 합계 |  | 79개 |

`SK_ID_CURR`, `TARGET`, `SPLIT`은 고객 연결·정답·분할을 위한 메타데이터이며 모델 입력 피처가 아니다. `CODE_GENDER`도 모델 입력에서 제외하고 공정성 점검용으로만 다룬다. 어떤 파생 산식에도 `TARGET`이나 다른 고객의 통계는 사용하지 않는다.

### 공통 표기와 결측 원칙

- **가능(신청):** 현재 대출 신청 시점에 관측 가능한 신청정보로 계산한다.
- **가능(외부이력):** 현재 신청 고객의 외부 신용이력 중 `DAYS_CREDIT_UPDATE <= 0`인 기록만 사용한다. 양수인 기록은 신청 이후 갱신 가능성이 있어 집계에서 제외한다.
- **가능(납부이력):** 현재 신청일 이후가 아닌 예정일과 실제 납부일만 사용한다. 현재 원본의 두 날짜는 모두 신청일보다 최소 하루 이전이다.
- 금액의 단위는 원본 데이터의 금액 단위를 따른다. 외부 신용이력의 금액 피처는 환율 정보가 없으므로 `CREDIT_CURRENCY = 'currency 1'`인 기록만 집계한다.
- 분모가 0이거나 비교할 관측값이 없으면 비율을 `NULL`로 둔다. 임의의 작은 값을 분모에 더하지 않는다.
- 이력이 없는 고객은 존재 여부와 건수 피처를 `0`으로 두고, 평균·최댓값·금액·비율처럼 관측값이 필요한 피처는 `NULL`로 둔다. 따라서 `0`과 `NULL`은 서로 다른 의미다.
- `DAYS_*` 원본 값은 현재 신청일을 0으로 하는 상대일수다. 음수일수의 절댓값을 취한 경과일 피처는 값이 클수록 더 오래된 이력을 뜻한다.

## 2. V1 신청정보 파생 피처 13개

V1은 `application_train.csv`의 원본 신청정보에 아래 피처를 추가한다. 원본 `DAYS_EMPLOYED=365243`은 실제 근속일이 아닌 sentinel이므로 분석 마트의 `DAYS_EMPLOYED`를 `NULL`로 바꾸고, 나머지 값은 그대로 보존한다. 이 정제된 원본 컬럼은 아래 13개 파생 피처 수에 포함하지 않는다.

| 피처명 | 산식·정의 | 단위 | `0`의 의미 | `NULL`의 의미 | 예측시점 |
|---|---|---|---|---|---|
| `DAYS_EMPLOYED_SENTINEL` | `DAYS_EMPLOYED = 365243`이면 1 | 0/1 | sentinel이 아닌 관측값 | 원본 값으로 판정할 수 없음 | 가능(신청) |
| `OWN_CAR_AGE_NOT_APPLICABLE` | `FLAG_OWN_CAR='N'`이고 `OWN_CAR_AGE`가 결측이면 1 | 0/1 | 차량연식이 구조적 해당 없음 조건이 아님 | 차량 보유 여부를 판정할 수 없음 | 가능(신청) |
| `OWN_CAR_AGE_MISSING` | `FLAG_OWN_CAR='Y'`이고 `OWN_CAR_AGE`가 결측이면 1 | 0/1 | 차량 보유자의 연식 결측이 아님 | 차량 보유 여부를 판정할 수 없음 | 가능(신청) |
| `APP_CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL`, 소득이 양수일 때 계산 | 비율 | 소득은 관측됐지만 대출금액이 0 | 소득이 결측·0 이하이거나 대출금액이 결측 | 가능(신청) |
| `APP_ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL`, 소득이 양수일 때 계산 | 비율 | 소득은 관측됐지만 연간 상환액이 0 | 소득이 결측·0 이하이거나 상환액이 결측 | 가능(신청) |
| `APP_CREDIT_ANNUITY_RATIO` | `AMT_CREDIT / AMT_ANNUITY`, 연간 상환액이 양수일 때 계산 | 비율 | 유효한 분모에서 대출금액이 0 | 상환액이 결측·0 이하이거나 대출금액이 결측 | 가능(신청) |
| `APP_CREDIT_GOODS_RATIO` | `AMT_GOODS_PRICE / AMT_CREDIT`, 대출금액이 양수일 때 계산 | 비율 | 유효한 분모에서 상품가격이 0 | 대출금액이 결측·0 이하이거나 상품가격이 결측 | 가능(신청) |
| `APP_INCOME_PER_FAMILY_MEMBER` | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`, 가족 수가 양수일 때 계산 | 금액/명 | 유효한 가족 수에서 소득이 0 | 가족 수가 결측·0 이하이거나 소득이 결측 | 가능(신청) |
| `APP_AGE_YEARS` | `-DAYS_BIRTH / 365.25`, `DAYS_BIRTH < 0`일 때 계산 | 년 | 유효 조건에서는 발생하지 않음 | 생년 상대일수가 결측이거나 0 이상 | 가능(신청) |
| `APP_EMPLOYED_YEARS` | `-DAYS_EMPLOYED / 365.25`, 음수인 실제 근속일만 사용 | 년 | 유효 조건에서는 발생하지 않음 | sentinel, 결측 또는 0 이상의 비정상 근속일 | 가능(신청) |
| `APP_EMPLOYED_AGE_RATIO` | `DAYS_EMPLOYED / DAYS_BIRTH`, 두 값이 모두 유효한 음수일 때 계산 | 비율 | 유효 조건에서는 일반적으로 발생하지 않음 | 생년·근속일 중 하나가 결측·비정상이거나 근속일이 sentinel | 가능(신청) |
| `APP_EXT_SOURCE_OBSERVED_COUNT` | `EXT_SOURCE_1~3` 중 결측이 아닌 값의 수 | 개 | 외부 신용평가값 3개가 모두 결측 | 산식상 발생하지 않음 | 가능(신청) |
| `APP_EXT_SOURCE_MEAN` | 관측된 `EXT_SOURCE_1~3`의 행별 산술평균 | 원본 정규화 점수 | 관측된 외부 점수의 평균이 실제로 0 | 세 외부 점수가 모두 결측 | 가능(신청) |

## 3. V2 외부 신용이력 파생 피처 37개

`bureau.csv`는 외부 신용거래 한 건이 한 행이다. 현재 신청 고객의 행만 남기고 `DAYS_CREDIT_UPDATE <= 0` 조건을 적용한 뒤 `SK_ID_CURR`별로 집계한다. 아래에서 “적격 이력”은 이 두 조건을 모두 만족하는 행을 뜻한다.

외부 신용이력의 건수·상태·일수 피처는 모든 통화의 적격 이력을 사용한다. 금액 피처와 금액 관측 건수는 `currency 1`만 사용하며, 다른 통화의 존재는 `BUREAU_NON_PRIMARY_CURRENCY_COUNT`로 따로 남긴다.

| 피처명 | 산식·정의 | 단위 | `0`의 의미 | `NULL`의 의미 | 예측시점 |
|---|---|---|---|---|---|
| `BUREAU_HAS_HISTORY` | 고객에게 적격 bureau 행이 하나 이상 있으면 1 | 0/1 | 적격 외부 신용이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_RECORD_COUNT` | 적격 bureau 행 수 `COUNT(*)` | 건 | 적격 외부 신용이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_LOAN_COUNT` | 고유 `SK_ID_BUREAU` 수 | 건 | 적격 외부 대출 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_ACTIVE_COUNT` | `CREDIT_ACTIVE='Active'`인 행 수 | 건 | 진행 중인 외부 신용거래가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_CLOSED_COUNT` | `CREDIT_ACTIVE='Closed'`인 행 수 | 건 | 종료된 외부 신용거래가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_SOLD_COUNT` | `CREDIT_ACTIVE='Sold'`인 행 수 | 건 | 매각 상태 거래가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_BAD_DEBT_COUNT` | `CREDIT_ACTIVE='Bad debt'`인 행 수 | 건 | 부실채권 상태 거래가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_CREDIT_TYPE_COUNT` | 고유 `CREDIT_TYPE` 수 | 개 | 적격 외부 신용이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_NON_PRIMARY_CURRENCY_COUNT` | `CREDIT_CURRENCY <> 'currency 1'`인 행 수 | 건 | 비주통화 거래가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_OVERDUE_LOAN_COUNT` | `CREDIT_DAY_OVERDUE > 0` 또는 `AMT_CREDIT_SUM_OVERDUE > 0`인 행 수 | 건 | 현재 연체 조건을 만족하는 거래가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_ACTIVE_RATIO` | `BUREAU_ACTIVE_COUNT / BUREAU_RECORD_COUNT` | 비율 | 이력은 있으나 Active 거래가 없음 | 적격 이력이 없음 | 가능(외부이력) |
| `BUREAU_OVERDUE_LOAN_RATIO` | `BUREAU_OVERDUE_LOAN_COUNT / BUREAU_RECORD_COUNT` | 비율 | 이력은 있으나 현재 연체 거래가 없음 | 적격 이력이 없음 | 가능(외부이력) |
| `BUREAU_DAYS_CREDIT_MEAN` | 적격 이력의 `DAYS_CREDIT` 평균 | 상대일 | 평균이 현재 신청일 0과 같음 | 적격 이력이 없음 | 가능(외부이력) |
| `BUREAU_DAYS_CREDIT_MIN` | 적격 이력의 `DAYS_CREDIT` 최솟값, 가장 오래된 신청일 | 상대일 | 가장 오래된 이력이 현재 신청일에 발생 | 적격 이력이 없음 | 가능(외부이력) |
| `BUREAU_DAYS_CREDIT_MAX` | 적격 이력의 `DAYS_CREDIT` 최댓값, 가장 최근 신청일 | 상대일 | 최근 이력이 현재 신청일에 발생 | 적격 이력이 없음 | 가능(외부이력) |
| `BUREAU_DAYS_SINCE_RECENT_CREDIT` | `-MAX(DAYS_CREDIT)` | 일 | 최근 외부 신용신청이 현재 신청일에 발생 | 적격 이력이 없음 | 가능(외부이력) |
| `BUREAU_DAYS_OVERDUE_MEAN` | `CREDIT_DAY_OVERDUE` 평균 | 일 | 모든 적격 거래의 현재 연체일이 0 | 적격 이력이 없음 | 가능(외부이력) |
| `BUREAU_DAYS_OVERDUE_MAX` | `CREDIT_DAY_OVERDUE` 최댓값 | 일 | 모든 적격 거래의 현재 연체일이 0 | 적격 이력이 없음 | 가능(외부이력) |
| `BUREAU_PROLONG_COUNT_SUM` | `CNT_CREDIT_PROLONG` 합계 | 회 | 연장 이력이 없거나 적격 이력 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_CREDIT_AMOUNT_OBSERVED_COUNT` | `currency 1` 중 `AMT_CREDIT_SUM` 관측 행 수 | 건 | 주통화의 관측 가능한 신용금액이 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_CREDIT_AMOUNT_SUM` | `currency 1`의 `AMT_CREDIT_SUM` 합계 | 금액 | 관측된 주통화 신용금액 합이 실제로 0 | 주통화 신용금액 관측값이 없음 | 가능(외부이력) |
| `BUREAU_CREDIT_AMOUNT_MEAN` | `currency 1`의 `AMT_CREDIT_SUM` 평균 | 금액 | 관측된 주통화 신용금액 평균이 실제로 0 | 주통화 신용금액 관측값이 없음 | 가능(외부이력) |
| `BUREAU_CREDIT_AMOUNT_MAX` | `currency 1`의 `AMT_CREDIT_SUM` 최댓값 | 금액 | 관측된 주통화 신용금액 최댓값이 실제로 0 | 주통화 신용금액 관측값이 없음 | 가능(외부이력) |
| `BUREAU_DEBT_OBSERVED_COUNT` | `currency 1` 중 `AMT_CREDIT_SUM_DEBT` 관측 행 수 | 건 | 주통화의 관측 가능한 잔액이 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_DEBT_SUM` | `currency 1`의 `AMT_CREDIT_SUM_DEBT` 합계 | 금액 | 관측된 잔액의 순합이 0 | 주통화 잔액 관측값이 없음 | 가능(외부이력) |
| `BUREAU_DEBT_MEAN` | `currency 1`의 `AMT_CREDIT_SUM_DEBT` 평균 | 금액 | 관측된 잔액 평균이 0 | 주통화 잔액 관측값이 없음 | 가능(외부이력) |
| `BUREAU_DEBT_MAX` | `currency 1`의 `AMT_CREDIT_SUM_DEBT` 최댓값 | 금액 | 관측된 잔액 최댓값이 0 | 주통화 잔액 관측값이 없음 | 가능(외부이력) |
| `BUREAU_OVERDUE_AMOUNT_SUM` | `currency 1`의 `AMT_CREDIT_SUM_OVERDUE` 합계 | 금액 | 주통화 거래가 있으나 현재 연체금액 합이 0 | 주통화 거래가 없음 | 가능(외부이력) |
| `BUREAU_OVERDUE_AMOUNT_MAX` | `currency 1`의 `AMT_CREDIT_SUM_OVERDUE` 최댓값 | 금액 | 주통화 거래가 있으나 현재 연체금액 최댓값이 0 | 주통화 거래가 없음 | 가능(외부이력) |
| `BUREAU_MAX_OVERDUE_OBSERVED_COUNT` | `currency 1` 중 `AMT_CREDIT_MAX_OVERDUE` 관측 행 수 | 건 | 주통화의 과거 최대 연체금액 관측값이 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_MAX_OVERDUE_AMOUNT` | `currency 1`의 `AMT_CREDIT_MAX_OVERDUE` 최댓값 | 금액 | 관측된 과거 최대 연체금액이 0 | 주통화의 해당 관측값이 없음 | 가능(외부이력) |
| `BUREAU_ACTIVE_CREDIT_SUM` | Active이면서 `currency 1`인 `AMT_CREDIT_SUM` 합계 | 금액 | 관측된 진행 중 주통화 신용금액 합이 0 | 해당 거래가 없거나 금액이 모두 결측 | 가능(외부이력) |
| `BUREAU_ACTIVE_DEBT_SUM` | Active이면서 `currency 1`인 `AMT_CREDIT_SUM_DEBT` 합계 | 금액 | 관측된 진행 중 주통화 잔액 합이 0 | 해당 거래가 없거나 잔액이 모두 결측 | 가능(외부이력) |
| `BUREAU_DEBT_CREDIT_RATIO` | `currency 1`에서 신용금액과 잔액이 함께 관측된 행들의 `SUM(AMT_CREDIT_SUM_DEBT) / SUM(AMT_CREDIT_SUM)` | 비율 | 대응되는 관측 잔액 합이 0 | 대응 행이 없거나 신용금액 합이 0 | 가능(외부이력) |
| `BUREAU_ANNUITY_OBSERVED_COUNT` | `currency 1` 중 `AMT_ANNUITY` 관측 행 수 | 건 | 주통화의 관측 가능한 연간 상환액이 없음 | 산식상 발생하지 않음 | 가능(외부이력) |
| `BUREAU_ANNUITY_SUM` | `currency 1`의 `AMT_ANNUITY` 합계 | 금액 | 관측된 주통화 연간 상환액 합이 0 | 주통화 연간 상환액 관측값이 없음 | 가능(외부이력) |
| `BUREAU_ANNUITY_MEAN` | `currency 1`의 `AMT_ANNUITY` 평균 | 금액 | 관측된 주통화 연간 상환액 평균이 0 | 주통화 연간 상환액 관측값이 없음 | 가능(외부이력) |

## 4. V3 과거 납부행동 파생 피처 29개

`installments_payments.csv`의 한 행은 납부회차가 아니라 실제 결제행이다. 한 납부회차가 분할납부로 여러 행에 나타날 수 있으므로 다음 두 단계를 거쳐야 한다.

1. 결제행을 `(SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER)`로 묶어 납부회차 한 행으로 만든다.
2. 납부회차를 `SK_ID_CURR`로 다시 묶어 고객 한 행으로 만든다.

같은 납부회차의 예정일과 예정금액은 각각 하나여야 한다. 예정일은 `MIN(DAYS_INSTALMENT)`, 예정금액은 `MAX(AMT_INSTALMENT)`로 가져오되 사전 검증에서 값의 일관성을 확인한다. 실제 납부금액은 분할 결제행의 `SUM(AMT_PAYMENT)`, 최종 납부일은 `MAX(DAYS_ENTRY_PAYMENT)`이다. 한 결제행이라도 실제 납부일 또는 금액이 결측이면 해당 납부회차의 납부금액·납부일·지연·부족납부 파생값을 `NULL`로 둔다.

납부회차 내부 계산은 다음과 같다.

| 내부 계산값 | 산식 |
|---|---|
| 최종 납부일 | `MAX(DAYS_ENTRY_PAYMENT)` |
| 예정금액 | `MAX(AMT_INSTALMENT)` |
| 실제 납부금액 | `SUM(AMT_PAYMENT)` |
| 지연일 | `MAX(최종 납부일 - 예정일, 0)` |
| 부족납부금액 | `MAX(예정금액 - 실제 납부금액, 0)` |
| 납부율 | `실제 납부금액 / 예정금액`, 예정금액이 양수일 때 계산 |
| 부족납부 여부 | `실제 납부금액 + 0.01 < 예정금액`이면 1 |

| 피처명 | 산식·정의 | 단위 | `0`의 의미 | `NULL`의 의미 | 예측시점 |
|---|---|---|---|---|---|
| `INST_HAS_HISTORY` | 고객에게 과거 납부 결제행이 하나 이상 있으면 1 | 0/1 | 과거 납부이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_SCHEDULE_COUNT` | 고객의 고유 납부회차 수 | 회차 | 납부이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_PREV_LOAN_COUNT` | 고유 `SK_ID_PREV` 수 | 건 | 과거 대출 납부이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_PAYMENT_EVENT_COUNT` | 납부회차에 포함된 원본 결제행 수의 고객별 합계 | 건 | 납부이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_PAYMENT_DATE_OBSERVED_SCHEDULE_COUNT` | 결제행의 납부일·금액이 모두 관측된 납부회차 수 | 회차 | 비교 가능한 납부회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_PAYMENT_AMOUNT_OBSERVED_SCHEDULE_COUNT` | 결제행의 납부일·금액이 모두 관측된 납부회차 수 | 회차 | 비교 가능한 납부회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_MISSING_PAYMENT_SCHEDULE_COUNT` | 납부일 또는 금액이 결측인 결제행을 하나 이상 포함한 납부회차 수 | 회차 | 결측 결제행을 포함한 납부회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_MISSING_PAYMENT_RATIO` | 결측 결제행 포함 납부회차 수 / 전체 납부회차 수 | 비율 | 이력은 있으나 결측 포함 회차가 없음 | 납부회차가 없음 | 가능(납부이력) |
| `INST_LATE_SCHEDULE_COUNT` | `지연일 > 0`인 관측 가능 납부회차 수 | 회차 | 지연 회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_LATE_RATIO` | 지연 회차 수 / 지연일 계산 가능 회차 수 | 비율 | 계산 가능 회차가 있지만 지연 회차 없음 | 지연일을 계산할 수 있는 회차가 없음 | 가능(납부이력) |
| `INST_DAYS_LATE_MEAN` | 납부회차별 지연일 평균 | 일 | 계산 가능 회차가 모두 예정일 이전 또는 당일 납부 | 지연일을 계산할 수 있는 회차가 없음 | 가능(납부이력) |
| `INST_DAYS_LATE_MAX` | 납부회차별 지연일 최댓값 | 일 | 계산 가능 회차가 모두 예정일 이전 또는 당일 납부 | 지연일을 계산할 수 있는 회차가 없음 | 가능(납부이력) |
| `INST_DAYS_LATE_SUM` | 납부회차별 지연일 합계 | 일 | 계산 가능 회차의 지연일 합이 0 | 지연일을 계산할 수 있는 회차가 없음 | 가능(납부이력) |
| `INST_UNDERPAID_SCHEDULE_COUNT` | `실제 납부금액 + 0.01 < 예정금액`인 회차 수 | 회차 | 부족납부 회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_UNDERPAID_RATIO` | 부족납부 회차 수 / 부족납부 판정 가능 회차 수 | 비율 | 판정 가능 회차가 있지만 부족납부 없음 | 예정금액이 양수이고 납부금액이 관측된 회차가 없음 | 가능(납부이력) |
| `INST_SCHEDULED_AMOUNT_SUM` | 모든 납부회차의 예정금액 합계 | 금액 | 관측된 예정금액 합이 실제로 0 | 납부이력 없음 또는 예정금액 관측값 없음 | 가능(납부이력) |
| `INST_PAID_AMOUNT_SUM` | 결측 결제행이 없는 납부회차의 실제 납부금액 합계 | 금액 | 관측 가능한 실제 납부금액 합이 0 | 이력 없음 또는 납부금액을 계산할 수 있는 회차가 없음 | 가능(납부이력) |
| `INST_PAYMENT_GAP_SUM` | 판정 가능한 납부회차별 부족납부금액 합계 | 금액 | 판정 가능한 회차의 부족납부금액 합이 0 | 판정 가능한 회차가 없음 | 가능(납부이력) |
| `INST_PAYMENT_GAP_MAX` | 판정 가능한 납부회차별 부족납부금액 최댓값 | 금액 | 판정 가능한 회차의 부족납부금액이 모두 0 | 판정 가능한 회차가 없음 | 가능(납부이력) |
| `INST_PAYMENT_RATIO` | 관측 가능 회차의 `SUM(실제 납부금액) / SUM(예정금액)` | 비율 | 관측 가능한 회차의 납부금액 합이 0 | 관측 가능 회차가 없거나 대응 예정금액 합이 0 | 가능(납부이력) |
| `INST_DAYS_SINCE_RECENT_DUE` | `-MAX(예정일 상대일수)` | 일 | 가장 최근 예정일이 현재 신청일 | 납부이력 없음 | 가능(납부이력) |
| `INST_OLDEST_DUE_AGE_DAYS` | `-MIN(예정일 상대일수)` | 일 | 가장 오래된 예정일이 현재 신청일 | 납부이력 없음 | 가능(납부이력) |
| `INST_HISTORY_SPAN_DAYS` | `MAX(예정일) - MIN(예정일)` | 일 | 납부회차가 하나이거나 모든 예정일이 같음 | 납부이력 없음 | 가능(납부이력) |
| `INST_LAST_365_SCHEDULE_COUNT` | 예정일이 현재 신청일 전 365일 이내인 납부회차 수 | 회차 | 최근 365일 이내 회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_LAST_365_LATE_COUNT` | 최근 365일 이내이며 `지연일 > 0`인 회차 수 | 회차 | 최근 365일 이내 지연 회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_LAST_365_LATE_RATIO` | 최근 365일 지연 회차 수 / 해당 기간의 지연일 계산 가능 회차 수 | 비율 | 계산 가능 회차가 있지만 지연 없음 | 해당 기간에 지연일을 계산할 수 있는 회차가 없음 | 가능(납부이력) |
| `INST_LAST_730_SCHEDULE_COUNT` | 예정일이 현재 신청일 전 730일 이내인 납부회차 수 | 회차 | 최근 730일 이내 회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_LAST_730_LATE_COUNT` | 최근 730일 이내이며 `지연일 > 0`인 회차 수 | 회차 | 최근 730일 이내 지연 회차가 없거나 이력 없음 | 산식상 발생하지 않음 | 가능(납부이력) |
| `INST_LAST_730_LATE_RATIO` | 최근 730일 지연 회차 수 / 해당 기간의 지연일 계산 가능 회차 수 | 비율 | 계산 가능 회차가 있지만 지연 없음 | 해당 기간에 지연일을 계산할 수 있는 회차가 없음 | 가능(납부이력) |

## 5. 사용 시 주의사항

- V1·V2·V3의 파생 피처는 고객 내부 정보만 사용하는 결정론적 산식이다. 결측 대치값, 범주 목록, 스케일 값과 모델 파라미터는 이 단계에서 계산하지 않는다.
- V2와 V3는 각각 고객 단위로 집계한 뒤 V1에 `LEFT JOIN`한다. 원본 일대다 테이블을 신청정보에 직접 조인하지 않는다.
- `BUREAU_HAS_HISTORY=0`과 `INST_HAS_HISTORY=0`은 관측 가능한 이력이 없다는 뜻이지 금융거래가 실제로 전혀 없었다는 단정이 아니다.
- 외부 신용잔액에는 원본 음수값이 존재한다. 업무 의미를 확인하지 않은 채 0으로 자르지 않으므로 `BUREAU_DEBT_*`가 음수가 될 수 있다.
- `INST_PAYMENT_RATIO`는 초과납부가 있으면 1보다 클 수 있다. 유효값을 1로 강제 제한하지 않는다.
- 신규 파생 피처의 분포·결측·TARGET 관계는 학습 파티션에서만 분석한다. validation과 test는 키·스키마·행 수·산식 불변식 검증에만 사용한다.
