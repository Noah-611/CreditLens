# Stage 7-1 공통 예측 함수와 입출력 계약

## 1. 무엇을 만드는가

학습이 끝난 모델을 화면과 API에서 똑같이 사용하기 위한 연결부다. 모델을 새로
학습하거나 성능을 높이는 단계가 아니다. 사용자는 고객별로 준비된 피처를 보내고,
프로그램은 입력 검사를 거쳐 상환곤란 확률·위험등급·우선검토 여부를 반환한다.

`준비된 V3 피처 → 입력 검사 → 저장된 전처리 → 저장된 LightGBM → 고정 위험구간 → 결과`

- 분석가: 준비된 고객 피처로 모델의 출력을 확인한다.
- 심사 담당자 역할의 시연 사용자: 확률과 위험구간을 우선검토 참고 정보로 본다.
- 운영자 역할의 시연 사용자: 입력 오류, 결측과 미등록 범주 경고를 확인한다.

원본 거래기록을 고객 단위로 집계하는 기능은 Stage 3의 역할이다. 이 인터페이스는
원본 CSV나 소득·대출금액 두 항목만 받는 함수가 아니다. **V3 피처 198개 전체**가
필요하며, 값이 없는 경우 허용된 항목에만 `null`을 명시한다. 없는 항목을 임의로
생성하지 않는다. 식별자·정답·분할·성별은 모델에 전달하지 않는다.

## 2. 고정된 구성과 이번 범위

| 항목 | 내용 |
|---|---|
| 모델 | `creditlens-v3-lightgbm-v1` (Stage 6 동결 V3 LightGBM) |
| 입력 | 수치형 184개 + 범주형 14개 = 198개 |
| 전처리 결과 | 학습 당시의 420개 구성요소, 재학습 없이 변환만 수행 |
| 보정 | 저장된 identity 보정기, 추가 확률 조정 없음 |
| 입력 계약 버전 | `creditlens-inference-v1` |
| 피처 버전 | `v3` |
| 이번 구현 | 입력 검사·공통 예측·합성 예시·CLI 확인·자동 테스트 |
| 다음 구현 | 7-2 화면/API 및 개별 예측요인 표시, 7-3 배치 실행·운영 검수·모니터링 |

현재 반환값에는 개별 SHAP 위험요인이 없다. 추후 이를 추가할 때도 같은 동결 모델을
사용한다. 이번 작업은 train/validation/test 파일을 읽지 않고 **수작업 합성 입력**으로
검증했다. 합성 예시 점수는 모델 성능이나 실제 고객 사례가 아니다.

## 3. 입력 계약

최상위 객체에는 `schema_version`, `feature_version`, `records`만 허용한다.
`records`는 1~1,000건이다. 각 행에는 `record_id`와 `features`만 둔다.

- `record_id`: 요청 내에서 유일한 1~128자 문자열. 빈 문자열·제어문자는 금지한다.
  응답을 입력과 연결하는 용도이며 모델 피처가 아니다. 시연에는 합성 식별자를 쓴다.
- `features`: 아래 목록의 198개 키를 정확히 포함한다. 키 순서는 자유이며 내부에서
  학습 당시 순서로 정렬한다. 추가 키와 누락 키는 모두 오류다.
- 수치형: JSON 숫자 또는 허용된 `null`. 숫자 문자열, `true/false`, NaN, Infinity와
  float32 표현 범위 초과값은 거절한다. 소수점이 없는 `1.0`도 정수값으로 인정한다.
- 범주형: 1~128자의 비어 있지 않은 문자열 또는 `null`. 대소문자와 공백을 임의로
  변경하지 않는다. 내부 전용 토큰 `__MISSING__`, `__RARE__` 입력은 거절한다.
- 미등록 범주: 오류로 모두 막지 않고 `UNKNOWN_CATEGORIES` 경고를 반환한다.
  기존 전처리의 `handle_unknown='ignore'` 규칙을 그대로 적용한다. 해당 범주의
  원핫 값은 모두 0이 된다. train에서 드물게 나타난 범주는 기존 rare 규칙을 따른다.
- 금액은 원본의 금액 단위다. 원화로 해석하거나 환율을 적용하지 않는다.
- `EMERGENCYSTATE_MODE`는 모델 입력에서 수치형 0/1/null이다. 원본의
  `No/Yes` 문자열을 그대로 보내지 않고 각각 0/1로 표현한다. 원본 데이터의
  자료형과 최종 모델 입력의 자료형이 모두 같지는 않다.

완전한 합성 JSON은 다음 명령으로 확인한다. 문서의 축약 예시를 API 입력으로
오인하지 않도록 198개 전체를 코드로 생성한다.

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.inference --example
```

모든 필드의 기계 판독용 명세는 [입력 계약 JSON](../reports/stage7_input_contract.json)에
있다. 이는 JSON Schema 표준 파일이 아니라 프로젝트의 필드 명세 스냅샷이다.
실제 검증은 `InputContract`가 수행하며, 테스트로 두 내용의 일치를 확인한다.
피처의 업무적 의미와 산식은 [원본 데이터 사전](Data_Dictionary.md)과
[파생 피처 사전](Feature_Dictionary.md)을 따른다.

### 필드 간 검사와 한계

- 차량연식과 해당 없음·결측 플래그가 모순되면 거절한다.
- 재직일 sentinel `365243`은 원본 그대로 받지 않는다. Stage 3처럼 `null`과
  sentinel 플래그로 표현해야 한다.
- 외부 점수 3개의 관측 개수·평균이 입력된 요약값과 일치하는지 검사한다.
- 이력 존재 플래그와 대표 건수를 비교한다. 이력이 없으면 해당 건수는 0,
  평균·금액·비율 등은 `null`이어야 한다. 0과 결측은 다르다.
- 음수 외부 신용잔액과 1을 넘는 납부율은 유효할 수 있으므로 일괄 잘라내지 않는다.
- 모든 파생변수의 산식과 원본 이력의 진위를 재검증하지는 않는다. Stage 3 방식으로
  만들어진 피처를 전제로 하며, 계약 통과가 데이터의 완전한 업무적 정합성이나
  예측 신뢰도를 보장하지 않는다. 결측이 많은 입력도 점수는 나올 수 있으므로
  경고와 데이터 품질을 함께 확인해야 한다.

## 4. 출력 계약

응답은 `schema_version`, `feature_version`, `model_release`,
`feature_schema_sha256`, `policy`, `predictions`, `notice`를 포함한다.
`feature_schema_sha256`은 Stage 6에 기록된 V3 마트 스키마 해시이며, 새 요청
데이터의 내용 해시나 본 문서의 해시는 아니다.

`predictions`는 입력과 같은 순서·건수로 반환한다.

| 필드 | 형식·의미 |
|---|---|
| `record_id` | 입력 연결용 문자열 |
| `repayment_difficulty_probability` | [0, 1] 실수, 공개 데이터의 TARGET=1 추정 확률 |
| `risk_band` | `low`, `medium`, `high`; 공식 신용등급이 아닌 모델의 시연용 구간 |
| `priority_review` | 고위험 경계 이상이면 `true`; 승인·거절이 아닌 우선검토 표시 |
| `warnings` | `MISSING_VALUES`와 개수, `UNKNOWN_CATEGORIES`와 해당 필드명 |

`policy`에는 아래 경계와 `comparison`, `calibration_method`,
`automatic_credit_decision=false`를 포함한다.

| 위험구간 | 전체 정밀도 점수의 조건 |
|---|---|
| low | 확률 < 0.08251815922901691 |
| medium | 0.08251815922901691 ≤ 확률 < 0.18607729049496233 |
| high | 확률 ≥ 0.18607729049496233 |

경계와 같은 점수는 상위 구간에 포함한다. 화면 표시용 반올림 전에 판정한다.
배치가 달라져도 같은 입력의 판정은 같다. 새 배치에서 반드시 10%가 고위험으로
선택되는 규칙이 아니다. `notice`에는 실제 승인·거절 용도가 아님을 표시한다.
원본 피처 값은 응답에 복사하지 않는다.

## 5. 오류 처리

입력이 하나라도 잘못되면 전체 요청을 중단하며 일부 행만 예측하지 않는다.
`InputContractError.as_dict()`는 `code`, 0부터 시작하는 `row`, `field`를 제공한다.
행·필드에 해당하지 않는 오류는 그 값을 `null`로 둔다. 오류 메시지에는 입력값
자체를 넣지 않는다. HTTP 상태 코드 변환은 7-2 API에서 구현한다.

| 오류 코드 | 원인 |
|---|---|
| REQUEST_FIELDS / RECORD_FIELDS / FEATURE_FIELDS | 최상위·행·피처 키 누락 또는 추가 |
| SCHEMA_VERSION / FEATURE_VERSION | 지원하지 않는 버전 |
| BATCH_SIZE | 빈 배치·1,000건 초과·목록이 아닌 입력 |
| RECORD_ID / DUPLICATE_RECORD_ID | 잘못된 또는 중복 식별자 |
| NULL_NOT_ALLOWED | 결측 불허 필드의 null |
| NUMERIC_TYPE / NUMERIC_RANGE / INTEGER_REQUIRED | 숫자 자료형·범위·정수 규칙 위반 |
| CATEGORY_VALUE | 범주 자료형·빈 값·예약 토큰 위반 |
| CAR_AGE_FLAGS / CAR_OWNERSHIP_FLAGS | 차량연식·보유 여부·플래그 모순 |
| EMPLOYMENT_SENTINEL | 정제되지 않은 재직일 또는 sentinel 플래그 모순 |
| EXTERNAL_SCORE_SUMMARY | 외부 점수 개수·평균 모순 |
| HISTORY_FLAG_COUNT / ABSENT_HISTORY_VALUES | 이력 존재·건수·결측 규칙 모순 |

모델 파일·해시·버전 불일치는 사용자 입력 오류가 아니라 서비스 시작 실패다.
자동 재학습이나 다른 모델로의 대체 실행을 하지 않는다.

## 6. 실행과 파일 연결

프로젝트 루트에서 기존 `.venv`와 로컬 동결 bundle을 사용한다. 별도 패키지를
설치하지 않았다. 동결 모델 로더는 Python과 관련 패키지 버전까지 비교하므로
기존 의존성을 임의로 업데이트하면 로드를 중단한다. 모델 파일은 Git에 없으며,
없는 환경에서는 Stage 5~6 산출물을 먼저 준비해야 한다.

```bash
# 모델·전처리·정책·실행환경 확인 (데이터셋 조회 없음)
PYTHONPATH=src .venv/bin/python -m creditlens.modeling.finalize_stage6 --verify

# 입력 명세 또는 합성 예측 결과 확인
PYTHONPATH=src .venv/bin/python -m creditlens.inference --contract
PYTHONPATH=src .venv/bin/python -m creditlens.inference --demo

# 자동 검증
.venv/bin/python -m pytest tests/test_stage7_inference.py -q
```

Python에서 사용할 때는 서비스를 한 번 불러온 뒤 같은 인스턴스를 재사용한다.

```python
from creditlens.inference.demo import synthetic_request
from creditlens.inference.predictor import PredictionService

service = PredictionService.from_frozen()
request = synthetic_request(service.contract)
response = service.predict(request)
```

- [contract.py](../src/creditlens/inference/contract.py): 필드 명세·자료형·입력 검증.
- [predictor.py](../src/creditlens/inference/predictor.py): 무결성 확인 후 로드·공통 예측·경고.
- [demo.py](../src/creditlens/inference/demo.py): 실제 고객을 복사하지 않은 합성 예시.
- [__main__.py](../src/creditlens/inference/__main__.py): 계약·예시·예측 CLI.
- [자동 테스트](../tests/test_stage7_inference.py): 정상·오류·경계·단건/배치·동결 모델 일치.

서비스는 고객 입력이나 예측을 파일로 저장하지 않는다. 실제 입력·출력을 따로
저장해야 할 경우 Git 제외 경로인 `data/processed/` 안에 둔다. 요청에서 모델
파일 경로를 지정하거나 업로드된 pickle/joblib을 로드하는 기능은 제공하지 않는다.
로컬 신뢰 모델의 해시 검증은 실수로 인한 변경 탐지이며 전자서명은 아니다.

### 7-1 검증 결과

- 새 추론 테스트 52개 통과. 모델 없는 환경에서는 스텁 기반 계약 테스트가 실행되고
  로컬 동결 모델 통합 테스트 1개만 건너뛴다. 현재 환경에서는 그 통합 테스트도 통과했다.
- 프로젝트 전체 테스트 247개 통과. SHAP의 기존 외부 라이브러리 폐기 예정 경고
  3건이 있으며 테스트 실패는 없다.
- 합성 입력의 공통 함수 예측과 모델 직접 예측이 일치한다. 같은 입력의 단건·배치
  결과, 경계 포함 판정, 미등록 범주·결측 처리를 확인했다.
- 동결 모델·보정기·정책·전처리 소스·실행환경 검증 PASS. 신규 패키지 설치나
  모델 재학습은 하지 않았다.
- 문서 링크·198개 필드 문서화·Git 제외 경로를 점검했다. 실제 금융 데이터와
  고객별 예측 파일을 새로 생성하지 않았다.

## 7. 전체 피처 목록

모든 키는 필수다. `null 가능`은 키를 생략해도 된다는 뜻이 아니다.
`F`는 float32 최대 유한값(약 3.402823466 × 10³⁸)이다.

| 피처 | 자료형 | null 가능 | 허용 범위 |
|---|---|---|---|
| `CNT_CHILDREN` | integer | 가능 | 0 ~ F |
| `AMT_INCOME_TOTAL` | number | 가능 | 0 ~ F |
| `AMT_CREDIT` | number | 가능 | 0 ~ F |
| `AMT_ANNUITY` | number | 가능 | 0 ~ F |
| `AMT_GOODS_PRICE` | number | 가능 | 0 ~ F |
| `REGION_POPULATION_RELATIVE` | number | 가능 | 0 ~ 1 |
| `DAYS_BIRTH` | number | 가능 | -F ~ F |
| `DAYS_REGISTRATION` | number | 가능 | -F ~ F |
| `DAYS_ID_PUBLISH` | number | 가능 | -F ~ F |
| `OWN_CAR_AGE` | number | 가능 | 0 ~ F |
| `FLAG_MOBIL` | integer | 가능 | 0 ~ 1 |
| `FLAG_EMP_PHONE` | integer | 가능 | 0 ~ 1 |
| `FLAG_WORK_PHONE` | integer | 가능 | 0 ~ 1 |
| `FLAG_CONT_MOBILE` | integer | 가능 | 0 ~ 1 |
| `FLAG_PHONE` | integer | 가능 | 0 ~ 1 |
| `FLAG_EMAIL` | integer | 가능 | 0 ~ 1 |
| `CNT_FAM_MEMBERS` | integer | 가능 | 0 ~ F |
| `REGION_RATING_CLIENT` | number | 가능 | -F ~ F |
| `REGION_RATING_CLIENT_W_CITY` | number | 가능 | -F ~ F |
| `HOUR_APPR_PROCESS_START` | integer | 가능 | 0 ~ 23 |
| `REG_REGION_NOT_LIVE_REGION` | integer | 가능 | 0 ~ 1 |
| `REG_REGION_NOT_WORK_REGION` | integer | 가능 | 0 ~ 1 |
| `LIVE_REGION_NOT_WORK_REGION` | integer | 가능 | 0 ~ 1 |
| `REG_CITY_NOT_LIVE_CITY` | integer | 가능 | 0 ~ 1 |
| `REG_CITY_NOT_WORK_CITY` | integer | 가능 | 0 ~ 1 |
| `LIVE_CITY_NOT_WORK_CITY` | integer | 가능 | 0 ~ 1 |
| `EXT_SOURCE_1` | number | 가능 | 0 ~ 1 |
| `EXT_SOURCE_2` | number | 가능 | 0 ~ 1 |
| `EXT_SOURCE_3` | number | 가능 | 0 ~ 1 |
| `APARTMENTS_AVG` | number | 가능 | -F ~ F |
| `BASEMENTAREA_AVG` | number | 가능 | -F ~ F |
| `YEARS_BEGINEXPLUATATION_AVG` | number | 가능 | -F ~ F |
| `YEARS_BUILD_AVG` | number | 가능 | -F ~ F |
| `COMMONAREA_AVG` | number | 가능 | -F ~ F |
| `ELEVATORS_AVG` | number | 가능 | -F ~ F |
| `ENTRANCES_AVG` | number | 가능 | -F ~ F |
| `FLOORSMAX_AVG` | number | 가능 | -F ~ F |
| `FLOORSMIN_AVG` | number | 가능 | -F ~ F |
| `LANDAREA_AVG` | number | 가능 | -F ~ F |
| `LIVINGAPARTMENTS_AVG` | number | 가능 | -F ~ F |
| `LIVINGAREA_AVG` | number | 가능 | -F ~ F |
| `NONLIVINGAPARTMENTS_AVG` | number | 가능 | -F ~ F |
| `NONLIVINGAREA_AVG` | number | 가능 | -F ~ F |
| `APARTMENTS_MODE` | number | 가능 | -F ~ F |
| `BASEMENTAREA_MODE` | number | 가능 | -F ~ F |
| `YEARS_BEGINEXPLUATATION_MODE` | number | 가능 | -F ~ F |
| `YEARS_BUILD_MODE` | number | 가능 | -F ~ F |
| `COMMONAREA_MODE` | number | 가능 | -F ~ F |
| `ELEVATORS_MODE` | number | 가능 | -F ~ F |
| `ENTRANCES_MODE` | number | 가능 | -F ~ F |
| `FLOORSMAX_MODE` | number | 가능 | -F ~ F |
| `FLOORSMIN_MODE` | number | 가능 | -F ~ F |
| `LANDAREA_MODE` | number | 가능 | -F ~ F |
| `LIVINGAPARTMENTS_MODE` | number | 가능 | -F ~ F |
| `LIVINGAREA_MODE` | number | 가능 | -F ~ F |
| `NONLIVINGAPARTMENTS_MODE` | number | 가능 | -F ~ F |
| `NONLIVINGAREA_MODE` | number | 가능 | -F ~ F |
| `APARTMENTS_MEDI` | number | 가능 | -F ~ F |
| `BASEMENTAREA_MEDI` | number | 가능 | -F ~ F |
| `YEARS_BEGINEXPLUATATION_MEDI` | number | 가능 | -F ~ F |
| `YEARS_BUILD_MEDI` | number | 가능 | -F ~ F |
| `COMMONAREA_MEDI` | number | 가능 | -F ~ F |
| `ELEVATORS_MEDI` | number | 가능 | -F ~ F |
| `ENTRANCES_MEDI` | number | 가능 | -F ~ F |
| `FLOORSMAX_MEDI` | number | 가능 | -F ~ F |
| `FLOORSMIN_MEDI` | number | 가능 | -F ~ F |
| `LANDAREA_MEDI` | number | 가능 | -F ~ F |
| `LIVINGAPARTMENTS_MEDI` | number | 가능 | -F ~ F |
| `LIVINGAREA_MEDI` | number | 가능 | -F ~ F |
| `NONLIVINGAPARTMENTS_MEDI` | number | 가능 | -F ~ F |
| `NONLIVINGAREA_MEDI` | number | 가능 | -F ~ F |
| `TOTALAREA_MODE` | number | 가능 | -F ~ F |
| `EMERGENCYSTATE_MODE` | integer | 가능 | 0 ~ 1 |
| `OBS_30_CNT_SOCIAL_CIRCLE` | number | 가능 | -F ~ F |
| `DEF_30_CNT_SOCIAL_CIRCLE` | number | 가능 | -F ~ F |
| `OBS_60_CNT_SOCIAL_CIRCLE` | number | 가능 | -F ~ F |
| `DEF_60_CNT_SOCIAL_CIRCLE` | number | 가능 | -F ~ F |
| `DAYS_LAST_PHONE_CHANGE` | number | 가능 | -F ~ F |
| `FLAG_DOCUMENT_2` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_3` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_4` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_5` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_6` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_7` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_8` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_9` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_10` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_11` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_12` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_13` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_14` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_15` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_16` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_17` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_18` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_19` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_20` | integer | 가능 | 0 ~ 1 |
| `FLAG_DOCUMENT_21` | integer | 가능 | 0 ~ 1 |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | number | 가능 | -F ~ F |
| `AMT_REQ_CREDIT_BUREAU_DAY` | number | 가능 | -F ~ F |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | number | 가능 | -F ~ F |
| `AMT_REQ_CREDIT_BUREAU_MON` | number | 가능 | -F ~ F |
| `AMT_REQ_CREDIT_BUREAU_QRT` | number | 가능 | -F ~ F |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | number | 가능 | -F ~ F |
| `DAYS_EMPLOYED` | number | 가능 | -F ~ F |
| `DAYS_EMPLOYED_SENTINEL` | integer | 가능 | 0 ~ 1 |
| `OWN_CAR_AGE_NOT_APPLICABLE` | integer | 불가 | 0 ~ 1 |
| `OWN_CAR_AGE_MISSING` | integer | 불가 | 0 ~ 1 |
| `APP_CREDIT_INCOME_RATIO` | number | 가능 | -F ~ F |
| `APP_ANNUITY_INCOME_RATIO` | number | 가능 | -F ~ F |
| `APP_CREDIT_ANNUITY_RATIO` | number | 가능 | -F ~ F |
| `APP_CREDIT_GOODS_RATIO` | number | 가능 | -F ~ F |
| `APP_INCOME_PER_FAMILY_MEMBER` | number | 가능 | -F ~ F |
| `APP_AGE_YEARS` | number | 가능 | 0 ~ F |
| `APP_EMPLOYED_YEARS` | number | 가능 | 0 ~ F |
| `APP_EMPLOYED_AGE_RATIO` | number | 가능 | -F ~ F |
| `APP_EXT_SOURCE_OBSERVED_COUNT` | integer | 불가 | 0 ~ 3 |
| `APP_EXT_SOURCE_MEAN` | number | 가능 | 0 ~ 1 |
| `BUREAU_HAS_HISTORY` | integer | 불가 | 0 ~ 1 |
| `BUREAU_RECORD_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_LOAN_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_ACTIVE_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_CLOSED_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_SOLD_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_BAD_DEBT_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_CREDIT_TYPE_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_NON_PRIMARY_CURRENCY_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_OVERDUE_LOAN_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_ACTIVE_RATIO` | number | 가능 | 0 ~ 1 |
| `BUREAU_OVERDUE_LOAN_RATIO` | number | 가능 | 0 ~ 1 |
| `BUREAU_DAYS_CREDIT_MEAN` | number | 가능 | -F ~ F |
| `BUREAU_DAYS_CREDIT_MIN` | number | 가능 | -F ~ F |
| `BUREAU_DAYS_CREDIT_MAX` | number | 가능 | -F ~ F |
| `BUREAU_DAYS_SINCE_RECENT_CREDIT` | number | 가능 | -F ~ F |
| `BUREAU_DAYS_OVERDUE_MEAN` | number | 가능 | -F ~ F |
| `BUREAU_DAYS_OVERDUE_MAX` | number | 가능 | -F ~ F |
| `BUREAU_PROLONG_COUNT_SUM` | integer | 불가 | 0 ~ F |
| `BUREAU_CREDIT_AMOUNT_OBSERVED_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_CREDIT_AMOUNT_SUM` | number | 가능 | -F ~ F |
| `BUREAU_CREDIT_AMOUNT_MEAN` | number | 가능 | -F ~ F |
| `BUREAU_CREDIT_AMOUNT_MAX` | number | 가능 | -F ~ F |
| `BUREAU_DEBT_OBSERVED_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_DEBT_SUM` | number | 가능 | -F ~ F |
| `BUREAU_DEBT_MEAN` | number | 가능 | -F ~ F |
| `BUREAU_DEBT_MAX` | number | 가능 | -F ~ F |
| `BUREAU_OVERDUE_AMOUNT_SUM` | number | 가능 | -F ~ F |
| `BUREAU_OVERDUE_AMOUNT_MAX` | number | 가능 | -F ~ F |
| `BUREAU_MAX_OVERDUE_OBSERVED_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_MAX_OVERDUE_AMOUNT` | number | 가능 | -F ~ F |
| `BUREAU_ACTIVE_CREDIT_SUM` | number | 가능 | -F ~ F |
| `BUREAU_ACTIVE_DEBT_SUM` | number | 가능 | -F ~ F |
| `BUREAU_DEBT_CREDIT_RATIO` | number | 가능 | -F ~ F |
| `BUREAU_ANNUITY_OBSERVED_COUNT` | integer | 불가 | 0 ~ F |
| `BUREAU_ANNUITY_SUM` | number | 가능 | -F ~ F |
| `BUREAU_ANNUITY_MEAN` | number | 가능 | -F ~ F |
| `INST_HAS_HISTORY` | integer | 불가 | 0 ~ 1 |
| `INST_SCHEDULE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_PREV_LOAN_COUNT` | integer | 불가 | 0 ~ F |
| `INST_PAYMENT_EVENT_COUNT` | integer | 불가 | 0 ~ F |
| `INST_PAYMENT_DATE_OBSERVED_SCHEDULE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_PAYMENT_AMOUNT_OBSERVED_SCHEDULE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_MISSING_PAYMENT_SCHEDULE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_MISSING_PAYMENT_RATIO` | number | 가능 | 0 ~ 1 |
| `INST_LATE_SCHEDULE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_LATE_RATIO` | number | 가능 | 0 ~ 1 |
| `INST_DAYS_LATE_MEAN` | number | 가능 | -F ~ F |
| `INST_DAYS_LATE_MAX` | number | 가능 | -F ~ F |
| `INST_DAYS_LATE_SUM` | number | 가능 | -F ~ F |
| `INST_UNDERPAID_SCHEDULE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_UNDERPAID_RATIO` | number | 가능 | 0 ~ 1 |
| `INST_SCHEDULED_AMOUNT_SUM` | number | 가능 | -F ~ F |
| `INST_PAID_AMOUNT_SUM` | number | 가능 | -F ~ F |
| `INST_PAYMENT_GAP_SUM` | number | 가능 | -F ~ F |
| `INST_PAYMENT_GAP_MAX` | number | 가능 | -F ~ F |
| `INST_PAYMENT_RATIO` | number | 가능 | -F ~ F |
| `INST_DAYS_SINCE_RECENT_DUE` | number | 가능 | -F ~ F |
| `INST_OLDEST_DUE_AGE_DAYS` | number | 가능 | -F ~ F |
| `INST_HISTORY_SPAN_DAYS` | number | 가능 | -F ~ F |
| `INST_LAST_365_SCHEDULE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_LAST_365_LATE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_LAST_365_LATE_RATIO` | number | 가능 | 0 ~ 1 |
| `INST_LAST_730_SCHEDULE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_LAST_730_LATE_COUNT` | integer | 불가 | 0 ~ F |
| `INST_LAST_730_LATE_RATIO` | number | 가능 | 0 ~ 1 |
| `NAME_CONTRACT_TYPE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `FLAG_OWN_CAR` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `FLAG_OWN_REALTY` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `NAME_TYPE_SUITE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `NAME_INCOME_TYPE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `NAME_EDUCATION_TYPE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `NAME_FAMILY_STATUS` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `NAME_HOUSING_TYPE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `OCCUPATION_TYPE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `WEEKDAY_APPR_PROCESS_START` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `ORGANIZATION_TYPE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `FONDKAPREMONT_MODE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `HOUSETYPE_MODE` | string | 가능 | 1~128자, 예약 토큰 제외 |
| `WALLSMATERIAL_MODE` | string | 가능 | 1~128자, 예약 토큰 제외 |
