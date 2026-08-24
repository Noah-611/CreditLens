# CreditLens 데이터 사전

> Stage 1 원본 데이터 검증 결과와 Kaggle 공식 컬럼 설명을 결합한 초안입니다.

## 작성 기준

- 생성 시각(UTC): `2026-08-24T06:52:21.619819+00:00`
- 설명 원본: `data/raw/HomeCredit_columns_description.csv`
- 원본 공식 설명은 의미 왜곡을 막기 위해 영문 그대로 보존했습니다.
- 관측 자료형·결측률·범위는 현재 로컬 원본을 전체 스캔한 실제 값입니다.
- 최종 전처리 방식은 Stage 2에서 확정합니다.

## 테이블 관계

- `application_train.csv`: 현재 신청 1건당 1행, `SK_ID_CURR` 유일
- `bureau.csv`: 외부 신용거래 1건당 1행, `SK_ID_BUREAU` 유일, 고객당 여러 행
- `installments_payments.csv`: 납부행위 1건당 1행, 분할납부 때문에 보장된 단일 행 기본키 없음
- 보조 테이블에는 `application_test.csv` 고객도 포함되므로 train 외 고객은 원본 오류가 아닙니다.

## `application_train.csv`

- 행 단위: 현재 대출 신청 1건당 1행
- 크기: 307,511행 × 122열

| 컬럼 | 의미 유형 | 관측 dtype | 결측 | 관측 범위/주요 값 | 공식 설명(영문) | Stage 2 메모 |
|---|---|---|---:|---|---|---|
| `SK_ID_CURR` | 식별자 | int64 | 0 (0.00%) | 100,002 ~ 456,255 | ID of loan in our sample | 조인·추적용 ID이며 모델 피처에서 제외 |
| `TARGET` | 타깃 | int64 | 0 (0.00%) | 0 ~ 1 | Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases) | 정답 레이블이며 입력 피처에서 제외 |
| `NAME_CONTRACT_TYPE` | 범주 | str | 0 (0.00%) | Cash loans(278,232), Revolving loans(29,279) | Identification if loan is cash or revolving | Stage 2에서 분포·결측 처리 기준 확정 |
| `CODE_GENDER` | 범주 | str | 0 (0.00%) | F(202,448), M(105,059), XNA(4) | Gender of the client | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_OWN_CAR` | 플래그 | str | 0 (0.00%) | N(202,924), Y(104,587) | Flag if the client owns a car | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_OWN_REALTY` | 플래그 | str | 0 (0.00%) | Y(213,312), N(94,199) | Flag if client owns a house or flat | Stage 2에서 분포·결측 처리 기준 확정 |
| `CNT_CHILDREN` | 개수 | int64 | 0 (0.00%) | 0 ~ 19 | Number of children the client has | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_INCOME_TOTAL` | 금액 | float64 | 0 (0.00%) | 25,650 ~ 117,000,000 | Income of the client | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_CREDIT` | 금액 | float64 | 0 (0.00%) | 45,000 ~ 4,050,000 | Credit amount of the loan | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_ANNUITY` | 금액 | float64 | 12 (0.00%) | 1,615.5 ~ 258,026 | Loan annuity | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_GOODS_PRICE` | 금액 | float64 | 278 (0.09%) | 40,500 ~ 4,050,000 | For consumer loans it is the price of the goods for which the loan is given | Stage 2에서 분포·결측 처리 기준 확정 |
| `NAME_TYPE_SUITE` | 범주 | str | 1,292 (0.42%) | Unaccompanied(248,526), Family(40,149), Spouse, partner(11,370), Children(3,267), Other_B(1,770), Other_A(866) 외 | Who was accompanying client when he was applying for the loan | Stage 2에서 분포·결측 처리 기준 확정 |
| `NAME_INCOME_TYPE` | 범주 | str | 0 (0.00%) | Working(158,774), Commercial associate(71,617), Pensioner(55,362), State servant(21,703), Unemployed(22), Student(18) 외 | Clients income type (businessman, working, maternity leave,…) | Stage 2에서 분포·결측 처리 기준 확정 |
| `NAME_EDUCATION_TYPE` | 범주 | str | 0 (0.00%) | Secondary / secondary special(218,391), Higher education(74,863), Incomplete higher(10,277), Lower secondary(3,816), Academic degree(164) | Level of highest education the client achieved | Stage 2에서 분포·결측 처리 기준 확정 |
| `NAME_FAMILY_STATUS` | 범주 | str | 0 (0.00%) | Married(196,432), Single / not married(45,444), Civil marriage(29,775), Separated(19,770), Widow(16,088), Unknown(2) | Family status of the client | Stage 2에서 분포·결측 처리 기준 확정 |
| `NAME_HOUSING_TYPE` | 범주 | str | 0 (0.00%) | House / apartment(272,868), With parents(14,840), Municipal apartment(11,183), Rented apartment(4,881), Office apartment(2,617), Co-op apartment(1,122) | What is the housing situation of the client (renting, living with parents, ...) | Stage 2에서 분포·결측 처리 기준 확정 |
| `REGION_POPULATION_RELATIVE` | 수치 | float64 | 0 (0.00%) | 0.00029 ~ 0.072508 | Normalized population of region where client lives (higher number means the client lives in more populated region) / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `DAYS_BIRTH` | 상대 일수 | int64 | 0 (0.00%) | -25,229 ~ -7,489 | Client's age in days at the time of application / Special: time only relative to the application | 신청일 기준 상대 일수와 부호 의미 유지 |
| `DAYS_EMPLOYED` | 상대 일수 | int64 | 0 (0.00%) | -17,912 ~ 365,243 | How many days before the application the person started current employment / Special: time only relative to the application | 365243 sentinel을 결측/별도 플래그로 검토 |
| `DAYS_REGISTRATION` | 상대 일수 | float64 | 0 (0.00%) | -24,672 ~ 0 | How many days before the application did client change his registration / Special: time only relative to the application | 신청일 기준 상대 일수와 부호 의미 유지 |
| `DAYS_ID_PUBLISH` | 상대 일수 | int64 | 0 (0.00%) | -7,197 ~ 0 | How many days before the application did client change the identity document with which he applied for the loan / Special: time only relative to the application | 신청일 기준 상대 일수와 부호 의미 유지 |
| `OWN_CAR_AGE` | 수치 | float64 | 202,929 (65.99%) | 0 ~ 91 | Age of client's car | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `FLAG_MOBIL` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide mobile phone (1=YES, 0=NO) | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_EMP_PHONE` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide work phone (1=YES, 0=NO) | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_WORK_PHONE` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide home phone (1=YES, 0=NO) | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_CONT_MOBILE` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Was mobile phone reachable (1=YES, 0=NO) | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_PHONE` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide home phone (1=YES, 0=NO) | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_EMAIL` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide email (1=YES, 0=NO) | Stage 2에서 분포·결측 처리 기준 확정 |
| `OCCUPATION_TYPE` | 범주 | str | 96,391 (31.35%) | Laborers(55,186), Sales staff(32,102), Core staff(27,570), Managers(21,371), Drivers(18,603), High skill tech staff(11,380) 외 | What kind of occupation does the client have | Stage 2에서 분포·결측 처리 기준 확정 |
| `CNT_FAM_MEMBERS` | 개수 | float64 | 2 (0.00%) | 1 ~ 20 | How many family members does client have | Stage 2에서 분포·결측 처리 기준 확정 |
| `REGION_RATING_CLIENT` | 수치 | int64 | 0 (0.00%) | 1 ~ 3 | Our rating of the region where client lives (1,2,3) | Stage 2에서 분포·결측 처리 기준 확정 |
| `REGION_RATING_CLIENT_W_CITY` | 수치 | int64 | 0 (0.00%) | 1 ~ 3 | Our rating of the region where client lives with taking city into account (1,2,3) | Stage 2에서 분포·결측 처리 기준 확정 |
| `WEEKDAY_APPR_PROCESS_START` | 범주 | str | 0 (0.00%) | TUESDAY(53,901), WEDNESDAY(51,934), MONDAY(50,714), THURSDAY(50,591), FRIDAY(50,338), SATURDAY(33,852) 외 | On which day of the week did the client apply for the loan | Stage 2에서 분포·결측 처리 기준 확정 |
| `HOUR_APPR_PROCESS_START` | 수치 | int64 | 0 (0.00%) | 0 ~ 23 | Approximately at what hour did the client apply for the loan / Special: rounded | Stage 2에서 분포·결측 처리 기준 확정 |
| `REG_REGION_NOT_LIVE_REGION` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Flag if client's permanent address does not match contact address (1=different, 0=same, at region level) | Stage 2에서 분포·결측 처리 기준 확정 |
| `REG_REGION_NOT_WORK_REGION` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Flag if client's permanent address does not match work address (1=different, 0=same, at region level) | Stage 2에서 분포·결측 처리 기준 확정 |
| `LIVE_REGION_NOT_WORK_REGION` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Flag if client's contact address does not match work address (1=different, 0=same, at region level) | Stage 2에서 분포·결측 처리 기준 확정 |
| `REG_CITY_NOT_LIVE_CITY` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Flag if client's permanent address does not match contact address (1=different, 0=same, at city level) | Stage 2에서 분포·결측 처리 기준 확정 |
| `REG_CITY_NOT_WORK_CITY` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Flag if client's permanent address does not match work address (1=different, 0=same, at city level) | Stage 2에서 분포·결측 처리 기준 확정 |
| `LIVE_CITY_NOT_WORK_CITY` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Flag if client's contact address does not match work address (1=different, 0=same, at city level) | Stage 2에서 분포·결측 처리 기준 확정 |
| `ORGANIZATION_TYPE` | 범주 | str | 0 (0.00%) | Business Entity Type 3(67,992), XNA(55,374), Self-employed(38,412), Other(16,683), Medicine(11,193), Business Entity Type 2(10,553) 외 | Type of organization where client works | Stage 2에서 분포·결측 처리 기준 확정 |
| `EXT_SOURCE_1` | 외부 신용점수 | float64 | 173,378 (56.38%) | 0.0145681 ~ 0.962693 | Normalized score from external data source / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `EXT_SOURCE_2` | 외부 신용점수 | float64 | 660 (0.21%) | 8.17362e-08 ~ 0.855 | Normalized score from external data source / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `EXT_SOURCE_3` | 외부 신용점수 | float64 | 60,965 (19.83%) | 0.000527265 ~ 0.89601 | Normalized score from external data source / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `APARTMENTS_AVG` | 수치 | float64 | 156,061 (50.75%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `BASEMENTAREA_AVG` | 수치 | float64 | 179,943 (58.52%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `YEARS_BEGINEXPLUATATION_AVG` | 수치 | float64 | 150,007 (48.78%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `YEARS_BUILD_AVG` | 수치 | float64 | 204,488 (66.50%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `COMMONAREA_AVG` | 수치 | float64 | 214,865 (69.87%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `ELEVATORS_AVG` | 수치 | float64 | 163,891 (53.30%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `ENTRANCES_AVG` | 수치 | float64 | 154,828 (50.35%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `FLOORSMAX_AVG` | 수치 | float64 | 153,020 (49.76%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLOORSMIN_AVG` | 수치 | float64 | 208,642 (67.85%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LANDAREA_AVG` | 수치 | float64 | 182,590 (59.38%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LIVINGAPARTMENTS_AVG` | 수치 | float64 | 210,199 (68.35%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LIVINGAREA_AVG` | 수치 | float64 | 154,350 (50.19%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `NONLIVINGAPARTMENTS_AVG` | 수치 | float64 | 213,514 (69.43%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `NONLIVINGAREA_AVG` | 수치 | float64 | 169,682 (55.18%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `APARTMENTS_MODE` | 수치 | float64 | 156,061 (50.75%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `BASEMENTAREA_MODE` | 수치 | float64 | 179,943 (58.52%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `YEARS_BEGINEXPLUATATION_MODE` | 수치 | float64 | 150,007 (48.78%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `YEARS_BUILD_MODE` | 수치 | float64 | 204,488 (66.50%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `COMMONAREA_MODE` | 수치 | float64 | 214,865 (69.87%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `ELEVATORS_MODE` | 수치 | float64 | 163,891 (53.30%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `ENTRANCES_MODE` | 수치 | float64 | 154,828 (50.35%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `FLOORSMAX_MODE` | 수치 | float64 | 153,020 (49.76%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLOORSMIN_MODE` | 수치 | float64 | 208,642 (67.85%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LANDAREA_MODE` | 수치 | float64 | 182,590 (59.38%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LIVINGAPARTMENTS_MODE` | 수치 | float64 | 210,199 (68.35%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LIVINGAREA_MODE` | 수치 | float64 | 154,350 (50.19%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `NONLIVINGAPARTMENTS_MODE` | 수치 | float64 | 213,514 (69.43%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `NONLIVINGAREA_MODE` | 수치 | float64 | 169,682 (55.18%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `APARTMENTS_MEDI` | 수치 | float64 | 156,061 (50.75%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `BASEMENTAREA_MEDI` | 수치 | float64 | 179,943 (58.52%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `YEARS_BEGINEXPLUATATION_MEDI` | 수치 | float64 | 150,007 (48.78%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `YEARS_BUILD_MEDI` | 수치 | float64 | 204,488 (66.50%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `COMMONAREA_MEDI` | 수치 | float64 | 214,865 (69.87%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `ELEVATORS_MEDI` | 수치 | float64 | 163,891 (53.30%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `ENTRANCES_MEDI` | 수치 | float64 | 154,828 (50.35%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `FLOORSMAX_MEDI` | 수치 | float64 | 153,020 (49.76%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLOORSMIN_MEDI` | 수치 | float64 | 208,642 (67.85%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LANDAREA_MEDI` | 수치 | float64 | 182,590 (59.38%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LIVINGAPARTMENTS_MEDI` | 수치 | float64 | 210,199 (68.35%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `LIVINGAREA_MEDI` | 수치 | float64 | 154,350 (50.19%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `NONLIVINGAPARTMENTS_MEDI` | 수치 | float64 | 213,514 (69.43%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `NONLIVINGAREA_MEDI` | 수치 | float64 | 169,682 (55.18%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `FONDKAPREMONT_MODE` | 범주 | str | 210,295 (68.39%) | reg oper account(73,830), reg oper spec account(12,080), not specified(5,687), org spec account(5,619) | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `HOUSETYPE_MODE` | 범주 | str | 154,297 (50.18%) | block of flats(150,503), specific housing(1,499), terraced house(1,212) | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `TOTALAREA_MODE` | 수치 | float64 | 148,431 (48.27%) | 0 ~ 1 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `WALLSMATERIAL_MODE` | 범주 | str | 156,341 (50.84%) | Panel(66,040), Stone, brick(64,815), Block(9,253), Wooden(5,362), Mixed(2,296), Monolithic(1,779) 외 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `EMERGENCYSTATE_MODE` | 범주 | str | 145,755 (47.40%) | No(159,428), Yes(2,328) | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor / Special: normalized | Stage 2에서 분포·결측 처리 기준 확정 |
| `OBS_30_CNT_SOCIAL_CIRCLE` | 수치 | float64 | 1,021 (0.33%) | 0 ~ 348 | How many observation of client's social surroundings with observable 30 DPD (days past due) default | Stage 2에서 분포·결측 처리 기준 확정 |
| `DEF_30_CNT_SOCIAL_CIRCLE` | 수치 | float64 | 1,021 (0.33%) | 0 ~ 34 | How many observation of client's social surroundings defaulted on 30 DPD (days past due) | Stage 2에서 분포·결측 처리 기준 확정 |
| `OBS_60_CNT_SOCIAL_CIRCLE` | 수치 | float64 | 1,021 (0.33%) | 0 ~ 344 | How many observation of client's social surroundings with observable 60 DPD (days past due) default | Stage 2에서 분포·결측 처리 기준 확정 |
| `DEF_60_CNT_SOCIAL_CIRCLE` | 수치 | float64 | 1,021 (0.33%) | 0 ~ 24 | How many observation of client's social surroundings defaulted on 60 (days past due) DPD | Stage 2에서 분포·결측 처리 기준 확정 |
| `DAYS_LAST_PHONE_CHANGE` | 상대 일수 | float64 | 1 (0.00%) | -4,292 ~ 0 | How many days before application did client change phone | 신청일 기준 상대 일수와 부호 의미 유지 |
| `FLAG_DOCUMENT_2` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 2 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_3` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 3 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_4` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 4 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_5` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 5 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_6` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 6 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_7` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 7 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_8` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 8 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_9` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 9 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_10` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 10 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_11` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 11 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_12` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 12 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_13` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 13 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_14` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 14 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_15` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 15 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_16` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 16 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_17` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 17 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_18` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 18 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_19` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 19 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_20` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 20 | Stage 2에서 분포·결측 처리 기준 확정 |
| `FLAG_DOCUMENT_21` | 플래그 | int64 | 0 (0.00%) | 0 ~ 1 | Did client provide document 21 | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | 조회 횟수 | float64 | 41,519 (13.50%) | 0 ~ 4 | Number of enquiries to Credit Bureau about the client one hour before application | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_REQ_CREDIT_BUREAU_DAY` | 조회 횟수 | float64 | 41,519 (13.50%) | 0 ~ 9 | Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application) | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | 조회 횟수 | float64 | 41,519 (13.50%) | 0 ~ 8 | Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application) | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_REQ_CREDIT_BUREAU_MON` | 조회 횟수 | float64 | 41,519 (13.50%) | 0 ~ 27 | Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application) | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_REQ_CREDIT_BUREAU_QRT` | 조회 횟수 | float64 | 41,519 (13.50%) | 0 ~ 261 | Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application) | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | 조회 횟수 | float64 | 41,519 (13.50%) | 0 ~ 25 | Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application) | Stage 2에서 분포·결측 처리 기준 확정 |

## `bureau.csv`

- 행 단위: 타 금융기관 신용거래 1건당 1행
- 크기: 1,716,428행 × 17열

| 컬럼 | 의미 유형 | 관측 dtype | 결측 | 관측 범위/주요 값 | 공식 설명(영문) | Stage 2 메모 |
|---|---|---|---:|---|---|---|
| `SK_ID_CURR` | 식별자 | int64 | 0 (0.00%) | 100,001 ~ 456,255 | ID of loan in our sample - one loan in our sample can have 0,1,2 or more related previous credits in credit bureau / Special: hashed | 조인·추적용 ID이며 모델 피처에서 제외 |
| `SK_ID_BUREAU` | 식별자 | int64 | 0 (0.00%) | 5,000,000 ~ 6,843,457 | Recoded ID of previous Credit Bureau credit related to our loan (unique coding for each loan application) / Special: hashed; 공식 설명 파일에는 SK_BUREAU_ID로 표기 | 조인·추적용 ID이며 모델 피처에서 제외 |
| `CREDIT_ACTIVE` | 범주 | str | 0 (0.00%) | Closed(1,079,273), Active(630,607), Sold(6,527), Bad debt(21) | Status of the Credit Bureau (CB) reported credits | Stage 2에서 분포·결측 처리 기준 확정 |
| `CREDIT_CURRENCY` | 범주 | str | 0 (0.00%) | currency 1(1,715,020), currency 2(1,224), currency 3(174), currency 4(10) | Recoded currency of the Credit Bureau credit / Special: recoded | Stage 2에서 분포·결측 처리 기준 확정 |
| `DAYS_CREDIT` | 상대 일수 | int64 | 0 (0.00%) | -2,922 ~ 0 | How many days before current application did client apply for Credit Bureau credit / Special: time only relative to the application | 신청일 기준 상대 일수와 부호 의미 유지 |
| `CREDIT_DAY_OVERDUE` | 수치 | int64 | 0 (0.00%) | 0 ~ 2,792 | Number of days past due on CB credit at the time of application for related loan in our sample | Stage 2에서 분포·결측 처리 기준 확정 |
| `DAYS_CREDIT_ENDDATE` | 상대 일수 | float64 | 105,553 (6.15%) | -42,060 ~ 31,199 | Remaining duration of CB credit (in days) at the time of application in Home Credit / Special: time only relative to the application | 신청일 기준 상대 일수와 부호 의미 유지 |
| `DAYS_ENDDATE_FACT` | 상대 일수 | float64 | 633,653 (36.92%) | -42,023 ~ 0 | Days since CB credit ended at the time of application in Home Credit (only for closed credit) / Special: time only relative to the application | 신청일 기준 상대 일수와 부호 의미 유지 |
| `AMT_CREDIT_MAX_OVERDUE` | 금액 | float64 | 1,124,488 (65.51%) | 0 ~ 115,987,185 | Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample) | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |
| `CNT_CREDIT_PROLONG` | 개수 | int64 | 0 (0.00%) | 0 ~ 9 | How many times was the Credit Bureau credit prolonged | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_CREDIT_SUM` | 금액 | float64 | 13 (0.00%) | 0 ~ 585,000,000 | Current credit amount for the Credit Bureau credit | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_CREDIT_SUM_DEBT` | 금액 | float64 | 257,669 (15.01%) | -4.7056e+06 ~ 170,100,000 | Current debt on Credit Bureau credit | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_CREDIT_SUM_LIMIT` | 금액 | float64 | 591,780 (34.48%) | -586,406 ~ 4.7056e+06 | Current credit limit of credit card reported in Credit Bureau | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_CREDIT_SUM_OVERDUE` | 금액 | float64 | 0 (0.00%) | 0 ~ 3,756,681 | Current amount overdue on Credit Bureau credit | Stage 2에서 분포·결측 처리 기준 확정 |
| `CREDIT_TYPE` | 범주 | str | 0 (0.00%) | Consumer credit(1,251,615), Credit card(402,195), Car loan(27,690), Mortgage(18,391), Microloan(12,413), Loan for business development(1,975) 외 | Type of Credit Bureau credit (Car, cash,...) | Stage 2에서 분포·결측 처리 기준 확정 |
| `DAYS_CREDIT_UPDATE` | 상대 일수 | int64 | 0 (0.00%) | -41,947 ~ 372 | How many days before loan application did last information about the Credit Bureau credit come / Special: time only relative to the application | 신청일 기준 상대 일수와 부호 의미 유지 |
| `AMT_ANNUITY` | 금액 | float64 | 1,226,791 (71.47%) | 0 ~ 1.18453e+08 | Annuity of the Credit Bureau credit | 고결측 변수: 의미와 활용 여부를 Stage 2에서 결정 |

## `installments_payments.csv`

- 행 단위: 예정 할부금에 대한 실제 납부행위 1건당 1행(분할납부 가능)
- 크기: 13,605,401행 × 8열

| 컬럼 | 의미 유형 | 관측 dtype | 결측 | 관측 범위/주요 값 | 공식 설명(영문) | Stage 2 메모 |
|---|---|---|---:|---|---|---|
| `SK_ID_PREV` | 식별자 | int64 | 0 (0.00%) | 1,000,001 ~ 2,843,499 | ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loans in Home Credit) / Special: hashed | 조인·추적용 ID이며 모델 피처에서 제외 |
| `SK_ID_CURR` | 식별자 | int64 | 0 (0.00%) | 100,001 ~ 456,255 | ID of loan in our sample / Special: hashed | 조인·추적용 ID이며 모델 피처에서 제외 |
| `NUM_INSTALMENT_VERSION` | 번호/수치 | float64 | 0 (0.00%) | 0 ~ 178 | Version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed | Stage 2에서 분포·결측 처리 기준 확정 |
| `NUM_INSTALMENT_NUMBER` | 번호/수치 | int64 | 0 (0.00%) | 1 ~ 277 | On which installment we observe payment | Stage 2에서 분포·결측 처리 기준 확정 |
| `DAYS_INSTALMENT` | 상대 일수 | float64 | 0 (0.00%) | -2,922 ~ -1 | When the installment of previous credit was supposed to be paid (relative to application date of current loan) / Special: time only relative to the application | 신청일 기준 상대 일수와 부호 의미 유지 |
| `DAYS_ENTRY_PAYMENT` | 상대 일수 | float64 | 2,905 (0.02%) | -4,921 ~ -1 | When was the installments of previous credit paid actually (relative to application date of current loan) / Special: time only relative to the application | 동시 결측은 미납/미기록 의미를 확인하고 단순 0 대치 금지 |
| `AMT_INSTALMENT` | 금액 | float64 | 0 (0.00%) | 0 ~ 3.77149e+06 | What was the prescribed installment amount of previous credit on this installment | Stage 2에서 분포·결측 처리 기준 확정 |
| `AMT_PAYMENT` | 금액 | float64 | 2,905 (0.02%) | 0 ~ 3.77149e+06 | What the client actually paid on previous credit on this installment | 동시 결측은 미납/미기록 의미를 확인하고 단순 0 대치 금지 |
