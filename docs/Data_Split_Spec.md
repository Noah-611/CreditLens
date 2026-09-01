# CreditLens 데이터 분할 명세

> Stage 2에서 모델 개발용 고객을 학습·검증·테스트 세트로 나누는 고정 규칙입니다.

## 목적

모델이 최종 테스트 데이터를 미리 보고 조정되는 데이터 누수를 막고, 모든 후속 Stage에서 같은 고객 분할을 재현합니다. 원본 `application_train.csv`는 수정하지 않습니다.

## 고정 설정

| 항목 | 값 |
|---|---|
| 분할 단위 | 고객 1명 (`SK_ID_CURR`) |
| 층화 기준 | `TARGET` |
| 학습 비율 | 70% |
| 검증 비율 | 15% |
| 테스트 비율 | 15% |
| 고정 seed | `42` |
| 알고리즘 | `blake2b-stratified-apportionment-v1` |
| 원본 SHA-256 | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |

고객 ID와 seed로 만든 안정적인 해시 순서를 `TARGET=0`, `TARGET=1` 안에서 각각 적용합니다. 따라서 입력 CSV의 행 순서가 바뀌어도 동일한 고객은 동일한 split에 배정됩니다.

## 실제 분할 결과

| split | 고객 수 | 전체 비율 | TARGET=0 | TARGET=1 | TARGET=1 비율 |
|---|---:|---:|---:|---:|---:|
| train | 215,258 | 70.0001% | 197,881 | 17,377 | 8.0726% |
| validation | 46,127 | 15.0001% | 42,403 | 3,724 | 8.0734% |
| test | 46,126 | 14.9998% | 42,402 | 3,724 | 8.0735% |
| 전체 | 307,511 | 100.0000% | 282,686 | 24,825 | 8.0729% |

- 고객 배정 SHA-256: `bb61706ea286086a726497ade1fa0aa3f1aad60ff505461d0612aca6222382af`
- 고객 중복: 0명
- 미배정 고객: 0명
- `TARGET` 변경: 0건
- 허용되지 않은 split 이름: 0건

정수 고객을 배정하므로 명목 비율과 실제 비율에는 1명 미만의 반올림 차이가 있습니다.

## 저장 위치와 Git 정책

- 고객별 배정표: `data/interim/customer_splits.csv`
- 공유 가능한 집계 요약: `reports/stage2_split_summary.json`
- 고객별 배정표 컬럼: `SK_ID_CURR`, `TARGET`, `SPLIT`

고객별 배정표에는 식별자가 있으므로 `.gitignore`가 적용되는 로컬 파일로만 유지합니다. Git에는 고객 ID가 없는 집계 요약과 이 명세만 기록합니다.

## 테스트 세트 봉인 규칙

Stage 8 최종 평가 전까지 테스트 세트에서는 다음 작업을 하지 않습니다.

- 피처 분포·결측률·상관관계 EDA
- 결측 대치값, 희소범주 기준, 이상치 경계 계산
- 피처 선택, 모델 선택, 하이퍼파라미터 조정
- 임계값 선택과 성능 평가

분할이 정상인지 확인하기 위한 고객 수, 배타성, `TARGET` 보존과 층화 비율만 기록합니다. Stage 2 EDA는 `SPLIT=train` 고객만 사용합니다.

## 후속 Stage 적용

- Stage 3의 `bureau`와 `installments_payments`도 `SK_ID_CURR` 기준으로 이 배정을 그대로 상속합니다.
- 고객 이력은 고객 단위로 집계하며 서로 다른 split의 통계를 섞지 않습니다.
- 학습이 필요한 대치·인코딩·스케일링 객체는 train에만 `fit`합니다.
- validation에는 학습된 변환을 적용해 모델과 정책을 비교합니다.
- test는 Stage 8에서 모델·전처리·cutoff를 고정한 뒤 한 번 평가합니다.

## 재실행 명령

```bash
PYTHONPATH=src .venv/bin/python -m creditlens.data.split_data
```

재실행 후 `reports/stage2_split_summary.json`의 모든 `invariants`가 `true`이고 고객 배정 SHA-256이 이 문서와 같은지 확인합니다.
