"""프로젝트 루트에서 실행하는 계약 조회 / 합성 예측 스모크 테스트."""

import argparse
import json

from creditlens.inference.contract import InputContract
from creditlens.inference.demo import synthetic_request
from creditlens.inference.predictor import PredictionService
from creditlens.modeling.frozen_model import verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--contract", action="store_true", help="198개 입력 피처 계약 출력")
    group.add_argument("--example", action="store_true", help="실고객이 아닌 합성 요청 출력")
    group.add_argument("--demo", action="store_true", help="동결 모델로 합성 요청 예측")
    args = parser.parse_args()
    if args.demo:
        service = PredictionService.from_frozen()
        result = service.predict(synthetic_request(service.contract))
    else:
        contract = InputContract(verify_bundle())
        result = contract.describe() if args.contract else synthetic_request(contract)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
