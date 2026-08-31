"""Stage 2 고객 단위 TARGET 층화 분할을 재현 가능하게 생성한다.

원본 ``application_train.csv``에서는 ``SK_ID_CURR``와 ``TARGET``만 읽는다.
고객 배정 파일은 로컬 ``data/interim``에 저장하고, Git으로 공유하는 요약에는
고객 식별자를 포함하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
SPLIT_COLUMN = "SPLIT"
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEFAULT_SEED = 42
DEFAULT_INPUT = Path("data/raw/application_train.csv")
DEFAULT_ASSIGNMENTS_OUTPUT = Path("data/interim/customer_splits.csv")
DEFAULT_SUMMARY_OUTPUT = Path("reports/stage2_split_summary.json")
SCHEMA_VERSION = "1.0"
ALGORITHM = "blake2b-stratified-apportionment-v1"


class DataSplitError(ValueError):
    """고객 분할 계약을 만족하지 못할 때 발생한다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_ratios(ratios: Mapping[str, float] | None) -> dict[str, float]:
    values = dict(DEFAULT_RATIOS if ratios is None else ratios)
    if set(values) != set(SPLIT_NAMES):
        raise DataSplitError(f"분할 비율의 키는 {SPLIT_NAMES}여야 합니다.")

    normalised: dict[str, float] = {}
    for split in SPLIT_NAMES:
        try:
            ratio = float(values[split])
        except (TypeError, ValueError) as error:
            raise DataSplitError(f"{split} 비율은 숫자여야 합니다.") from error
        if not math.isfinite(ratio) or ratio <= 0:
            raise DataSplitError(f"{split} 비율은 0보다 큰 유한한 값이어야 합니다.")
        normalised[split] = ratio

    if not math.isclose(sum(normalised.values()), 1.0, abs_tol=1e-12):
        raise DataSplitError("train/validation/test 비율의 합은 1이어야 합니다.")
    return normalised


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise DataSplitError("seed는 0 이상의 정수여야 합니다.")
    value = int(seed)
    if value < 0 or value > 2**64 - 1:
        raise DataSplitError("seed는 0 이상 2^64-1 이하의 정수여야 합니다.")
    return value


def load_application_targets(source_path: str | Path) -> pd.DataFrame:
    """원본 신청 파일에서 고객 ID와 TARGET만 읽고 계약을 검증한다."""

    path = Path(source_path)
    if not path.is_file():
        raise DataSplitError("application_train.csv를 찾을 수 없습니다.")

    try:
        raw = pd.read_csv(
            path,
            usecols=[ID_COLUMN, TARGET_COLUMN],
            dtype={ID_COLUMN: "string", TARGET_COLUMN: "string"},
        )
    except ValueError as error:
        raise DataSplitError(
            f"원본 CSV에는 {ID_COLUMN}와 {TARGET_COLUMN} 컬럼이 모두 필요합니다."
        ) from error

    if raw.empty:
        raise DataSplitError("분할할 고객 행이 없습니다.")

    id_missing = raw[ID_COLUMN].isna()
    if id_missing.any():
        raise DataSplitError(
            f"{ID_COLUMN} 결측값이 {int(id_missing.sum()):,}건 있습니다."
        )

    id_numeric = pd.to_numeric(raw[ID_COLUMN], errors="coerce")
    id_values = id_numeric.to_numpy(dtype="float64", na_value=np.nan)
    id_invalid = ~np.isfinite(id_values) | (np.floor(id_values) != id_values)
    id_invalid |= id_values <= 0
    id_invalid |= id_values > np.iinfo(np.int64).max
    if id_invalid.any():
        raise DataSplitError(
            f"{ID_COLUMN}에는 양의 정수만 허용됩니다: {int(id_invalid.sum()):,}건 위반"
        )

    customer_ids = pd.Series(id_values.astype("int64"), name=ID_COLUMN)
    duplicate_count = int(customer_ids.duplicated(keep=False).sum())
    if duplicate_count:
        raise DataSplitError(
            f"{ID_COLUMN} 중복 고객 행이 {duplicate_count:,}건 있습니다."
        )

    target_missing = raw[TARGET_COLUMN].isna()
    if target_missing.any():
        raise DataSplitError(
            f"{TARGET_COLUMN} 결측값이 {int(target_missing.sum()):,}건 있습니다."
        )

    target_numeric = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")
    target_invalid = target_numeric.isna() | ~target_numeric.isin([0, 1])
    if target_invalid.any():
        raise DataSplitError(
            f"{TARGET_COLUMN}에는 0과 1만 허용됩니다: "
            f"{int(target_invalid.sum()):,}건 위반"
        )

    targets = target_numeric.astype("int8").reset_index(drop=True)
    customers = pd.DataFrame({ID_COLUMN: customer_ids, TARGET_COLUMN: targets})
    if set(customers[TARGET_COLUMN].unique()) != {0, 1}:
        raise DataSplitError("TARGET 0과 1이 모두 존재해야 층화 분할할 수 있습니다.")
    return customers


def _apportion(total: int, weights: Sequence[float]) -> list[int]:
    """Hamilton 방식으로 합계가 정확히 ``total``인 정수 몫을 계산한다."""

    raw = np.asarray(weights, dtype="float64") * total
    counts = np.floor(raw).astype("int64")
    remainder = int(total - counts.sum())
    fractions = raw - counts
    order = sorted(range(len(weights)), key=lambda index: (-fractions[index], index))
    for index in order[:remainder]:
        counts[index] += 1
    return [int(value) for value in counts]


def _expected_count_matrix(
    customers: pd.DataFrame, ratios: Mapping[str, float]
) -> dict[int, dict[str, int]]:
    """전체 split 크기와 가장 가까운 이진 TARGET 층화 정수 배분을 만든다."""

    total = len(customers)
    split_sizes = _apportion(total, [ratios[name] for name in SPLIT_NAMES])
    if any(size == 0 for size in split_sizes):
        raise DataSplitError("모든 split에 고객을 배정하려면 더 많은 행이 필요합니다.")

    positive_total = int((customers[TARGET_COLUMN] == 1).sum())
    positive_weights = [size / total for size in split_sizes]
    positive_counts = _apportion(positive_total, positive_weights)
    negative_counts = [
        split_size - positive_count
        for split_size, positive_count in zip(
            split_sizes, positive_counts, strict=True
        )
    ]
    if any(count < 0 for count in negative_counts):
        raise DataSplitError("TARGET 층화 정수 배분을 계산할 수 없습니다.")

    return {
        0: dict(zip(SPLIT_NAMES, negative_counts, strict=True)),
        1: dict(zip(SPLIT_NAMES, positive_counts, strict=True)),
    }


def _ranking_digest(customer_id: int, seed: int) -> bytes:
    payload = f"{seed}:{customer_id}".encode("ascii")
    return hashlib.blake2b(
        payload,
        digest_size=16,
        person=b"CreditLens-v1",
    ).digest()


def create_customer_splits(
    customers: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """고객을 해시 순서로 정렬한 뒤 TARGET별 정수 몫에 맞춰 배정한다."""

    seed = _validate_seed(seed)
    ratio_map = _normalise_ratios(ratios)
    required = {ID_COLUMN, TARGET_COLUMN}
    if set(customers.columns) != required:
        raise DataSplitError(f"입력 DataFrame 컬럼은 {sorted(required)}여야 합니다.")
    if customers.empty:
        raise DataSplitError("분할할 고객 행이 없습니다.")
    if customers[ID_COLUMN].isna().any() or customers[TARGET_COLUMN].isna().any():
        raise DataSplitError("고객 ID와 TARGET에는 결측값이 없어야 합니다.")
    if customers[ID_COLUMN].duplicated().any():
        raise DataSplitError("고객 ID는 고객당 한 번만 나타나야 합니다.")
    if set(customers[TARGET_COLUMN].unique()) != {0, 1}:
        raise DataSplitError("TARGET은 0과 1을 모두 포함해야 합니다.")

    # 호출자가 필터링·셔플한 DataFrame의 중복 index를 전달해도 고객 ID만을
    # 분할 단위로 사용하도록 위치 index를 새로 만든다.
    working = customers[[ID_COLUMN, TARGET_COLUMN]].reset_index(drop=True).copy()
    expected = _expected_count_matrix(working, ratio_map)
    split_by_index: dict[int, str] = {}
    for target in (0, 1):
        group = working.loc[working[TARGET_COLUMN] == target, ID_COLUMN]
        ranked = sorted(
            ((int(index), int(customer_id)) for index, customer_id in group.items()),
            key=lambda item: (_ranking_digest(item[1], seed), item[1]),
        )
        cursor = 0
        for split in SPLIT_NAMES:
            count = expected[target][split]
            for index, _customer_id in ranked[cursor : cursor + count]:
                split_by_index[index] = split
            cursor += count
        if cursor != len(ranked):
            raise DataSplitError("TARGET별 고객 배정 수가 원본과 일치하지 않습니다.")

    assignments = working.copy()
    assignments[SPLIT_COLUMN] = assignments.index.map(split_by_index)
    if assignments[SPLIT_COLUMN].isna().any():
        raise DataSplitError("split이 배정되지 않은 고객이 있습니다.")
    assignments = assignments.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
    assignments[SPLIT_COLUMN] = assignments[SPLIT_COLUMN].astype("string")
    validate_split_assignments(working, assignments, ratios=ratio_map)
    return assignments[[ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN]]


def validate_split_assignments(
    customers: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, bool]:
    """분할의 배타성, 전체 보존, TARGET 보존과 층화 배분을 검증한다."""

    ratio_map = _normalise_ratios(ratios)
    required = {ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}
    if set(assignments.columns) != required:
        raise DataSplitError(f"배정 결과 컬럼은 {sorted(required)}여야 합니다.")

    mutually_exclusive = not assignments[ID_COLUMN].duplicated().any()
    valid_split_names = set(assignments[SPLIT_COLUMN].dropna().unique()) == set(
        SPLIT_NAMES
    )
    if not mutually_exclusive:
        raise DataSplitError("split 사이에 중복 고객이 있습니다.")
    if not valid_split_names:
        raise DataSplitError("train/validation/test 이외의 split 또는 빈 split이 있습니다.")

    source_view = customers[[ID_COLUMN, TARGET_COLUMN]].copy()
    assignment_view = assignments[[ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN]].copy()
    merged = source_view.merge(
        assignment_view,
        on=ID_COLUMN,
        how="outer",
        suffixes=("_SOURCE", "_ASSIGNED"),
        indicator=True,
        validate="one_to_one",
    )
    all_customers_assigned = bool((merged["_merge"] == "both").all())
    target_values_preserved = bool(
        all_customers_assigned
        and (
            merged[f"{TARGET_COLUMN}_SOURCE"]
            == merged[f"{TARGET_COLUMN}_ASSIGNED"]
        ).all()
    )
    if not all_customers_assigned:
        raise DataSplitError("분할 결과가 원본 고객 전체를 정확히 보존하지 않습니다.")
    if not target_values_preserved:
        raise DataSplitError("분할 결과에서 고객 TARGET 값이 변경되었습니다.")

    expected = _expected_count_matrix(customers, ratio_map)
    observed = (
        assignments.groupby([TARGET_COLUMN, SPLIT_COLUMN], observed=True)
        .size()
        .to_dict()
    )
    stratified_target_counts = all(
        int(observed.get((target, split), 0)) == expected[target][split]
        for target in (0, 1)
        for split in SPLIT_NAMES
    )
    if not stratified_target_counts:
        raise DataSplitError("각 split의 TARGET 층화 건수가 계약과 일치하지 않습니다.")

    return {
        "mutually_exclusive_customer_ids": mutually_exclusive,
        "all_customers_assigned_once": all_customers_assigned,
        "target_values_preserved": target_values_preserved,
        "valid_split_names": valid_split_names,
        "stratified_target_counts": stratified_target_counts,
    }


def assignment_sha256(assignments: pd.DataFrame) -> str:
    """ID 오름차순의 canonical CSV 표현으로 고객 배정 해시를 만든다."""

    canonical = assignments[[ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN]].sort_values(
        ID_COLUMN, kind="stable"
    )
    digest = hashlib.sha256()
    digest.update(f"{ID_COLUMN},{TARGET_COLUMN},{SPLIT_COLUMN}\n".encode("ascii"))
    for customer_id, target, split in canonical.itertuples(index=False, name=None):
        digest.update(f"{int(customer_id)},{int(target)},{split}\n".encode("ascii"))
    return digest.hexdigest()


def _safe_display_path(path: Path) -> str:
    """절대경로나 상위 디렉터리를 노출하지 않는 표시 경로를 반환한다."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return path.name
    if ".." in relative.parts:
        return path.name
    return relative.as_posix()


def build_split_summary(
    *,
    source_path: str | Path,
    source_sha256: str,
    assignments: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """고객 ID와 절대경로가 없는 공유용 집계 요약을 만든다."""

    ratio_map = _normalise_ratios(ratios)
    seed = _validate_seed(seed)
    total = len(assignments)
    overall_positive_rate = float(assignments[TARGET_COLUMN].mean())
    split_summary: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        frame = assignments.loc[assignments[SPLIT_COLUMN] == split]
        target_counts = frame[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
        positive_rate = float(target_counts.loc[1] / len(frame))
        split_summary[split] = {
            "rows": int(len(frame)),
            "row_ratio": float(len(frame) / total),
            "target_counts": {
                "0": int(target_counts.loc[0]),
                "1": int(target_counts.loc[1]),
            },
            "target_1_rate": positive_rate,
            "target_1_rate_difference_from_overall": float(
                positive_rate - overall_positive_rate
            ),
        }

    configuration = {
        "algorithm": ALGORITHM,
        "seed": seed,
        "ratios": ratio_map,
    }
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    invariants = {
        "mutually_exclusive_customer_ids": True,
        "all_customers_assigned_once": True,
        "target_values_preserved": True,
        "valid_split_names": True,
        "stratified_target_counts": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc
        or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source": {
            "display_path": _safe_display_path(Path(source_path)),
            "sha256": source_sha256,
            "rows": total,
        },
        "strategy": configuration,
        "overall_target_counts": {
            "0": int((assignments[TARGET_COLUMN] == 0).sum()),
            "1": int((assignments[TARGET_COLUMN] == 1).sum()),
        },
        "overall_target_1_rate": overall_positive_rate,
        "splits": split_summary,
        "assignment_sha256": assignment_sha256(assignments),
        "configuration_sha256": configuration_sha256,
        "invariants": invariants,
    }


def _check_output_paths(source_path: Path, outputs: Sequence[Path]) -> None:
    source_resolved = source_path.resolve()
    output_resolved = [path.resolve() for path in outputs]
    if source_resolved in output_resolved:
        raise DataSplitError("출력 파일은 원본 application CSV와 달라야 합니다.")
    if len(set(output_resolved)) != len(output_resolved):
        raise DataSplitError("고객 배정 CSV와 요약 JSON은 서로 다른 파일이어야 합니다.")


def split_application_data(
    source_path: str | Path = DEFAULT_INPUT,
    *,
    assignments_output: str | Path | None = DEFAULT_ASSIGNMENTS_OUTPUT,
    summary_output: str | Path | None = DEFAULT_SUMMARY_OUTPUT,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """원본을 변경하지 않고 고객 배정 CSV와 비식별 집계 JSON을 생성한다."""

    source = Path(source_path)
    ratio_map = _normalise_ratios(ratios)
    seed = _validate_seed(seed)
    output_paths = [
        Path(path)
        for path in (assignments_output, summary_output)
        if path is not None
    ]
    _check_output_paths(source, output_paths)

    source_hash_before = _sha256(source) if source.is_file() else ""
    customers = load_application_targets(source)
    assignments = create_customer_splits(customers, seed=seed, ratios=ratio_map)
    invariants = validate_split_assignments(customers, assignments, ratios=ratio_map)
    source_hash_after = _sha256(source)
    if source_hash_before != source_hash_after:
        raise DataSplitError("분할 실행 중 원본 CSV가 변경되어 출력을 중단했습니다.")

    summary = build_split_summary(
        source_path=source,
        source_sha256=source_hash_after,
        assignments=assignments,
        seed=seed,
        ratios=ratio_map,
    )
    summary["invariants"] = invariants

    if assignments_output is not None:
        assignment_path = Path(assignments_output)
        assignment_path.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(assignment_path, index=False, lineterminator="\n")
        if _sha256(assignment_path) != summary["assignment_sha256"]:
            raise DataSplitError("저장된 고객 배정 CSV의 해시가 canonical 해시와 다릅니다.")

    if summary_output is not None:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return assignments, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="application_train 고객을 70/15/15 TARGET 층화 분할합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--assignments-output", type=Path, default=DEFAULT_ASSIGNMENTS_OUTPUT
    )
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument(
        "--validation-ratio", type=float, default=DEFAULT_RATIOS["validation"]
    )
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    _assignments, summary = split_application_data(
        args.input,
        assignments_output=args.assignments_output,
        summary_output=args.summary_output,
        seed=args.seed,
        ratios=ratios,
    )
    print(
        "Stage 2 분할 완료: "
        + ", ".join(
            f"{split}={summary['splits'][split]['rows']:,}" for split in SPLIT_NAMES
        )
    )
    print(f"배정 SHA256: {summary['assignment_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
