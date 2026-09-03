"""Stage 5 2/3 V3 TensorFlow MLP를 누수 없이 학습하고 비교한다.

공식 train 내부에서 early stopping용 split을 만들고 best epoch를 정한다. 그 뒤
새 전처리기와 새 MLP를 공식 train 전체로 다시 학습하며, 공식 validation은
마지막 예측과 공통 지표 계산에만 사용한다. test는 계속 봉인한다.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


# TensorFlow import 전에 CPU 결정론과 로그 환경을 고정한다.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/creditlens-matplotlib")

import joblib
import keras
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit

from creditlens.evaluation import evaluate_binary_metrics
from creditlens.modeling.data import DevelopmentDataset, load_development_data
from creditlens.modeling.preprocessing import (
    make_preprocessor,
    transformed_feature_names,
)
from creditlens.modeling.train_lightgbm import (
    _atomic_write_json,
    _atomic_write_text,
    _display_path,
    _finite_number,
    _git_ignored,
    _load_baseline_reference,
    _peak_rss_mb,
    _sha256,
)


SCHEMA_VERSION = "1.0"
RUN_VERSION = "stage5-mlp-baseline-v1"
STAGE_PART = "2/3"
RANDOM_SEED = 42
INNER_HOLDOUT_FRACTION = 0.10
CLASSIFICATION_THRESHOLD = 0.5
TOP_FRACTION = 0.1
DEFAULT_OUTPUT = Path("reports/stage5_mlp_results.json")
DEFAULT_REPORT = Path("docs/Stage5_MLP_Report.md")
DEFAULT_FIGURE = Path("reports/figures/stage5_mlp_training_history.png")
DEFAULT_ARTIFACT_DIR = Path("models/stage5")
DEFAULT_STAGE4_REFERENCE = Path("reports/stage4_baseline_results.json")
DEFAULT_LIGHTGBM_REFERENCE = Path("reports/stage5_lightgbm_results.json")
MAX_DENSE_MATRIX_MIB = 1024.0


class MLPTrainingError(RuntimeError):
    """Stage 5 MLP 실험 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class InnerSplit:
    fit_positions: np.ndarray
    early_stop_positions: np.ndarray
    audit: dict[str, Any]


_TF_THREAD_CONFIG: tuple[int, int] | None = None


def _mlp_settings() -> dict[str, Any]:
    """공식 validation 확인 전에 고정한 MLP 기준 설정."""

    return {
        "hidden_units": [128, 64],
        "activation": "relu",
        "dropout_rates": [0.20, 0.10],
        "l2_regularization": 0.00001,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "clipnorm": 1.0,
        "loss": "binary_crossentropy",
        "batch_size": 1024,
        "predict_batch_size": 4096,
        "max_epochs": 50,
        "early_stopping_monitor": "val_pr_auc_approx",
        "early_stopping_mode": "max",
        "early_stopping_patience": 6,
        "early_stopping_min_delta": 0.0,
        "early_stopping_restore_best_weights": True,
        "metric_thresholds": 1000,
        "class_weight": "balanced_from_current_fit_labels",
        "shuffle": True,
    }


def _validate_shared_paths(
    *,
    output: Path,
    report: Path,
    figure: Path,
    artifacts: Path,
    stage4_reference: Path,
    lightgbm_reference: Path,
) -> None:
    if output.suffix.lower() != ".json":
        raise MLPTrainingError("집계 결과 경로는 .json이어야 합니다.")
    if report.suffix.lower() != ".md":
        raise MLPTrainingError("보고서 경로는 .md여야 합니다.")
    if figure.suffix.lower() != ".png":
        raise MLPTrainingError("학습 이력 그림 경로는 .png여야 합니다.")
    paths = (
        output.resolve(),
        report.resolve(),
        figure.resolve(),
        artifacts.resolve(),
        stage4_reference.resolve(),
        lightgbm_reference.resolve(),
    )
    if len(set(paths)) != len(paths):
        raise MLPTrainingError("공유 결과·모델·기준 파일 경로는 서로 달라야 합니다.")
    for shared in (output.resolve(), report.resolve(), figure.resolve()):
        try:
            shared.relative_to(artifacts.resolve())
        except ValueError:
            continue
        raise MLPTrainingError("공유 산출물은 Git 제외 모델 경로 아래에 둘 수 없습니다.")


def _artifact_ignore_status(path: Path) -> bool | None:
    try:
        ignored = _git_ignored(path)
    except RuntimeError as error:
        raise MLPTrainingError(str(error)) from error
    if ignored is False:
        raise MLPTrainingError(
            f"모델·행별 점수 경로는 Git에서 제외되어야 합니다: {_display_path(path)}"
        )
    return ignored


def _artifact_metadata(path: Path) -> dict[str, Any]:
    ignored = _artifact_ignore_status(path)
    if not path.is_file():
        raise MLPTrainingError(f"저장된 로컬 산출물을 찾을 수 없습니다: {path}")
    return {
        "display_path": _display_path(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "git_ignored": ignored,
    }


def _atomic_joblib_dump(value: Any, path: Path) -> dict[str, Any]:
    _artifact_ignore_status(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".joblib.tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        joblib.dump(value, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return _artifact_metadata(path)


def _atomic_keras_save(model: keras.Model, path: Path) -> dict[str, Any]:
    _artifact_ignore_status(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".keras", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    temporary.unlink(missing_ok=True)
    try:
        model.save(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return _artifact_metadata(path)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise MLPTrainingError(f"{label} 파일을 찾을 수 없습니다: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise MLPTrainingError(f"{label} JSON을 읽을 수 없습니다.") from error
    if not isinstance(payload, dict) or payload.get("run_status") != "complete":
        raise MLPTrainingError(f"완료된 {label} 결과가 필요합니다.")
    return payload


def _load_references(
    stage4_path: Path,
    lightgbm_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        stage4 = _load_baseline_reference(stage4_path)
    except RuntimeError as error:
        raise MLPTrainingError(str(error)) from error
    lightgbm = _load_json(lightgbm_path, label="Stage 5 LightGBM")
    if lightgbm.get("stage_part") != "1/3":
        raise MLPTrainingError("Stage 5 LightGBM 단계 표기가 1/3이 아닙니다.")
    scope = lightgbm.get("data_scope")
    if not isinstance(scope, dict) or (
        scope.get("test_feature_rows_used") != 0
        or scope.get("test_predictions_created") is not False
        or scope.get("customer_ids_in_shared_outputs") is not False
        or scope.get("row_level_predictions_in_shared_outputs") is not False
    ):
        raise MLPTrainingError("LightGBM 결과의 test·공유 출력 계약이 다릅니다.")
    settings = lightgbm.get("settings")
    if not isinstance(settings, dict):
        raise MLPTrainingError("LightGBM 평가 설정이 없습니다.")
    for key, expected in (
        ("classification_threshold", CLASSIFICATION_THRESHOLD),
        ("top_fraction", TOP_FRACTION),
    ):
        value = settings.get(key)
        if not _finite_number(value) or not np.isclose(
            float(value), expected, rtol=0.0, atol=1e-15
        ):
            raise MLPTrainingError(f"LightGBM {key} 설정이 현재 평가와 다릅니다.")

    versions = lightgbm.get("data_versions")
    if not isinstance(versions, dict) or not isinstance(versions.get("v3"), dict):
        raise MLPTrainingError("LightGBM V3 데이터 계약이 없습니다.")
    stage4_v3 = stage4["data_versions"]["v3"]
    lightgbm_v3 = versions["v3"]
    for key in (
        "schema_sha256",
        "parquet_sha256",
        "model_feature_columns",
        "numeric_feature_columns",
        "categorical_feature_columns",
        "train_rows",
        "validation_rows",
    ):
        if stage4_v3.get(key) != lightgbm_v3.get(key):
            raise MLPTrainingError(f"Stage 4와 LightGBM의 V3 {key}가 다릅니다.")
    stage4_reference = lightgbm.get("baseline_reference")
    if not isinstance(stage4_reference, dict) or stage4_reference.get(
        "sha256"
    ) != _sha256(stage4_path):
        raise MLPTrainingError("LightGBM이 참조한 Stage 4 결과가 현재 파일과 다릅니다.")

    lightgbm_experiments = lightgbm.get("experiments")
    if not isinstance(lightgbm_experiments, list):
        raise MLPTrainingError("LightGBM 실험 결과가 없습니다.")
    by_key = {
        item.get("key"): item
        for item in lightgbm_experiments
        if isinstance(item, dict)
    }
    candidate = by_key.get("lightgbm_v3")
    if not isinstance(candidate, dict) or candidate.get("data_version") != "v3":
        raise MLPTrainingError("V3 LightGBM 비교 후보가 없습니다.")
    _validate_reference_metrics(candidate.get("metrics"), label="V3 LightGBM")
    return stage4, lightgbm


def _validate_reference_metrics(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise MLPTrainingError(f"{label} 평가 지표가 없습니다.")
    for metric in ("roc_auc", "pr_auc", "ks", "gini", "brier_score"):
        if not _finite_number(value.get(metric)):
            raise MLPTrainingError(f"{label} {metric}이 유한한 숫자가 아닙니다.")
    top_k = value.get("top_k_metrics")
    if not isinstance(top_k, dict):
        raise MLPTrainingError(f"{label} Top-K 지표가 없습니다.")
    for metric in ("recall", "lift"):
        if not _finite_number(top_k.get(metric)):
            raise MLPTrainingError(f"{label} Top-K {metric}이 없습니다.")


def _reference_candidates(
    stage4: dict[str, Any],
    lightgbm: dict[str, Any],
) -> list[dict[str, Any]]:
    stage4_by_key = {item["key"]: item for item in stage4["experiments"]}
    lightgbm_by_key = {item["key"]: item for item in lightgbm["experiments"]}
    selected = (
        stage4_by_key["logistic_v3"],
        stage4_by_key["random_forest_v3"],
        lightgbm_by_key["lightgbm_v3"],
    )
    return [
        {
            "key": item["key"],
            "display_name": item["display_name"],
            "data_version": item["data_version"],
            "metrics": item["metrics"],
        }
        for item in selected
    ]


def _validate_dataset(
    dataset: DevelopmentDataset,
    stage4: dict[str, Any],
    lightgbm: dict[str, Any],
) -> None:
    if dataset.version != "v3":
        raise MLPTrainingError("TensorFlow MLP는 V3 데이터만 사용해야 합니다.")
    audit = dataset.audit
    if (
        audit.get("test_sealed") is not True
        or audit.get("test_feature_rows_used") != 0
        or audit.get("feature_rows_loaded", {}).get("test") != 0
    ):
        raise MLPTrainingError("MLP 실행에서 test 봉인 상태가 확인되지 않았습니다.")
    if hasattr(dataset, "test"):
        raise MLPTrainingError("개발 데이터 객체에 test split이 노출되었습니다.")

    expected = stage4["data_versions"]["v3"]
    observed = {
        "schema_sha256": dataset.schema_sha256,
        "parquet_sha256": dataset.parquet_sha256,
        "model_feature_columns": len(dataset.roles.model_features),
        "numeric_feature_columns": len(dataset.roles.numeric),
        "categorical_feature_columns": len(dataset.roles.categorical),
        "train_rows": len(dataset.train.y),
        "validation_rows": len(dataset.validation.y),
    }
    for key, value in observed.items():
        if expected.get(key) != value:
            raise MLPTrainingError(f"현재 V3 {key}가 Stage 4 기준과 다릅니다.")
        if lightgbm["data_versions"]["v3"].get(key) != value:
            raise MLPTrainingError(f"현재 V3 {key}가 LightGBM 기준과 다릅니다.")
    expected_rate = stage4["data_scope"].get("validation_positive_rate")
    if not _finite_number(expected_rate) or not np.isclose(
        float(dataset.validation.y.mean()),
        float(expected_rate),
        rtol=0.0,
        atol=1e-15,
    ):
        raise MLPTrainingError("현재 validation 양성률이 Stage 4 기준과 다릅니다.")


def _class_distribution(y: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise MLPTrainingError("학습 TARGET은 0/1의 1차원 배열이어야 합니다.")
    positive = int(np.count_nonzero(labels == 1))
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        raise MLPTrainingError("class weight 계산에는 두 TARGET 클래스가 필요합니다.")
    return {
        "rows": int(labels.size),
        "negative_count": negative,
        "positive_count": positive,
        "positive_rate": positive / labels.size,
    }


def _balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    distribution = _class_distribution(y)
    rows = distribution["rows"]
    return {
        0: rows / (2.0 * distribution["negative_count"]),
        1: rows / (2.0 * distribution["positive_count"]),
    }


def _weights_for_json(weights: dict[int, float]) -> dict[str, float]:
    return {str(key): float(value) for key, value in sorted(weights.items())}


def _positions_sha256(values: np.ndarray) -> str:
    positions = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(positions.tobytes(order="C")).hexdigest()


def _make_inner_split(y: np.ndarray) -> InnerSplit:
    labels = np.asarray(y, dtype=np.int8)
    _class_distribution(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=INNER_HOLDOUT_FRACTION,
        random_state=RANDOM_SEED,
    )
    fit_positions, stop_positions = next(
        splitter.split(np.zeros(labels.size, dtype=np.int8), labels)
    )
    fit_positions = np.sort(fit_positions.astype(np.int64, copy=False))
    stop_positions = np.sort(stop_positions.astype(np.int64, copy=False))
    if np.intersect1d(fit_positions, stop_positions).size:
        raise MLPTrainingError("inner-fit과 early-stop 고객이 겹칩니다.")
    combined = np.sort(np.concatenate([fit_positions, stop_positions]))
    if not np.array_equal(combined, np.arange(labels.size, dtype=np.int64)):
        raise MLPTrainingError("inner split이 공식 train 전체를 정확히 나누지 못했습니다.")
    fit_distribution = _class_distribution(labels[fit_positions])
    stop_distribution = _class_distribution(labels[stop_positions])
    if not np.isclose(
        fit_distribution["positive_rate"],
        stop_distribution["positive_rate"],
        rtol=0.0,
        atol=1.0 / min(len(fit_positions), len(stop_positions)),
    ):
        raise MLPTrainingError("inner split의 TARGET 층화가 유지되지 않았습니다.")
    return InnerSplit(
        fit_positions=fit_positions,
        early_stop_positions=stop_positions,
        audit={
            "method": "StratifiedShuffleSplit",
            "random_seed": RANDOM_SEED,
            "early_stop_fraction": INNER_HOLDOUT_FRACTION,
            "fit": fit_distribution,
            "early_stop": stop_distribution,
            "disjoint": True,
            "covers_full_train": True,
            "customer_ids_in_shared_output": False,
            "fit_positions_sha256": _positions_sha256(fit_positions),
            "early_stop_positions_sha256": _positions_sha256(stop_positions),
        },
    )


def _dense_float32(values: Any, *, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    if not hasattr(values, "shape") or len(values.shape) != 2:
        raise MLPTrainingError(f"{label} 전처리 결과는 2차원이어야 합니다.")
    expected_bytes = int(values.shape[0]) * int(values.shape[1]) * 4
    if expected_bytes > MAX_DENSE_MATRIX_MIB * 1024 * 1024:
        raise MLPTrainingError(
            f"{label} dense float32 예상 크기가 메모리 제한을 넘습니다."
        )
    source_sparse = sparse.issparse(values)
    source_nnz: int | None = None
    source_bytes: int | None = None
    if source_sparse:
        csr = values.tocsr().astype(np.float32, copy=False)
        source_nnz = int(csr.nnz)
        source_bytes = int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
        dense = csr.toarray(order="C")
    else:
        dense = np.asarray(values, dtype=np.float32)
    dense = np.ascontiguousarray(dense, dtype=np.float32)
    if not np.isfinite(dense).all():
        raise MLPTrainingError(f"{label} MLP 입력에 결측값이나 무한값이 있습니다.")
    rows, columns = dense.shape
    denominator = rows * columns
    return dense, {
        "rows": int(rows),
        "columns": int(columns),
        "dtype": str(dense.dtype),
        "c_contiguous": bool(dense.flags.c_contiguous),
        "dense_bytes": int(dense.nbytes),
        "dense_mib": dense.nbytes / (1024 * 1024),
        "source_sparse": source_sparse,
        "source_nnz": source_nnz,
        "source_density": None
        if source_nnz is None or denominator == 0
        else source_nnz / denominator,
        "source_sparse_bytes": source_bytes,
        "full_dense_materialization": True,
    }


def _configure_tensorflow(*, intra_threads: int, inter_threads: int) -> None:
    global _TF_THREAD_CONFIG
    requested = (intra_threads, inter_threads)
    if intra_threads < 1 or intra_threads > 4 or inter_threads != 1:
        raise MLPTrainingError(
            "TensorFlow CPU thread는 intra 1~4, inter 1로 제한합니다."
        )
    if _TF_THREAD_CONFIG is not None:
        if _TF_THREAD_CONFIG != requested:
            raise MLPTrainingError("한 프로세스에서 TensorFlow thread 설정을 바꿀 수 없습니다.")
        return
    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except RuntimeError as error:
        raise MLPTrainingError(
            "TensorFlow 연산 시작 전에 CPU thread 설정을 고정해야 합니다."
        ) from error
    tf.config.experimental.enable_op_determinism()
    if tf.config.get_visible_devices("GPU"):
        raise MLPTrainingError("MLP 기준 실행은 GPU를 비활성화한 CPU 계약이어야 합니다.")
    _TF_THREAD_CONFIG = requested


def _reset_tensorflow(
    *,
    intra_threads: int,
    inter_threads: int,
) -> None:
    _configure_tensorflow(
        intra_threads=intra_threads,
        inter_threads=inter_threads,
    )
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)


def _build_mlp(input_dim: int, settings: dict[str, Any]) -> keras.Model:
    if input_dim < 1:
        raise MLPTrainingError("MLP 입력 피처 수는 1개 이상이어야 합니다.")
    hidden_units = settings["hidden_units"]
    dropout_rates = settings["dropout_rates"]
    if len(hidden_units) != 2 or len(dropout_rates) != 2:
        raise MLPTrainingError("현재 MLP 기준 구조는 은닉층과 Dropout이 각각 2개입니다.")
    regularizer = keras.regularizers.l2(settings["l2_regularization"])
    inputs = keras.Input(shape=(input_dim,), dtype="float32", name="features")
    values = keras.layers.Dense(
        hidden_units[0],
        activation=settings["activation"],
        kernel_regularizer=regularizer,
        name="dense_128",
    )(inputs)
    values = keras.layers.Dropout(
        dropout_rates[0], seed=RANDOM_SEED + 1, name="dropout_20pct"
    )(values)
    values = keras.layers.Dense(
        hidden_units[1],
        activation=settings["activation"],
        kernel_regularizer=regularizer,
        name="dense_64",
    )(values)
    values = keras.layers.Dropout(
        dropout_rates[1], seed=RANDOM_SEED + 2, name="dropout_10pct"
    )(values)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="risk_score")(values)
    model = keras.Model(inputs=inputs, outputs=outputs, name="creditlens_mlp_v3")
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=settings["learning_rate"],
            clipnorm=settings["clipnorm"],
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(
                curve="ROC",
                num_thresholds=settings["metric_thresholds"],
                name="roc_auc_approx",
            ),
            keras.metrics.AUC(
                curve="PR",
                num_thresholds=settings["metric_thresholds"],
                name="pr_auc_approx",
            ),
        ],
    )
    return model


def _history_for_json(history: dict[str, list[Any]]) -> dict[str, list[float]]:
    expected = (
        "loss",
        "roc_auc_approx",
        "pr_auc_approx",
        "val_loss",
        "val_roc_auc_approx",
        "val_pr_auc_approx",
    )
    result: dict[str, list[float]] = {}
    for key in expected:
        values = history.get(key)
        if not isinstance(values, list) or not values:
            raise MLPTrainingError(f"TensorFlow 학습 이력에 {key}가 없습니다.")
        converted = [float(value) for value in values]
        if not all(math.isfinite(value) for value in converted):
            raise MLPTrainingError(f"TensorFlow 학습 이력 {key}에 비유한값이 있습니다.")
        result[key] = converted
    lengths = {len(values) for values in result.values()}
    if len(lengths) != 1:
        raise MLPTrainingError("TensorFlow 학습 이력의 epoch 길이가 다릅니다.")
    return result


def _feature_names_sha256(names: Sequence[str]) -> str:
    payload = "\n".join(str(name) for name in names)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_discovery(
    dataset: DevelopmentDataset,
    split: InnerSplit,
    settings: dict[str, Any],
    *,
    intra_threads: int,
    inter_threads: int,
    fit_verbose: int,
) -> dict[str, Any]:
    y_all = dataset.train.y.to_numpy(dtype=np.int8, copy=False)
    fit_y = y_all[split.fit_positions]
    stop_y = y_all[split.early_stop_positions]
    class_weights = _balanced_class_weights(fit_y)
    preprocessor = make_preprocessor(dataset.roles, model_family="linear")

    preprocess_started = time.perf_counter()
    fit_frame = dataset.train.X.iloc[split.fit_positions]
    preprocessor.fit(fit_frame)
    fit_values = preprocessor.transform(fit_frame)
    fit_dense, fit_matrix_audit = _dense_float32(
        fit_values, label="inner-fit"
    )
    del fit_values, fit_frame
    stop_frame = dataset.train.X.iloc[split.early_stop_positions]
    stop_values = preprocessor.transform(stop_frame)
    stop_dense, stop_matrix_audit = _dense_float32(
        stop_values, label="inner-early-stop"
    )
    del stop_values, stop_frame
    preprocess_seconds = time.perf_counter() - preprocess_started
    feature_names = transformed_feature_names(preprocessor)
    if fit_dense.shape[1] != stop_dense.shape[1] or fit_dense.shape[1] != len(
        feature_names
    ):
        raise MLPTrainingError("inner MLP 전처리 출력 피처 수가 일치하지 않습니다.")

    _reset_tensorflow(
        intra_threads=intra_threads,
        inter_threads=inter_threads,
    )
    model = _build_mlp(fit_dense.shape[1], settings)
    callback = keras.callbacks.EarlyStopping(
        monitor=settings["early_stopping_monitor"],
        mode=settings["early_stopping_mode"],
        patience=settings["early_stopping_patience"],
        min_delta=settings["early_stopping_min_delta"],
        restore_best_weights=settings["early_stopping_restore_best_weights"],
        start_from_epoch=0,
        verbose=fit_verbose,
    )
    train_started = time.perf_counter()
    history_object = model.fit(
        fit_dense,
        fit_y,
        validation_data=(stop_dense, stop_y),
        epochs=settings["max_epochs"],
        batch_size=settings["batch_size"],
        class_weight=class_weights,
        shuffle=settings["shuffle"],
        callbacks=[callback],
        verbose=fit_verbose,
    )
    train_seconds = time.perf_counter() - train_started
    history = _history_for_json(history_object.history)
    monitored = np.asarray(history[settings["early_stopping_monitor"]])
    if settings["early_stopping_mode"] == "max":
        best_epoch = int(np.argmax(monitored)) + 1
        best_value = float(np.max(monitored))
    else:
        best_epoch = int(np.argmin(monitored)) + 1
        best_value = float(np.min(monitored))
    epochs_ran = len(monitored)
    if best_epoch < 1 or best_epoch > epochs_ran:
        raise MLPTrainingError("early stopping best epoch가 학습 이력 범위를 벗어납니다.")

    result = {
        "preprocessor_fit_scope": "inner_fit_only",
        "official_validation_used": False,
        "test_used": False,
        "class_distribution": {
            "fit": _class_distribution(fit_y),
            "early_stop": _class_distribution(stop_y),
        },
        "class_weight_from": "inner_fit_labels_only",
        "class_weight": _weights_for_json(class_weights),
        "input_feature_columns": len(dataset.roles.model_features),
        "transformed_feature_columns": len(feature_names),
        "transformed_feature_names_sha256": _feature_names_sha256(feature_names),
        "matrix_audit": {
            "fit": fit_matrix_audit,
            "early_stop": stop_matrix_audit,
        },
        "preprocess_seconds": round(preprocess_seconds, 3),
        "train_seconds": round(train_seconds, 3),
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "best_monitor_value": best_value,
        "early_stopping": {
            "monitor": settings["early_stopping_monitor"],
            "mode": settings["early_stopping_mode"],
            "patience": settings["early_stopping_patience"],
            "min_delta": settings["early_stopping_min_delta"],
            "restore_best_weights": settings[
                "early_stopping_restore_best_weights"
            ],
            "stopped_epoch_zero_based": int(callback.stopped_epoch),
        },
        "history": history,
    }
    del model, preprocessor, fit_dense, stop_dense, fit_y, stop_y, history_object
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def _positive_scores(model: keras.Model, values: np.ndarray, *, batch_size: int) -> np.ndarray:
    predictions = model.predict(values, batch_size=batch_size, verbose=0)
    scores = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if (
        scores.shape != (len(values),)
        or not np.isfinite(scores).all()
        or ((scores < 0.0) | (scores > 1.0)).any()
    ):
        raise MLPTrainingError("MLP validation 출력이 [0, 1] 유한 점수가 아닙니다.")
    return scores


def _score_summary(scores: np.ndarray) -> dict[str, float]:
    percentiles = np.quantile(scores, [0.01, 0.1, 0.5, 0.9, 0.99])
    return {
        "minimum": float(np.min(scores)),
        "p01": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "median": float(percentiles[2]),
        "mean": float(np.mean(scores)),
        "p90": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "maximum": float(np.max(scores)),
        "standard_deviation": float(np.std(scores)),
    }


def _run_final_refit(
    dataset: DevelopmentDataset,
    settings: dict[str, Any],
    *,
    best_epoch: int,
    artifact_dir: Path,
    intra_threads: int,
    inter_threads: int,
    fit_verbose: int,
) -> dict[str, Any]:
    if best_epoch < 1 or best_epoch > settings["max_epochs"]:
        raise MLPTrainingError("full-train refit epoch가 고정 범위를 벗어납니다.")
    train_y = dataset.train.y.to_numpy(dtype=np.int8, copy=False)
    validation_y = dataset.validation.y.to_numpy(dtype=np.int8, copy=False)
    class_weights = _balanced_class_weights(train_y)
    preprocessor = make_preprocessor(dataset.roles, model_family="linear")

    preprocess_started = time.perf_counter()
    preprocessor.fit(dataset.train.X)
    train_values = preprocessor.transform(dataset.train.X)
    train_dense, train_matrix_audit = _dense_float32(
        train_values, label="full-train"
    )
    del train_values
    preprocess_train_seconds = time.perf_counter() - preprocess_started
    feature_names = transformed_feature_names(preprocessor)
    if train_dense.shape[1] != len(feature_names):
        raise MLPTrainingError("full-train 전처리 출력 피처 수가 일치하지 않습니다.")

    _reset_tensorflow(
        intra_threads=intra_threads,
        inter_threads=inter_threads,
    )
    model = _build_mlp(train_dense.shape[1], settings)
    train_started = time.perf_counter()
    final_history_object = model.fit(
        train_dense,
        train_y,
        epochs=best_epoch,
        batch_size=settings["batch_size"],
        class_weight=class_weights,
        shuffle=settings["shuffle"],
        verbose=fit_verbose,
    )
    train_seconds = time.perf_counter() - train_started
    if len(final_history_object.history.get("loss", [])) != best_epoch:
        raise MLPTrainingError("full-train refit이 best epoch만큼 실행되지 않았습니다.")
    del train_dense
    gc.collect()

    validation_transform_started = time.perf_counter()
    validation_values = preprocessor.transform(dataset.validation.X)
    validation_dense, validation_matrix_audit = _dense_float32(
        validation_values, label="official-validation"
    )
    del validation_values
    validation_transform_seconds = time.perf_counter() - validation_transform_started
    if validation_dense.shape[1] != len(feature_names):
        raise MLPTrainingError("validation 전처리 출력 피처 수가 full-train과 다릅니다.")
    predict_started = time.perf_counter()
    scores = _positive_scores(
        model,
        validation_dense,
        batch_size=settings["predict_batch_size"],
    )
    predict_seconds = time.perf_counter() - predict_started
    del validation_dense
    metrics = evaluate_binary_metrics(
        validation_y,
        scores,
        threshold=CLASSIFICATION_THRESHOLD,
        top_fraction=TOP_FRACTION,
    )

    preprocessor_path = artifact_dir / "mlp_v3_preprocessor.joblib"
    model_path = artifact_dir / "mlp_v3.keras"
    score_path = artifact_dir / "mlp_validation_scores.joblib"
    manifest_path = artifact_dir / "mlp_v3_manifest.json"
    preprocessor_artifact = _atomic_joblib_dump(preprocessor, preprocessor_path)
    model_artifact = _atomic_keras_save(model, model_path)
    score_artifact = _atomic_joblib_dump(
        {
            "schema_version": SCHEMA_VERSION,
            "run_version": RUN_VERSION,
            "split": "validation",
            "customer_ids_included": False,
            "y_true": validation_y.copy(),
            "scores": {"mlp_v3": scores},
        },
        score_path,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "data_version": "v3",
        "parquet_sha256": dataset.parquet_sha256,
        "schema_sha256": dataset.schema_sha256,
        "best_epoch": best_epoch,
        "input_feature_columns": len(dataset.roles.model_features),
        "transformed_feature_columns": len(feature_names),
        "transformed_feature_names_sha256": _feature_names_sha256(feature_names),
        "artifacts": {
            "preprocessor": preprocessor_artifact,
            "model": model_artifact,
            "validation_scores": score_artifact,
        },
    }
    _artifact_ignore_status(manifest_path)
    _atomic_write_json(manifest_path, manifest)
    manifest_artifact = _artifact_metadata(manifest_path)
    result = {
        "key": "mlp_v3",
        "display_name": "V3 TensorFlow MLP",
        "data_version": "v3",
        "estimator": "tensorflow_mlp",
        "preprocessor_fit_scope": "full_train_only",
        "full_refit_validation_data_used": False,
        "full_refit_callbacks_used": False,
        "official_validation_prediction_calls": 1,
        "test_used": False,
        "best_epoch_source": "train_internal_early_stop",
        "best_epoch": best_epoch,
        "class_distribution": _class_distribution(train_y),
        "class_weight_from": "full_train_labels_only",
        "class_weight": _weights_for_json(class_weights),
        "probability_calibrated": False,
        "brier_comparable_to_unweighted_models": False,
        "threshold_0_5_is_diagnostic_only": True,
        "input_feature_columns": len(dataset.roles.model_features),
        "transformed_feature_columns": len(feature_names),
        "transformed_feature_names_sha256": _feature_names_sha256(feature_names),
        "model_input_columns": int(model.input_shape[-1]),
        "trainable_parameters": int(model.count_params()),
        "matrix_audit": {
            "train": train_matrix_audit,
            "validation": validation_matrix_audit,
        },
        "preprocess_train_seconds": round(preprocess_train_seconds, 3),
        "train_seconds": round(train_seconds, 3),
        "validation_transform_seconds": round(validation_transform_seconds, 3),
        "validation_predict_seconds": round(predict_seconds, 3),
        "score_summary": _score_summary(scores),
        "metrics": metrics,
        "artifacts": {
            "preprocessor": preprocessor_artifact,
            "model": model_artifact,
            "validation_scores": score_artifact,
            "manifest": manifest_artifact,
        },
    }
    del model, preprocessor, final_history_object, scores
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def _metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    if metric == "recall_at_10pct":
        value = metrics["top_k_metrics"].get("recall")
    elif metric == "lift_at_10pct":
        value = metrics["top_k_metrics"].get("lift")
    else:
        value = metrics.get(metric)
    return None if value is None else float(value)


def _build_comparisons(
    experiment: dict[str, Any],
    references: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for reference in references:
        deltas: dict[str, float | None] = {}
        for metric in (
            "roc_auc",
            "pr_auc",
            "ks",
            "gini",
            "recall_at_10pct",
            "lift_at_10pct",
        ):
            newer = _metric_value(experiment["metrics"], metric)
            older = _metric_value(reference["metrics"], metric)
            deltas[metric] = (
                None if newer is None or older is None else newer - older
            )
        result[f"mlp_v3_minus_{reference['key']}"] = {
            **deltas,
            "brier_score": None,
            "brier_score_note": "not_comparable_class_weighted_uncalibrated_mlp",
        }
    return result


def _atomic_history_figure(history: dict[str, list[float]], path: Path) -> dict[str, Any]:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["loss"], label="inner-fit")
    axes[0].plot(epochs, history["val_loss"], label="inner-early-stop")
    axes[0].set(title="MLP Binary Crossentropy", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, history["pr_auc_approx"], label="inner-fit")
    axes[1].plot(
        epochs,
        history["val_pr_auc_approx"],
        label="inner-early-stop",
    )
    axes[1].set(title="MLP PR-AUC (Keras approximation)", xlabel="Epoch", ylabel="PR-AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        figure.savefig(temporary, dpi=160, bbox_inches="tight")
        temporary.replace(path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return {
        "display_path": _display_path(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _format_metric(value: Any, digits: int = 4, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    prefix = "+" if signed and float(value) >= 0 else ""
    return f"{prefix}{float(value):.{digits}f}"


def render_markdown_report(payload: dict[str, Any]) -> str:
    experiment = payload["experiment"]
    metrics = experiment["metrics"]
    top_k = metrics["top_k_metrics"]
    reference_by_key = {
        item["key"]: item for item in payload["reference_candidates"]
    }
    lines = [
        "# Stage 5 2/3 TensorFlow MLP 보고서",
        "",
        "> V3 공식 train 내부에서 early stopping epoch를 정한 뒤 새 MLP를 train 전체로 재학습하고 공식 validation에서 한 번 비교한 결과입니다. test는 사용하지 않았습니다.",
        "",
        "## 왜 MLP를 비교했는가",
        "",
        "LightGBM은 트리를 순차적으로 결합하고 MLP는 여러 신경망 층을 통해 피처 조합을 학습합니다. 같은 V3 데이터에서 MLP를 비교해 딥러닝의 추가 복잡도와 계산비용이 실제 성능 향상으로 이어지는지 확인했습니다.",
        "",
        "## 누수 방지 학습 흐름",
        "",
        "1. 공식 train을 90% inner-fit과 10% inner-early-stop으로 층화 분리했습니다.",
        "2. 발견 단계 전처리는 inner-fit에만 fit하고 inner-early-stop은 학습 중단 epoch 선택에만 사용했습니다.",
        "3. best epoch를 고정한 뒤 발견 모델을 버리고, 새 전처리기와 새 MLP를 공식 train 전체로 해당 epoch만큼 재학습했습니다.",
        "4. 공식 validation은 full-train refit에 전달하지 않고 마지막 예측 한 번에만 사용했습니다.",
        "5. test 피처·예측·평가는 사용하지 않았습니다.",
        "",
        "## Early stopping 결과",
        "",
        f"- 실행 epoch: {payload['inner_development']['epochs_ran']}",
        f"- 선택한 best epoch: {payload['inner_development']['best_epoch']}",
        f"- best inner PR-AUC 근사값: {_format_metric(payload['inner_development']['best_monitor_value'])}",
        f"- inner-fit 고객: {payload['inner_split']['fit']['rows']:,}명",
        f"- inner-early-stop 고객: {payload['inner_split']['early_stop']['rows']:,}명",
        "",
        f"![MLP 학습 이력](../{payload['training_history_figure']['display_path']})",
        "",
        "Keras PR-AUC는 epoch 선택용 threshold 근사치입니다. 아래 공식 비교 수치는 프로젝트 공통 평가 함수의 Average Precision으로 다시 계산했습니다.",
        "",
        "## 공식 validation 결과",
        "",
        "| 모델 | ROC-AUC | PR-AUC(AP) | KS | Gini | Brier(진단) | Recall@10% | Lift@10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| V3 TensorFlow MLP | {roc} | {pr} | {ks} | {gini} | {brier} | {recall} | {lift} |".format(
            roc=_format_metric(metrics["roc_auc"]),
            pr=_format_metric(metrics["pr_auc"]),
            ks=_format_metric(metrics["ks"]),
            gini=_format_metric(metrics["gini"]),
            brier=_format_metric(metrics["brier_score"]),
            recall=_format_metric(top_k["recall"]),
            lift=_format_metric(top_k["lift"]),
        ),
        "",
        "상환곤란 고객이 적어 학습 시 class weight를 사용했습니다. 따라서 sigmoid 출력은 아직 보정된 실제 확률이 아니며 Brier와 0.5 threshold 결과는 보정 전 진단값입니다. 비가중 Logistic·LightGBM과 Brier를 직접 비교하지 않습니다.",
        "",
        "## 기존 V3 모델과의 순위 성능 비교",
        "",
        "| 비교 | Δ ROC-AUC | Δ PR-AUC | Δ KS | Δ Recall@10% |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("logistic_v3", "MLP - Logistic"),
        ("random_forest_v3", "MLP - Random Forest"),
        ("lightgbm_v3", "MLP - LightGBM"),
    ):
        delta = payload["comparisons"][f"mlp_v3_minus_{key}"]
        lines.append(
            "| {label} | {roc} | {pr} | {ks} | {recall} |".format(
                label=label,
                roc=_format_metric(delta["roc_auc"], signed=True),
                pr=_format_metric(delta["pr_auc"], signed=True),
                ks=_format_metric(delta["ks"], signed=True),
                recall=_format_metric(delta["recall_at_10pct"], signed=True),
            )
        )
    lightgbm_metrics = reference_by_key["lightgbm_v3"]["metrics"]
    lines.extend(
        [
            "",
            "## 현재 해석",
            "",
            f"- V3 MLP의 ROC-AUC/PR-AUC는 {_format_metric(metrics['roc_auc'])}/{_format_metric(metrics['pr_auc'])}입니다.",
            f"- 현재 고정 설정 V3 LightGBM은 {_format_metric(lightgbm_metrics['roc_auc'])}/{_format_metric(lightgbm_metrics['pr_auc'])}입니다.",
            "- 이 결과는 MLP 고정 비교 후보를 추가한 것이며 Stage 5의 최종 모델 선정이 아닙니다.",
            "- validation 결과를 보고 MLP 구조, class weight 또는 epoch를 다시 바꾸지 않습니다. 제한 개선과 확률 보정은 Stage 5 3/3에서 train 내부 데이터로만 수행합니다.",
            "",
            "## 실행 환경과 산출물 감사",
            "",
            f"- TensorFlow/Keras: {payload['environment']['tensorflow']} / {payload['environment']['keras']}",
            f"- CPU thread: intra {payload['settings']['tensorflow_threads']['intra_op']}, inter {payload['settings']['tensorflow_threads']['inter_op']}",
            f"- train 고객: {payload['data_scope']['train_rows']:,}명",
            f"- validation 고객: {payload['data_scope']['validation_rows']:,}명",
            f"- test 피처 사용: {payload['data_scope']['test_feature_rows_used']}행",
            "- 모델·전처리·행별 점수는 models/stage5에 저장하고 Git에서 제외했습니다.",
            "- 공유 JSON·Markdown에는 고객 ID와 행별 점수를 포함하지 않았습니다.",
            "",
            "## 다음 작업",
            "",
            "Stage 5 3/3에서 train 내부 제한 튜닝, class weight 대안, 확률 보정과 피처군 분석을 수행한 뒤 전체 후보를 비교합니다. test는 계속 봉인합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def run_mlp_experiment(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    figure_path: str | Path = DEFAULT_FIGURE,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    stage4_reference_path: str | Path = DEFAULT_STAGE4_REFERENCE,
    lightgbm_reference_path: str | Path = DEFAULT_LIGHTGBM_REFERENCE,
    intra_threads: int = 2,
    inter_threads: int = 1,
    fit_verbose: int = 2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if fit_verbose not in (0, 2):
        raise MLPTrainingError("TensorFlow fit verbose는 0 또는 2여야 합니다.")
    output = Path(output_path)
    report = Path(report_path)
    figure = Path(figure_path)
    artifacts = Path(artifact_dir)
    stage4_path = Path(stage4_reference_path)
    lightgbm_path = Path(lightgbm_reference_path)
    _validate_shared_paths(
        output=output,
        report=report,
        figure=figure,
        artifacts=artifacts,
        stage4_reference=stage4_path,
        lightgbm_reference=lightgbm_path,
    )
    _configure_tensorflow(
        intra_threads=intra_threads,
        inter_threads=inter_threads,
    )
    stage4, lightgbm = _load_references(stage4_path, lightgbm_path)
    settings = _mlp_settings()
    started = time.perf_counter()
    dataset = load_development_data("v3")
    _validate_dataset(dataset, stage4, lightgbm)
    split = _make_inner_split(dataset.train.y.to_numpy(dtype=np.int8, copy=False))
    devices = [
        {"device_type": device.device_type, "name": device.name}
        for device in tf.config.list_physical_devices()
    ]
    visible_devices = [
        {"device_type": device.device_type, "name": device.name}
        for device in tf.config.get_visible_devices()
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_version": RUN_VERSION,
        "stage_part": STAGE_PART,
        "run_status": "in_progress",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "physical_devices": devices,
            "visible_devices": visible_devices,
            "gpu_used": any(
                device["device_type"] == "GPU" for device in visible_devices
            ),
        },
        "settings": {
            "random_seed": RANDOM_SEED,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "top_fraction": TOP_FRACTION,
            "inner_holdout_fraction": INNER_HOLDOUT_FRACTION,
            "mlp": settings,
            "tensorflow_threads": {
                "intra_op": intra_threads,
                "inter_op": inter_threads,
            },
            "op_determinism_enabled": True,
            "onednn_custom_ops_enabled": False,
            "configuration_locked_before_official_validation": True,
            "official_validation_used_for_early_stopping": False,
            "official_validation_result_driven_retuning": False,
        },
        "data_scope": {
            "train_rows": len(dataset.train.y),
            "validation_rows": len(dataset.validation.y),
            "validation_positive_rate": float(dataset.validation.y.mean()),
            "test_feature_rows_used": 0,
            "test_predictions_created": False,
            "customer_ids_in_shared_outputs": False,
            "row_level_predictions_in_shared_outputs": False,
        },
        "references": {
            "stage4": {
                "display_path": _display_path(stage4_path),
                "sha256": _sha256(stage4_path),
                "run_version": stage4.get("run_version"),
            },
            "lightgbm": {
                "display_path": _display_path(lightgbm_path),
                "sha256": _sha256(lightgbm_path),
                "run_version": lightgbm.get("run_version"),
            },
        },
        "reference_candidates": _reference_candidates(stage4, lightgbm),
        "data_version": {
            "version": "v3",
            "schema_sha256": dataset.schema_sha256,
            "parquet_sha256": dataset.parquet_sha256,
            "model_feature_columns": len(dataset.roles.model_features),
            "numeric_feature_columns": len(dataset.roles.numeric),
            "categorical_feature_columns": len(dataset.roles.categorical),
            "stage3_summary_verified": dataset.audit["stage3_summary_verified"],
            "parquet_sha256_verified": dataset.audit["parquet_sha256_verified"],
            "stage4_reference_verified": True,
            "lightgbm_reference_verified": True,
        },
        "inner_split": split.audit,
        "inner_development": {},
        "experiment": {},
        "comparisons": {},
        "training_history_figure": None,
        "resources": {},
    }
    _atomic_write_json(output, payload)
    if progress is not None:
        progress("train 내부 90%/10% 전처리와 early stopping 학습 시작")
    discovery = _run_discovery(
        dataset,
        split,
        settings,
        intra_threads=intra_threads,
        inter_threads=inter_threads,
        fit_verbose=fit_verbose,
    )
    payload["inner_development"] = discovery
    _atomic_write_json(output, payload)
    if progress is not None:
        progress(
            f"best epoch {discovery['best_epoch']} 확정; "
            "새 전처리기·새 MLP로 full-train refit 시작"
        )
    experiment = _run_final_refit(
        dataset,
        settings,
        best_epoch=discovery["best_epoch"],
        artifact_dir=artifacts,
        intra_threads=intra_threads,
        inter_threads=inter_threads,
        fit_verbose=fit_verbose,
    )
    payload["experiment"] = experiment
    payload["comparisons"] = _build_comparisons(
        experiment, payload["reference_candidates"]
    )
    payload["training_history_figure"] = _atomic_history_figure(
        discovery["history"], figure
    )
    payload["resources"] = {
        "total_seconds": round(time.perf_counter() - started, 3),
        "process_peak_rss_mb": _peak_rss_mb(),
        "measurement_scope": "current_process_lifetime",
        "dense_matrix_limit_mib": MAX_DENSE_MATRIX_MIB,
    }
    payload["run_status"] = "complete"
    payload["completed_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _atomic_write_json(output, payload)
    _atomic_write_text(report, render_markdown_report(payload))
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5 2/3 V3 TensorFlow MLP를 누수 없이 학습합니다."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--stage4-reference", type=Path, default=DEFAULT_STAGE4_REFERENCE
    )
    parser.add_argument(
        "--lightgbm-reference", type=Path, default=DEFAULT_LIGHTGBM_REFERENCE
    )
    parser.add_argument("--intra-threads", type=int, default=2)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument("--fit-verbose", type=int, choices=(0, 2), default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_mlp_experiment(
        output_path=args.output,
        report_path=args.report,
        figure_path=args.figure,
        artifact_dir=args.artifact_dir,
        stage4_reference_path=args.stage4_reference,
        lightgbm_reference_path=args.lightgbm_reference,
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
        fit_verbose=args.fit_verbose,
        progress=lambda message: print(message, flush=True),
    )
    metrics = result["experiment"]["metrics"]
    print(
        "Stage 5 2/3 TensorFlow MLP 완료: "
        f"best epoch {result['inner_development']['best_epoch']}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
        f"test 피처 {result['data_scope']['test_feature_rows_used']}행 사용"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
