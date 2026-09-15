"""정책 경계·동결 변조·재실행·검토 범위를 검증한다."""

import json
from pathlib import Path

import numpy as np
import pytest

from creditlens.modeling import frozen_model as frozen
from creditlens.modeling import finalize_stage6 as final


def policy():
    return {
        "schema_version": "1.0", "status": "frozen", "medium_cutoff": 0.08,
        "high_cutoff": 0.18, "review_cutoff": 0.18, "comparison": ">=",
        "calibration_method": "identity", "group_specific_cutoffs": False,
        "automatic_credit_decision": False,
    }


def test_boundary_inclusion_before_rounding_and_batch_independence():
    scores = [0, np.nextafter(.08, 0), .08, np.nextafter(.18, 0), .18, 1]
    result = frozen.apply_risk_policy(scores, policy())
    assert result["risk_band"].tolist() == ["low", "low", "medium", "medium", "high", "high"]
    assert result["priority_review"].tolist() == [False, False, False, False, True, True]
    # 단건과 배치의 판정이 같고 경계 동점은 모두 포함한다.
    assert frozen.apply_risk_policy([.18], policy())["risk_band"].item() == "high"
    assert frozen.apply_risk_policy([.18] * 10, policy())["priority_review"].all()


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [-.01], [1.01], [], [[.2]]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        frozen.apply_risk_policy(scores, policy())


def test_policy_and_array_shape_mismatch_rejected():
    with pytest.raises(frozen.FrozenModelError):
        frozen.apply_risk_policy([.2], {**policy(), "review_cutoff": .4})
    with pytest.raises(frozen.FrozenModelError):
        final.check_close(np.zeros((2, 1)), np.zeros(2), "shape")


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = source / "model.joblib"
    calibrator = source / "calibrator.joblib"
    # 무결성 검사는 pickle 역직렬화 없이 임의 바이트에도 수행 가능해야 한다.
    model.write_bytes(b"synthetic-model")
    calibrator.write_bytes(b"synthetic-calibrator")
    root = tmp_path / "project"
    for relative in frozen.RUNTIME_SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source", encoding="utf-8")
    dest = tmp_path / "models" / "frozen"
    manifest = {
        "status": "frozen", "release": "test-release", "runtime": frozen.runtime_versions(),
        "runtime_sources": {p: frozen.file_digest(root / p) for p in frozen.RUNTIME_SOURCES},
    }
    meta = final.write_bundle(dest, model, calibrator, policy(), manifest)
    record = {"run_status": "complete", "stage_part": "3/3", "release": "test-release",
              "data_scope": {"test_feature_rows_used": 0}, "policy": policy(),
              "bundle_manifest": meta}
    result = tmp_path / "result.json"
    result.write_text(json.dumps(record), encoding="utf-8")
    return dest, result, root, model, calibrator, manifest


def test_bundle_verification_needs_no_data_or_deserialization(bundle, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("검증 전용 실행은 데이터/모델 로드를 해서는 안 됩니다.")
    monkeypatch.setattr(final, "load_model_split", forbidden)
    monkeypatch.setattr(frozen.joblib, "load", forbidden)
    dest, result, root, *_ = bundle
    assert frozen.verify_bundle(dest, result, project_root=root)["status"] == "frozen"


@pytest.mark.parametrize("name", ["model.joblib", "calibrator.joblib", "policy.json", "manifest.json"])
def test_changed_bundle_file_rejected(bundle, name):
    dest, result, root, *_ = bundle
    path = dest / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(frozen.FrozenModelError, match="불일치"):
        frozen.verify_bundle(dest, result, project_root=root)


def test_changed_source_or_runtime_rejected(bundle, monkeypatch):
    dest, result, root, *_ = bundle
    original = frozen.runtime_versions()
    monkeypatch.setattr(frozen, "runtime_versions", lambda: {**original, "python": "changed"})
    with pytest.raises(frozen.FrozenModelError, match="버전"):
        frozen.verify_bundle(dest, result, project_root=root)
    monkeypatch.setattr(frozen, "runtime_versions", lambda: original)
    (root / frozen.RUNTIME_SOURCES[0]).write_text("changed", encoding="utf-8")
    with pytest.raises(frozen.FrozenModelError, match="소스"):
        frozen.verify_bundle(dest, result, project_root=root)


def test_existing_bundle_is_never_overwritten(bundle):
    dest, _, _, model, calibrator, manifest = bundle
    before = frozen.file_digest(dest / "model.joblib")
    with pytest.raises(frozen.FrozenModelError, match="이미"):
        final.write_bundle(dest, model, calibrator, policy(), manifest)
    assert before == frozen.file_digest(dest / "model.joblib")


def test_new_alert_requires_review_before_freeze():
    with pytest.raises(frozen.FrozenModelError, match="검토 대상"):
        final.audit_alerts({"subgroup_analysis": {"reliable_group_alerts": []}},
                           None, np.array([0, 1]), np.array([.1, .2]), .18)


def test_finalizer_refuses_existing_result_before_loading_data(tmp_path, monkeypatch):
    dest = tmp_path / "existing"
    dest.mkdir()
    monkeypatch.setattr(final, "DEFAULT_BUNDLE", dest)
    with pytest.raises(frozen.FrozenModelError, match="덮어쓰지"):
        final.finalize_stage6()
