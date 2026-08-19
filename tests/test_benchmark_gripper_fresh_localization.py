from contextlib import nullcontext
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodi.sim.benchmark import file_sha256
import embodi.sim.benchmark_gripper_fresh_localization as fresh


ROOT = Path(__file__).parents[1]


def _metric_record(value: float, condition: str, seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "condition": condition,
        "morphology": "so101",
        "task": "push_to_zone",
        "scenario_id": "synthetic",
        "phase": "push",
        "endpoint": "first",
        "metrics": {
            **{name: value for name in fresh.exp43.METRICS},
            "channel_losses": [value] * 10,
            "degenerate_rotation": False,
        },
    }


def _metrics(full: float, half: float, joint: float) -> dict[str, object]:
    values = {"full": full, "half": half, "joint": joint}
    return fresh.exp43.aggregate_metrics(
        [
            _metric_record(values[condition], condition, seed)
            for seed in fresh.SEEDS
            for condition in fresh.CONDITIONS
        ]
    )


def _difference(value: str) -> dict[str, object]:
    return {
        "left_difference": float(Decimal(value)),
        "right_difference": 0.0,
        "difference": float(Decimal(value)),
        "difference_decimal": value,
    }


def _localization(
    difference: str = "0.02", *, left: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "left": "closed_target",
        "right": "open_target",
        "left_metrics": left or _metrics(0.10, 0.105, 0.12),
        "right_metrics": _metrics(0.10, 0.105, 0.10),
        "overall": _difference(difference),
        "by_seed": {str(seed): _difference(difference) for seed in fresh.SEEDS},
        "by_morphology": {
            morphology: _difference(difference) for morphology in fresh.MORPHOLOGIES
        },
    }


def test_support_binding_checks_both_report_and_support_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "synthetic-support.json"
    report = {"provenance": {"evaluator_sha256": fresh.SUPPORT_EVALUATOR_SHA256}}
    raw = json.dumps(report).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(fresh, "SUPPORT_PATH", path)
    monkeypatch.setattr(fresh, "SUPPORT_REPORT_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(fresh.support, "validate_support_report", lambda value, manifest: None)
    assert fresh.load_support_report(path, SimpleNamespace()) == report

    path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fresh.load_support_report(path, SimpleNamespace())

    path.write_bytes(raw)
    report["provenance"]["evaluator_sha256"] = "0" * 64
    changed = json.dumps(report).encode()
    path.write_bytes(changed)
    monkeypatch.setattr(fresh, "SUPPORT_REPORT_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="support evaluator identity"):
        fresh.load_support_report(path, SimpleNamespace())


def test_recapture_comparison_excludes_only_image_and_rejects_policy_or_extra_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "scenario_id": "development-so101-push_to_zone-0042",
        "morphology": "so101",
        "state_sha256": "a" * 64,
        "expert_target": [[0.0] * 10],
        "image_sha256": "1" * 64,
    }
    monkeypatch.setattr(fresh.support, "_validate_records", lambda *args: None)
    monkeypatch.setattr(fresh.support, "support_scenarios", lambda *args: ())
    monkeypatch.setattr(fresh, "EXPECTED_ENDPOINTS", 2)
    monkeypatch.setattr(fresh, "EXPECTED_EPISODES", 2)
    support_report = {
        "records": [base],
        "episodes": [{"scenario_id": "development-so101-push_to_zone-0042"}],
    }
    actual = {**base, "image_sha256": "2" * 64}
    fresh.validate_recaptured_support(
        [actual], support_report["episodes"], support_report, SimpleNamespace(), "so101"
    )
    with pytest.raises(ValueError, match="structural support mismatch"):
        fresh.validate_recaptured_support(
            [{**actual, "seeds": {}}],
            support_report["episodes"],
            support_report,
            SimpleNamespace(),
            "so101",
        )
    assert fresh._without_policy_results([{**actual, "seeds": {}}]) == [actual]
    with pytest.raises(ValueError, match="structural support mismatch"):
        fresh.validate_recaptured_support(
            [{**actual, "unexpected": True}],
            support_report["episodes"],
            support_report,
            SimpleNamespace(),
            "so101",
        )
    actual["state_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="structural support mismatch"):
        fresh.validate_recaptured_support(
            [actual], support_report["episodes"], support_report, SimpleNamespace(), "so101"
        )


def test_second_morphology_recapture_failure_precedes_all_policy_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(fresh, "REGISTERED_RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(fresh, "REGISTERED_LOCK_PATH", tmp_path / ".lock")
    monkeypatch.setattr(fresh, "registered_run_lock", lambda: nullcontext())
    monkeypatch.setattr(fresh, "registered_evaluator_sha256", lambda: file_sha256(Path(fresh.__file__)))
    monkeypatch.setattr(fresh, "_runtime_provenance", lambda _device: {"git_dirty": False})
    monkeypatch.setattr(fresh, "load_support_report", lambda *_args: {"synthetic": True})
    recaptures: list[str] = []

    def recapture(_manifest, _report, morphology, **_kwargs):
        recaptures.append(morphology)
        if morphology == "panda":
            raise ValueError("synthetic Panda recapture mismatch")
        return [], []

    calls = {"source_closure": 0, "source_contract": 0, "resolver": 0, "policy_loader": 0}

    def counted(name: str):
        def call(*_args, **_kwargs):
            calls[name] += 1
            return None

        return call

    monkeypatch.setattr(fresh, "recapture_support", recapture)
    monkeypatch.setattr(fresh, "_validate_source_file_contract", counted("source_closure"))
    monkeypatch.setattr(fresh, "_source_contract", counted("source_contract"))
    monkeypatch.setattr(fresh, "resolve_checkpoint", counted("resolver"))
    monkeypatch.setattr(fresh, "load_geometry_policy", counted("policy_loader"))
    args = SimpleNamespace(device="cpu", support=Path("synthetic.json"))
    with pytest.raises(ValueError, match="Panda recapture mismatch"):
        fresh._run_all_registered(args, SimpleNamespace(), SimpleNamespace())
    assert recaptures == list(fresh.MORPHOLOGIES)
    assert calls == {
        "source_closure": 0,
        "source_contract": 0,
        "resolver": 0,
        "policy_loader": 0,
    }


def _persisted_report(morphology: str = "so101") -> dict[str, object]:
    runtime = {name: None for name in fresh.exp43.RUNTIME_FIELDS}
    runtime.update({"device": "cpu", "python": "3.12", "torch": "2", "cuda_runtime": None})
    signature = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "python": runtime["python"],
        "torch": runtime["torch"],
        "cuda": runtime["cuda_runtime"],
        "cudnn": runtime["cudnn"],
        "lerobot": "synthetic",
        "transformers": "synthetic",
        "trainer_sha256": fresh.SOURCE_FILE_SHA256["src/embodi/joint_train.py"],
        "runtime": runtime,
        "runtime_signature_sha256": signature,
        "evaluator_sha256": file_sha256(Path(fresh.__file__)),
        "imported_exp41_evaluator_sha256": fresh.EXP41_EVALUATOR_SHA256,
        "imported_exp42_evaluator_sha256": fresh.EXP42_EVALUATOR_SHA256,
        "imported_exp43_evaluator_sha256": fresh.EXP43_EVALUATOR_SHA256,
        "imported_exp44_evaluator_sha256": fresh.EXP44_EVALUATOR_SHA256,
        "imported_support_evaluator_sha256": fresh.SUPPORT_EVALUATOR_SHA256,
        "expert_sha256": fresh.EXPERT_SHA256,
        "support_report_sha256": fresh.SUPPORT_REPORT_SHA256,
        "source_file_sha256": dict(fresh.SOURCE_FILE_SHA256),
    }
    return {
        "identity": {
            "morphology": morphology,
            "model_seeds": list(fresh.SEEDS),
            "checkpoints": {},
        },
        "provenance": provenance,
        "run": {
            "started_utc": "2026-01-01T00:00:00Z",
            "finished_utc": "2026-01-01T00:00:01Z",
            "device": "cpu",
            "lock_identity": fresh.LOCK_IDENTITY,
            "lock_path": str(fresh.REGISTERED_LOCK_PATH),
            "schedule_identity": fresh.SCHEDULE_IDENTITY,
            "schedule_position": fresh.MORPHOLOGIES.index(morphology),
            "orchestrator_run_id": "b" * 32,
            "seed_order": list(fresh.SEEDS),
            "condition_order_formula": fresh.exp43.PROTOCOL["condition_order_formula"],
        },
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("identity", "model_seeds"), [36001]),
        (("run", "seed_order"), [36001]),
        (("run", "condition_order_formula"), "changed"),
        (("provenance", "imported_exp41_evaluator_sha256"), "0" * 64),
        (("provenance", "expert_sha256"), "0" * 64),
        (("provenance", "source_file_sha256"), {}),
        (("provenance", "git_commit"), "not-a-commit"),
    ],
)
def test_persisted_identity_and_provenance_tampering_fails(
    path: tuple[str, str], value: object
) -> None:
    report = _persisted_report()
    report[path[0]][path[1]] = value
    with pytest.raises(ValueError, match="persisted identity or provenance"):
        fresh._validate_persisted_identity(report, "so101")


def test_runtime_field_and_cross_shard_provenance_tampering_fails() -> None:
    report = _persisted_report()
    fresh._validate_persisted_identity(report, "so101")
    changed_runtime = deepcopy(report)
    changed_runtime["provenance"]["runtime"]["extra"] = True
    with pytest.raises(ValueError, match="persisted identity or provenance"):
        fresh._validate_persisted_identity(changed_runtime, "so101")

    panda = _persisted_report("panda")
    panda["run"]["started_utc"] = "2026-01-01T00:00:02Z"
    panda["run"]["finished_utc"] = "2026-01-01T00:00:03Z"
    reports = {"so101": report, "panda": panda}
    assert fresh._common_shard_identity(reports) == ("a" * 40, "b" * 32, report["provenance"]["runtime_signature_sha256"])
    reports["panda"]["provenance"]["git_commit"] = "c" * 40
    with pytest.raises(ValueError, match="common clean commit/runtime"):
        fresh._common_shard_identity(reports)


def test_live_provenance_rejects_shared_fake_commit_and_recomputed_fake_runtime() -> None:
    report = _persisted_report()
    persisted = report["provenance"]
    committed = fresh._expected_committed_source_sha256()
    fresh._validate_live_provenance(
        persisted,
        "cpu",
        expected_current_provenance=persisted,
        expected_commit="a" * 40,
        committed_source_sha256=committed,
    )

    # A shared, syntactically valid claim is insufficient when trusted HEAD differs.
    with pytest.raises(ValueError, match="committed source provenance differs"):
        fresh._validate_live_provenance(
            persisted,
            "cpu",
            expected_current_provenance=persisted,
            expected_commit="c" * 40,
            committed_source_sha256=committed,
        )

    recomputed = deepcopy(persisted)
    recomputed["runtime"]["python"] = "different-runtime"
    recomputed["python"] = "different-runtime"
    recomputed["runtime_signature_sha256"] = hashlib.sha256(
        json.dumps(recomputed["runtime"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="live runtime"):
        fresh._validate_live_provenance(
            persisted,
            "cpu",
            expected_current_provenance=recomputed,
            expected_commit="a" * 40,
            committed_source_sha256=committed,
        )

    changed_blobs = dict(committed)
    changed_blobs["src/embodi/sim/benchmark.py"] = "0" * 64
    with pytest.raises(ValueError, match="committed source provenance differs"):
        fresh._validate_live_provenance(
            persisted,
            "cpu",
            expected_current_provenance=persisted,
            expected_commit="a" * 40,
            committed_source_sha256=changed_blobs,
        )


def test_complete_source_file_closure_rejects_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    required = {
        "src/embodi/processing.py",
        "src/embodi/token_config.py",
        "src/embodi/token_policy.py",
        "src/embodi/token_canonical.py",
        "src/embodi/backbone.py",
        "src/embodi/token_expert.py",
        "src/embodi/sim/benchmark.py",
        "src/embodi/sim/benchmark_geometry_evaluation.py",
        "src/embodi/sim/benchmark_task_instruction.py",
    }
    assert required <= set(fresh.SOURCE_FILE_SHA256)
    fresh._validate_source_file_contract()
    changed = dict(fresh.SOURCE_FILE_SHA256)
    changed["src/embodi/processing.py"] = "0" * 64
    monkeypatch.setattr(fresh, "SOURCE_FILE_SHA256", changed)
    with pytest.raises(ValueError, match="source closure hash mismatch"):
        fresh._validate_source_file_contract()


def test_three_condition_decimal_decomposition_is_exact() -> None:
    metrics = _metrics(0.10, 0.115, 0.13)
    contrasts = metrics["overall"]["effective_gripper_error"]["contrasts"]
    assert Decimal(contrasts["J-F"]["difference_decimal"]) == Decimal(
        contrasts["H-F"]["difference_decimal"]
    ) + Decimal(contrasts["J-H"]["difference_decimal"])


def test_materiality_exact_boundaries_and_zero_denominator() -> None:
    assert fresh._material_gate(_metrics(0.10, 0.105, 0.11))["supported"] is True
    assert fresh._material_gate(_metrics(0.10, 0.105, 0.109999))["supported"] is False
    zero = fresh._material_gate(_metrics(0.0, 0.0, 0.02))
    assert zero["supported"] is False
    assert zero["ratio"] is None


def test_localization_strict_signs_reject_zero_seed_and_morphology() -> None:
    contrast = _localization()
    contrast["by_seed"][str(fresh.SEEDS[0])] = _difference("0")
    contrast["by_seed"][str(fresh.SEEDS[1])] = _difference("0")
    assert fresh._localization_gate("target_state", contrast)["localized"] is False

    contrast = _localization()
    contrast["by_morphology"][fresh.MORPHOLOGIES[0]] = _difference("0")
    assert fresh._localization_gate("target_state", contrast)["localized"] is False


def test_opening_scope_broadness_and_materiality_boundaries() -> None:
    gate = fresh._localization_gate("opening_transition", _localization("0.01"))
    assert gate["localized"] is True
    assert gate["broad"] is False
    assert gate["classification"] == "pick-place-specific-closed_target-localization"

    below = _localization("0.009999")
    assert fresh._localization_gate("target_state", below)["localized"] is False


def test_decision_precedence_invalid_no_partition_multiple_and_distributed() -> None:
    low = _metrics(0.10, 0.102, 0.105)
    high = _metrics(0.10, 0.105, 0.12)
    low_partitions = {name: {"metrics": low} for name in fresh.PARTITIONS}
    high_partitions = {**low_partitions, fresh.PARTITIONS[0]: {"metrics": high}}
    localized = _localization()

    assert fresh.classify_localization(low_partitions, {}, delivery=False)["classification"] == (
        "invalid_delivery"
    )
    assert fresh.classify_localization(
        low_partitions, {"target_state": localized}, delivery=True
    )["classification"] == "not_materially_localized"
    assert fresh.classify_localization(
        high_partitions,
        {"target_state": localized, "terminal_behavior": localized},
        delivery=True,
    )["classification"] == "multiple-localizations"
    assert fresh.classify_localization(high_partitions, {}, delivery=True)["classification"] == (
        "distributed_material_deficit"
    )


def test_deterministic_gzip_binds_compressed_and_decompressed_hashes() -> None:
    raw = b'{"synthetic":true}\n'
    first = fresh.deterministic_gzip_bytes(raw)
    second = fresh.deterministic_gzip_bytes(raw)
    assert first == second
    assert fresh.validated_archive_bytes(
        first,
        compressed_sha256=hashlib.sha256(first).hexdigest(),
        decompressed_sha256=hashlib.sha256(raw).hexdigest(),
    ) == raw
    with pytest.raises(ValueError, match="compressed source hash"):
        fresh.validated_archive_bytes(
            first, compressed_sha256="0" * 64, decompressed_sha256=hashlib.sha256(raw).hexdigest()
        )
    with pytest.raises(ValueError, match="decompressed source hash"):
        fresh.validated_archive_bytes(
            first,
            compressed_sha256=hashlib.sha256(first).hexdigest(),
            decompressed_sha256="0" * 64,
        )


def test_cli_registration_guards_and_frozen_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    run = fresh.parse_args(["run-registered"])
    aggregate = fresh.parse_args(["aggregate"])
    assert run.support == fresh.SUPPORT_PATH
    assert aggregate.output == fresh.SUMMARY_PATH
    assert fresh.registered_shard_filename("panda") == (
        "benchmark-exp45-panda-fresh-gripper.json"
    )
    assert fresh.registered_archive_filename("panda").endswith(".json.gz")
    assert (
        fresh.EXPECTED_SCORED_CHUNKS,
        fresh.EXPECTED_REPEAT_CHUNKS,
        fresh.EXPECTED_INFERENCES,
        fresh.EXPECTED_CONTRAST_OBSERVATIONS,
    ) == (3276, 3276, 6552, 1092)
    monkeypatch.setattr(
        fresh,
        "parse_args",
        lambda: SimpleNamespace(command="aggregate", support=Path("other.json")),
    )
    with pytest.raises(ValueError, match="support path"):
        fresh.main()


def test_policy_registration_hash_and_cli_entrypoint() -> None:
    assert fresh.registered_evaluator_sha256() == file_sha256(Path(fresh.__file__))
    assert "embodi-evaluate-exp45-fresh-gripper" in (ROOT / "pyproject.toml").read_text()
