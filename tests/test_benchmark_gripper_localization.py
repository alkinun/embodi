from decimal import Decimal
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

import embodi.sim.benchmark_gripper_localization as localization
from embodi.sim.benchmark import file_sha256


def _anchor(
    *,
    morphology: str = "so101",
    current: float = 16.0,
    target: float = 16.0,
    phase: str = "precontact",
    post_phase: str | None = None,
) -> dict[str, object]:
    width = 6 if morphology == "so101" else 8
    current_state = np.zeros(width, dtype=np.float32)
    current_state[-1] = current
    target_state = current_state.copy()
    target_state[-1] = target
    return {
        "pre_call_native_state": current_state.tolist(),
        "expert_native": target_state.tolist(),
        "phase": phase,
        "post_action_phase": post_phase or phase,
    }


@pytest.mark.parametrize(
    ("anchor", "morphology", "expected"),
    [
        (_anchor(), "so101", "open_steady"),
        (_anchor(current=0.0, phase="release"), "so101", "open_openward"),
        (_anchor(current=16.0, target=0.0, phase="close"), "so101", "closed_closeward"),
        (_anchor(current=0.0, target=0.0, phase="lift"), "so101", "closed_steady"),
        (
            _anchor(current=6.5, target=6.5, phase="retreat", post_phase="done"),
            "so101",
            "terminal_noop",
        ),
        (
            _anchor(morphology="panda", current=0.08, target=0.08),
            "panda",
            "open_steady",
        ),
    ],
)
def test_partition_labels_registered_setpoints(
    anchor: dict[str, object], morphology: str, expected: str
) -> None:
    assert localization.partition_label(anchor, morphology) == expected


def test_partition_labels_reject_unregistered_or_non_noop_targets() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        localization.partition_label(_anchor(target=8.0), "so101")
    with pytest.raises(ValueError, match="exact native no-op"):
        localization.partition_label(
            _anchor(current=6.5, target=6.0, phase="retreat", post_phase="done"),
            "so101",
        )


def _record(
    *,
    seed: int,
    slice_name: str,
    morphology: str,
    condition: str,
    error: float,
    partition: str = "open_steady",
    endpoint: str = "first",
    scenario: int = 0,
) -> dict[str, object]:
    return {
        "seed": seed,
        "slice": slice_name,
        "morphology": morphology,
        "task": "pick_place_bin",
        "scenario_id": f"{slice_name}-{morphology}-{scenario}",
        "phase": "release" if partition == "open_openward" else "precontact",
        "endpoint": endpoint,
        "condition": condition,
        "partition": partition,
        "metrics": {
            "effective_gripper_error": error,
            "raw_gripper_absolute_error": error,
            "signed_gripper_bias": error,
            "gripper_saturation_rate": 0.0,
            "normalized_channel_9_loss": error * error,
        },
    }


def _cell(full: float, half: float | None, joint: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed in localization.SEEDS:
        for slice_name in localization.SLICES:
            for morphology in localization.MORPHOLOGIES:
                for scenario in range(7):
                    for endpoint in ("first", "last"):
                        records.append(
                            _record(
                                seed=seed,
                                slice_name=slice_name,
                                morphology=morphology,
                                condition="full",
                                error=full,
                                endpoint=endpoint,
                                scenario=scenario,
                            )
                        )
                        if half is not None:
                            records.append(
                                _record(
                                    seed=seed,
                                    slice_name=slice_name,
                                    morphology=morphology,
                                    condition="half",
                                    error=half,
                                    endpoint=endpoint,
                                    scenario=scenario,
                                )
                            )
                        records.append(
                            _record(
                                seed=seed,
                                slice_name=slice_name,
                                morphology=morphology,
                                condition="joint",
                                error=joint,
                                endpoint=endpoint,
                                scenario=scenario,
                            )
                        )
    return records


def test_three_condition_contrasts_decompose_exactly() -> None:
    metrics = localization.condition_contrasts(
        _cell(0.10, 0.12, 0.15), ("full", "half", "joint")
    )
    contrasts = metrics["effective_gripper_error"]["contrasts"]
    assert Decimal(contrasts["J-F"]["difference_decimal"]) == Decimal(
        contrasts["H-F"]["difference_decimal"]
    ) + Decimal(contrasts["J-H"]["difference_decimal"])


def test_localization_requires_material_winner_and_replicated_interaction() -> None:
    left = _cell(0.10, None, 0.14)
    right = _cell(0.10, None, 0.11)
    contrast = localization.localization_contrast(
        left,
        right,
        left_label="openward",
        right_label="open_steady",
        expected_tasks={"pick_place_bin"},
    )
    assert contrast["overall"]["difference"] == pytest.approx(0.03)
    gate = localization._localization_gate("opening_transition", contrast)
    assert gate["localized"] is True
    assert gate["broad"] is False
    assert gate["classification"] == "pick-place-specific-openward-localization"


def test_material_gate_requires_difference_ratio_and_two_seeds() -> None:
    supported = localization._material_gate(
        localization._view(_cell(0.10, None, 0.12), ("full", "joint"))
    )
    assert supported["supported"] is True
    too_small = localization._material_gate(
        localization._view(_cell(0.10, None, 0.109), ("full", "joint"))
    )
    assert too_small["supported"] is False


def test_material_gate_includes_exact_boundaries() -> None:
    boundary = localization._material_gate(
        localization._view(_cell(0.10, None, 0.11), ("full", "joint"))
    )
    assert boundary["supported"] is True


def test_zero_interactions_do_not_count_as_directional_replication() -> None:
    contrast = localization.localization_contrast(
        _cell(0.10, None, 0.12),
        _cell(0.10, None, 0.12),
        left_label="closeward",
        right_label="closed_steady",
        expected_tasks={"pick_place_bin"},
    )
    gate = localization._localization_gate("closing_transition", contrast)
    assert gate["localized"] is False


def test_localization_rejects_incomplete_common_support() -> None:
    left = _cell(0.10, None, 0.14)
    left = [record for record in left if record["morphology"] != "panda"]
    with pytest.raises(ValueError, match="common support"):
        localization.localization_contrast(
            left,
            _cell(0.10, None, 0.11),
            left_label="openward",
            right_label="open_steady",
            expected_tasks={"pick_place_bin"},
        )


def test_localization_rejects_different_scenario_support() -> None:
    left = _cell(0.10, None, 0.14)
    left = [
        record
        for record in left
        if not (
            record["slice"] == "exp42"
            and record["morphology"] == "so101"
            and record["scenario_id"] == "exp42-so101-6"
        )
    ]
    with pytest.raises(ValueError, match="common support"):
        localization.localization_contrast(
            left,
            _cell(0.10, None, 0.11),
            left_label="openward",
            right_label="open_steady",
            expected_tasks={"pick_place_bin"},
        )


def test_no_material_partition_precedes_composite_localization() -> None:
    low_view = localization._view(_cell(0.10, None, 0.105), ("full", "joint"))
    partitions = {
        name: {"pooled_jf": low_view} for name in localization.PARTITIONS
    }
    localized = localization.localization_contrast(
        _cell(0.10, None, 0.14),
        _cell(0.10, None, 0.11),
        left_label="openward",
        right_label="open_steady",
        expected_tasks={"pick_place_bin"},
    )
    result = localization.classify_localization(
        partitions, {"opening_transition": localized}, delivery=True
    )
    assert result["classification"] == "not_materially_localized"


def test_deterministic_archive_checks_stored_and_source_hashes(tmp_path: Path) -> None:
    raw = b'{"registered":true}\n'
    stored = gzip.compress(raw, mtime=0)
    path = tmp_path / "source.json.gz"
    path.write_bytes(stored)
    assert localization._source_bytes(
        path,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(stored).hexdigest(),
    ) == raw
    with pytest.raises(ValueError, match="archive hash mismatch"):
        localization._source_bytes(path, hashlib.sha256(raw).hexdigest(), "0" * 64)


def test_registered_support_and_cli_contract() -> None:
    assert sum(value[0] for value in localization.EXPECTED_SUPPORT.values()) == 364
    assert sum(value[1] for value in localization.EXPECTED_SUPPORT.values()) == 210
    args = localization.parse_args([])
    assert args.output == localization.SUMMARY_PATH
    assert args.exp43_so101.suffix == ".gz"
    assert localization.registered_evaluator_sha256() == file_sha256(
        Path(localization.__file__)
    )
