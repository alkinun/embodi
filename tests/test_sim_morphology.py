from pathlib import Path

import numpy as np
import pytest

from embodi.sim.assets import _mesh_files
from embodi.sim.benchmark import BenchmarkDefinition
from embodi.sim.morphology import PANDA, SO101, morphology_by_id


DEFINITION_PATH = Path(__file__).parents[1] / "benchmarks" / "sim-v1" / "definition.json"


def test_mesh_inventory_supports_stl_and_obj() -> None:
    model = '<asset><mesh file="arm.stl"/><mesh file="hand.obj"/><mesh file="arm.stl"/></asset>'
    assert _mesh_files(model) == ("arm.stl", "hand.obj")


def test_morphology_contracts_are_distinct_and_finite() -> None:
    assert SO101.native_state_dim == SO101.native_action_dim == 6
    assert PANDA.native_state_dim == PANDA.native_action_dim == 8
    assert len(SO101.arm_joint_names) == 5
    assert len(PANDA.arm_joint_names) == 7
    assert len(PANDA.sim_joint_names) == 9
    assert np.isfinite(SO101.home_native_state).all()
    assert np.isfinite(PANDA.home_native_state).all()
    assert morphology_by_id("panda") is PANDA
    with pytest.raises(ValueError, match="unknown morphology"):
        morphology_by_id("missing")


def test_morphology_contracts_match_frozen_benchmark() -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    frozen = {value["id"]: value for value in definition.values["morphologies"]}
    for morphology in (SO101, PANDA):
        expected = frozen[morphology.id]
        assert expected["asset"] == f"mujoco_menagerie/{morphology.menagerie_directory}"
        assert expected["asset_revision"] == morphology.asset_revision
        assert expected["native_state_dim"] == morphology.native_state_dim
        assert expected["native_action_dim"] == morphology.native_action_dim
