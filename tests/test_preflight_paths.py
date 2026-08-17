import builtins
import json
import sys
from pathlib import Path

import pytest

from embodi.physical import preflight


def test_dependency_report_records_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "deepdiff":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    report = preflight.dependency_report()

    assert report["deepdiff"] is False
    assert report["serial"] is True
    assert report["feetech_sdk"] is True


def test_preflight_main_reports_calibration_without_hardware(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calibration = Path(__file__).parents[1] / "configs" / "so101-physical-calibration.example.json"
    monkeypatch.setattr(sys, "argv", ["embodi-so101-preflight", "--canonical-calibration", str(calibration)])

    preflight.main()

    report = json.loads(capsys.readouterr().out)
    assert report["hardware_connected"] is False
    assert report["canonical_calibration"]["format_version"] == 1


def test_preflight_main_requires_port_and_robot_id_for_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["embodi-so101-preflight", "--connect-hardware"])

    with pytest.raises(ValueError, match="requires --port and --robot-id"):
        preflight.main()
