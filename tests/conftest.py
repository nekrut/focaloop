from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from focaloop.cli import main

ROOT = Path(__file__).resolve().parent.parent
BLUEPRINTS = ROOT / "blueprints"
BLUEPRINT_IDS = ("panexpr", "malariavar", "focaloop_query", "focaloop_bait")


def run_cli(*args: str) -> int:
    return main([str(a) for a in args])


@pytest.fixture(scope="session")
def panexpr_run(tmp_path_factory) -> dict:
    """One full `focaloop run` on the panexpr fixtures, shared across tests."""
    d = tmp_path_factory.mktemp("panexpr_run")
    td = BLUEPRINTS / "panexpr" / "test-data"
    rc = run_cli("run", "--blueprint", BLUEPRINTS / "panexpr" / "blueprint.focaloop.yml", "--spec", td / "spec.yml",
                 "--state-dir", d / "state", "--state-archive", d / "state.tar.zst",
                 "--out-table", d / "de.tsv", "--out-telemetry", d / "telemetry.json")
    return {"rc": rc, "dir": d, "telemetry": json.loads((d / "telemetry.json").read_text())}


@pytest.fixture()
def panexpr_data(tmp_path: Path) -> Path:
    dst = tmp_path / "data"
    shutil.copytree(BLUEPRINTS / "panexpr" / "test-data", dst)
    return dst
