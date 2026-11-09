"""Dual Galaxy packaging: strict GalaxyUserTool YAML + ToolShed XML for every blueprint."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
import yaml

from focaloop.galaxy_lint import lint_tool_xml, lint_user_tool, load_and_lint_user_tool, official_udt_errors, xml_test_data_refs

from .conftest import BLUEPRINT_IDS, BLUEPRINTS


@pytest.mark.parametrize("bid", BLUEPRINT_IDS)
def test_udt_is_strict_galaxy_user_tool(bid: str) -> None:
    path = BLUEPRINTS / bid / "galaxy_udt.yml"
    doc = yaml.safe_load(path.read_text())
    assert doc["class"] == "GalaxyUserTool"
    assert isinstance(doc["version"], str)
    assert doc["container"] == "ghcr.io/galaxyproject/focaloop:0.1.0"
    assert load_and_lint_user_tool(path) == []
    for o in doc["outputs"]:
        assert o.get("from_work_dir") or o.get("discover_datasets")
    # every multi-line command is one shell command (line continuations), not N separate commands
    lines = [ln for ln in doc["shell_command"].strip().splitlines()]
    assert all(ln.rstrip().endswith("\\") for ln in lines[:-1])


@pytest.mark.parametrize("bid", BLUEPRINT_IDS)
def test_udt_official_schema_when_available(bid: str) -> None:
    errs = official_udt_errors(BLUEPRINTS / bid / "galaxy_udt.yml")
    if errs is None:
        pytest.skip("galaxy-tool-util>=26.1 not installed")
    assert errs == []


def test_udt_linter_rejects_xml_habits() -> None:
    bad = {
        "class": "GalaxyUserTool", "id": "Bad Tool", "version": 1, "name": "abc", "container": "",
        "command": "x", "shell_command": "tool $threads '$(inputs.missing.path)' --label $(inputs.label)",
        "inputs": [{"name": "flag", "type": "boolean", "truevalue": "--x"},
                   {"name": "reads", "type": "data", "min": 1},
                   {"name": "label", "type": "text"}],
        "outputs": [{"name": "out", "type": "data", "format": "txt"}],
        "help": "bare string",
    }
    errs = "\n".join(lint_user_tool(bad))
    for needle in ("unknown top-level field 'command'", "id 'Bad Tool'", "version", "name must be",
                   "blank_container", "truevalue", "'min' not allowed on data", "undeclared_input_ref: inputs.missing",
                   "output_unclaimed", "help must be", "Cheetah-style", "interpolated unquoted"):
        assert needle in errs, needle


@pytest.mark.parametrize("bid", BLUEPRINT_IDS)
def test_tool_xml_structure_and_test_data(bid: str) -> None:
    path = BLUEPRINTS / bid / "galaxy_tool.xml"
    assert lint_tool_xml(path) == []
    root = ET.parse(path).getroot()
    assert root.find("command").get("detect_errors") == "exit_code"
    reqs = root.find("requirements")
    assert reqs.find("requirement").text == "focaloop"
    assert reqs.find("container").text.startswith("ghcr.io/galaxyproject/focaloop:")
    refs = xml_test_data_refs(path)
    assert refs
    for ref in refs:
        f = BLUEPRINTS / bid / "test-data" / ref
        assert f.is_file(), f
        assert f.stat().st_size < 1_000_000


def test_xml_linter_flags_order_and_orphans(tmp_path) -> None:
    p = tmp_path / "t.xml"
    p.write_text("""<tool id="x_y" name="X" version="1" profile="25.0"><description>d</description>
        <command detect_errors="exit_code">echo '$a' > '$out'</command>
        <requirements><requirement type="package">x</requirement></requirements>
        <inputs><param name="a" type="text"/><param name="b" type="text"/></inputs>
        <outputs><data name="out" format="txt"/></outputs><tests><test><param name="a" value="1"/></test></tests>
        <help>h</help><citations><citation type="doi">10.1/x</citation></citations></tool>""")
    errs = "\n".join(lint_tool_xml(p))
    assert "element order" in errs and "'b' not used" in errs and "expect_num_outputs" in errs


@pytest.mark.parametrize("bid", BLUEPRINT_IDS)
def test_blueprint_dir_is_complete(bid: str) -> None:
    d = BLUEPRINTS / bid
    for name in ("blueprint.focaloop.yml", "galaxy_udt.yml", "galaxy_tool.xml"):
        assert (d / name).is_file()
    assert any((d / "test-data").iterdir())


def test_udt_linter_catches_live_galaxy_gotchas() -> None:
    base = yaml.safe_load((BLUEPRINTS / "panexpr" / "galaxy_udt.yml").read_text())
    assert lint_user_tool(base) == []
    bad = dict(base, container="docker://ghcr.io/galaxyproject/focaloop:0.1.0")
    assert any("explicit_singularity adds docker://" in e for e in lint_user_tool(bad))
    for sub in ("$(pwd)", "$(date +%s)", "$(nproc)"):
        bad = dict(base, shell_command="echo " + sub + " > log.txt && " + base["shell_command"])
        assert any("bash command substitution" in e for e in lint_user_tool(bad)), sub
    ok = dict(base, shell_command="echo \\$(date) `pwd` $PWD > log.txt && " + base["shell_command"])
    assert lint_user_tool(ok) == []
