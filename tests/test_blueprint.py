"""`.focaloop.yml` model + strict validator across all four shipped blueprints."""

from __future__ import annotations

import copy

import pytest
import yaml

from focaloop.blueprint import (
    MODALITIES,
    TRAP_CODES,
    BlueprintError,
    load_blueprint,
    regime_settings,
    validate_blueprint,
)

from .conftest import BLUEPRINT_IDS, BLUEPRINTS, run_cli


def _doc(bid: str) -> dict:
    return yaml.safe_load((BLUEPRINTS / bid / "blueprint.focaloop.yml").read_text())


@pytest.mark.parametrize("bid", BLUEPRINT_IDS)
def test_shipped_blueprint_validates_and_loads(bid: str) -> None:
    assert validate_blueprint(_doc(bid)) == []
    bp = load_blueprint(BLUEPRINTS / bid / "blueprint.focaloop.yml")
    assert bp.id == bid and bp.modality in MODALITIES
    assert set(bp.traps) == set(TRAP_CODES)
    assert bp.container == "ghcr.io/galaxyproject/focaloop:0.1.0"


def test_builtin_id_resolution_and_cli_validate(capsys) -> None:
    assert load_blueprint("panexpr").modality == "expression"
    assert run_cli("validate", *[BLUEPRINTS / b / "blueprint.focaloop.yml" for b in BLUEPRINT_IDS]) == 0
    assert capsys.readouterr().out.count("OK ") == 4


def test_validator_collects_every_violation(tmp_path, capsys) -> None:
    doc = _doc("panexpr")
    doc["modality"] = "rnaseq"
    doc["id"] = "Bad-ID"
    doc["container"] = "focaloop:latest"
    del doc["traps"]["LOW_MAP_RATE"]
    doc["traps"]["TIE_SYMMETRY_FAIL"]["threshold"] = 0.5
    doc["kernel_pool"]["kernels"].append("bwa")
    doc["surprise"] = 1
    errs = validate_blueprint(doc)
    joined = "\n".join(errs)
    for needle in ("modality", "id:", "container", "LOW_MAP_RATE", "0.80", "unknown kernel 'bwa'", "unknown key 'surprise'"):
        assert needle in joined, needle
    p = tmp_path / "bad.focaloop.yml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(BlueprintError) as ei:
        load_blueprint(p)
    assert len(ei.value.errors) == len(errs)
    assert run_cli("validate", p) == 2
    assert "INVALID" in capsys.readouterr().out


def test_modality_specific_rules() -> None:
    q = _doc("focaloop_query")
    q["kernel_pool"]["kernels"] = ["kmer_eq"]
    q["stop_gate"]["max_bytes"] = 1024
    errs = "\n".join(validate_blueprint(q))
    assert "zero new bytes" in errs and "query modality must be 0" in errs
    b = copy.deepcopy(_doc("focaloop_bait"))
    b["kernel_pool"]["kernels"] = ["kmer_eq"]
    assert any("bait modality" in e for e in validate_blueprint(b))
    e = _doc("panexpr")
    e["kernel_pool"]["kernels"] = []
    assert any("needs at least one kernel" in x for x in validate_blueprint(e))
    e = _doc("panexpr")
    e["variants"] = {"min_alt_reads": 2}
    assert any("variants: block only allowed" in x for x in validate_blueprint(e))


def test_regime_rules_evaluate_restricted_predicates() -> None:
    bp = load_blueprint("panexpr")
    assert regime_settings(bp, {"n_assemblies": 2, "n_targets": 10}) == {"conditioning": "strain_aware", "k": 31}
    assert regime_settings(bp, {"n_assemblies": 1, "n_targets": 500}) == {}
    mv = load_blueprint("malariavar")
    assert regime_settings(mv, {"has_decoy": True})["decoy_sieve"] is True
