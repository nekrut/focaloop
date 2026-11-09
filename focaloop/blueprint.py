"""Typed `.focaloop.yml` blueprint model and strict validator.

A blueprint declares *what* a focal loop computes (modality, kernel pool,
conditioner, stop gate, traps, outputs); a run spec declares *which* data.
`validate_blueprint` returns every violation at once so authors fix a file in
one pass; `load_blueprint` raises `BlueprintError` carrying that list.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MODALITIES = ("expression", "variants", "metagenomics", "chromatin", "proteomics", "query", "bait")
TRAP_CODES = ("LOW_MAP_RATE", "TIE_SYMMETRY_FAIL", "BYTE_CAP_UNCONVERGED", "CONFOUNDED_DESIGN")
TRAP_ACTIONS = ("abort", "warn")
KNOWN_KERNELS = (
    "kmer_eq",        # builtin tie-retaining canonical k-mer equivalence-class kernel
    "kmer_besthit",   # builtin best-hit kernel (breaks ties; tournament baseline)
    "kmer_bait",      # builtin Stage-0 bait screen
    "bwa-mem2",
    "strobealign",
    "minimap2",
    "bowtie2",
    "salmon",
    "piscem",
)
OUTPUT_KINDS = ("table", "telemetry", "state", "verdict", "bed")
CONDITIONER_METHODS = ("em", "none")

TOP_LEVEL_KEYS = {
    "id", "name", "version", "modality", "description", "container", "regime_rules",
    "kernel_pool", "conditioner", "stop_gate", "traps", "outputs", "variants", "citations",
}
REQUIRED_KEYS = ("id", "name", "version", "modality", "container", "regime_rules",
                 "kernel_pool", "conditioner", "stop_gate", "traps", "outputs")
ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+([+-][0-9A-Za-z.]+)?$")
CONTAINER_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._+-]+)?(@sha256:[0-9a-f]{64})?$")


class BlueprintError(ValueError):
    def __init__(self, errors: list[str], source: str = "<blueprint>") -> None:
        self.errors = list(errors)
        super().__init__(f"{source}: " + "; ".join(errors))


@dataclass(frozen=True)
class RegimeRule:
    name: str
    when: str
    set: dict[str, Any]


@dataclass(frozen=True)
class TournamentConfig:
    real_reads: int = 25_000
    synthetic_per_regime: int = 600
    n_regimes: int = 5
    tie_symmetry_min: float = 0.80
    k: int = 31
    seed: int = 42


@dataclass(frozen=True)
class KernelPool:
    kernels: tuple[str, ...]
    tournament: TournamentConfig


@dataclass(frozen=True)
class ConditionerConfig:
    method: str = "em"
    max_iter: int = 200
    tol: float = 1e-6
    dirichlet_alpha: float = 0.5
    external_priors: bool = False


@dataclass(frozen=True)
class StopGateConfig:
    block_bytes: int = 65_536
    max_bytes: int = 64 * 1024 * 1024
    holdout_fraction: float = 0.10
    min_posterior_precision: float = 25.0
    max_tech_var_fraction: float = 0.25
    delta_tol: float = 0.02
    min_iterations: int = 2


@dataclass(frozen=True)
class Trap:
    code: str
    action: str
    threshold: float | None


@dataclass(frozen=True)
class OutputDecl:
    name: str
    kind: str
    format: str


@dataclass(frozen=True)
class Blueprint:
    id: str
    name: str
    version: str
    modality: str
    container: str
    regime_rules: tuple[RegimeRule, ...]
    kernel_pool: KernelPool
    conditioner: ConditionerConfig
    stop_gate: StopGateConfig
    traps: dict[str, Trap]
    outputs: tuple[OutputDecl, ...]
    description: str = ""
    variants: dict[str, Any] = field(default_factory=dict)
    citations: tuple[str, ...] = ()
    source_path: str | None = None

    def trap(self, code: str) -> Trap:
        return self.traps[code]


def _num(errors: list[str], where: str, val: Any, lo: float | None = None, hi: float | None = None,
         integer: bool = False) -> None:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or (integer and not isinstance(val, int)):
        errors.append(f"{where}: expected {'integer' if integer else 'number'}, got {val!r}")
        return
    if lo is not None and val < lo:
        errors.append(f"{where}: {val} < minimum {lo}")
    if hi is not None and val > hi:
        errors.append(f"{where}: {val} > maximum {hi}")


def _check_keys(errors: list[str], where: str, obj: dict, allowed: set[str]) -> None:
    for k in obj:
        if k not in allowed:
            errors.append(f"{where}: unknown key {k!r}")


def validate_blueprint(doc: Any) -> list[str]:
    """Return a list of human-readable violations (empty list == valid)."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["blueprint root must be a mapping"]
    _check_keys(errors, "root", doc, TOP_LEVEL_KEYS)
    for k in REQUIRED_KEYS:
        if k not in doc:
            errors.append(f"root: missing required key {k!r}")
    if errors and any("missing" in e for e in errors):
        # keep going: report as much as possible on present keys
        pass

    bid = doc.get("id")
    if "id" in doc and (not isinstance(bid, str) or not ID_RE.match(bid)):
        errors.append(f"id: {bid!r} must match {ID_RE.pattern}")
    name = doc.get("name")
    if "name" in doc and (not isinstance(name, str) or len(name.strip()) < 5):
        errors.append("name: must be a string of at least 5 characters")
    ver = doc.get("version")
    if "version" in doc and (not isinstance(ver, str) or not VERSION_RE.match(ver)):
        errors.append(f"version: {ver!r} must be a quoted semantic version like \"0.1.0\"")
    modality = doc.get("modality")
    if "modality" in doc and modality not in MODALITIES:
        errors.append(f"modality: {modality!r} not in {list(MODALITIES)}")
    cont = doc.get("container")
    if "container" in doc and (not isinstance(cont, str) or not CONTAINER_RE.match(cont) or ":" not in cont
                               or cont.endswith(":latest")):
        errors.append(f"container: {cont!r} must be a pinned image reference (registry/name:tag, not :latest)")

    rules = doc.get("regime_rules")
    if "regime_rules" in doc:
        if not isinstance(rules, list):
            errors.append("regime_rules: must be a list")
        else:
            seen: set[str] = set()
            for i, r in enumerate(rules):
                w = f"regime_rules[{i}]"
                if not isinstance(r, dict):
                    errors.append(f"{w}: must be a mapping")
                    continue
                _check_keys(errors, w, r, {"name", "when", "set"})
                for req in ("name", "when", "set"):
                    if req not in r:
                        errors.append(f"{w}: missing {req!r}")
                if isinstance(r.get("name"), str):
                    if r["name"] in seen:
                        errors.append(f"{w}: duplicate rule name {r['name']!r}")
                    seen.add(r["name"])
                if "when" in r and not isinstance(r["when"], str):
                    errors.append(f"{w}.when: must be a string predicate")
                if "set" in r and not isinstance(r["set"], dict):
                    errors.append(f"{w}.set: must be a mapping")

    pool = doc.get("kernel_pool")
    if "kernel_pool" in doc:
        if not isinstance(pool, dict):
            errors.append("kernel_pool: must be a mapping with 'kernels' and 'tournament'")
        else:
            _check_keys(errors, "kernel_pool", pool, {"kernels", "tournament"})
            ks = pool.get("kernels")
            if not isinstance(ks, list):
                errors.append("kernel_pool.kernels: must be a list")
            else:
                for k in ks:
                    if k not in KNOWN_KERNELS:
                        errors.append(f"kernel_pool.kernels: unknown kernel {k!r}")
                if len(set(ks)) != len(ks):
                    errors.append("kernel_pool.kernels: duplicate kernels")
                if modality in ("expression", "variants", "metagenomics", "chromatin") and not ks:
                    errors.append(f"kernel_pool.kernels: modality {modality!r} needs at least one kernel")
                if modality == "query" and ks:
                    errors.append("kernel_pool.kernels: query modality reads zero new bytes and must declare no kernels")
                if modality == "bait" and ks != ["kmer_bait"]:
                    errors.append("kernel_pool.kernels: bait modality must declare exactly ['kmer_bait']")
            t = pool.get("tournament", {})
            if not isinstance(t, dict):
                errors.append("kernel_pool.tournament: must be a mapping")
            else:
                _check_keys(errors, "kernel_pool.tournament", t,
                            {"real_reads", "synthetic_per_regime", "n_regimes", "tie_symmetry_min", "k", "seed"})
                if "real_reads" in t:
                    _num(errors, "kernel_pool.tournament.real_reads", t["real_reads"], 0, 10_000_000, integer=True)
                if "synthetic_per_regime" in t:
                    _num(errors, "kernel_pool.tournament.synthetic_per_regime", t["synthetic_per_regime"], 0, 100_000, integer=True)
                if "n_regimes" in t and t["n_regimes"] != 5:
                    errors.append("kernel_pool.tournament.n_regimes: must be 5")
                if "tie_symmetry_min" in t:
                    _num(errors, "kernel_pool.tournament.tie_symmetry_min", t["tie_symmetry_min"], 0.80, 1.0)
                if "k" in t:
                    _num(errors, "kernel_pool.tournament.k", t["k"], 11, 31, integer=True)
                if "seed" in t:
                    _num(errors, "kernel_pool.tournament.seed", t["seed"], 0, None, integer=True)

    cond = doc.get("conditioner")
    if "conditioner" in doc:
        if not isinstance(cond, dict):
            errors.append("conditioner: must be a mapping")
        else:
            _check_keys(errors, "conditioner", cond, {"method", "max_iter", "tol", "dirichlet_alpha", "external_priors"})
            if cond.get("method", "em") not in CONDITIONER_METHODS:
                errors.append(f"conditioner.method: {cond.get('method')!r} not in {list(CONDITIONER_METHODS)}")
            if "max_iter" in cond:
                _num(errors, "conditioner.max_iter", cond["max_iter"], 1, 100_000, integer=True)
            if "tol" in cond:
                _num(errors, "conditioner.tol", cond["tol"], 0.0, 1.0)
            if "dirichlet_alpha" in cond:
                _num(errors, "conditioner.dirichlet_alpha", cond["dirichlet_alpha"], 0.0, 100.0)
            if "external_priors" in cond and not isinstance(cond["external_priors"], bool):
                errors.append("conditioner.external_priors: must be boolean")

    gate = doc.get("stop_gate")
    if "stop_gate" in doc:
        if not isinstance(gate, dict):
            errors.append("stop_gate: must be a mapping")
        else:
            _check_keys(errors, "stop_gate", gate, {"block_bytes", "max_bytes", "holdout_fraction",
                                                   "min_posterior_precision", "max_tech_var_fraction",
                                                   "delta_tol", "min_iterations"})
            if "block_bytes" in gate:
                _num(errors, "stop_gate.block_bytes", gate["block_bytes"], 1024, 1 << 30, integer=True)
            if "max_bytes" in gate:
                _num(errors, "stop_gate.max_bytes", gate["max_bytes"], 0, None, integer=True)
            if "block_bytes" in gate and "max_bytes" in gate and isinstance(gate["max_bytes"], int) \
                    and isinstance(gate["block_bytes"], int) and 0 < gate["max_bytes"] < gate["block_bytes"]:
                errors.append("stop_gate.max_bytes: must be >= block_bytes (or 0 for zero-byte modalities)")
            if "holdout_fraction" in gate:
                _num(errors, "stop_gate.holdout_fraction", gate["holdout_fraction"], 0.0, 0.5)
            if "min_posterior_precision" in gate:
                _num(errors, "stop_gate.min_posterior_precision", gate["min_posterior_precision"], 0.0, None)
            if "max_tech_var_fraction" in gate:
                _num(errors, "stop_gate.max_tech_var_fraction", gate["max_tech_var_fraction"], 0.0, 1.0)
            if "delta_tol" in gate:
                _num(errors, "stop_gate.delta_tol", gate["delta_tol"], 0.0, 10.0)
            if "min_iterations" in gate:
                _num(errors, "stop_gate.min_iterations", gate["min_iterations"], 1, 10_000, integer=True)
            if modality == "query" and gate.get("max_bytes", 0) != 0:
                errors.append("stop_gate.max_bytes: query modality must be 0 (zero new bytes)")

    traps = doc.get("traps")
    if "traps" in doc:
        if not isinstance(traps, dict):
            errors.append("traps: must be a mapping of trap code -> {action, threshold}")
        else:
            for code in traps:
                if code not in TRAP_CODES:
                    errors.append(f"traps: unknown trap code {code!r}")
            for code in TRAP_CODES:
                if code not in traps:
                    errors.append(f"traps: missing required trap {code!r}")
                    continue
                spec = traps[code]
                if not isinstance(spec, dict):
                    errors.append(f"traps.{code}: must be a mapping")
                    continue
                _check_keys(errors, f"traps.{code}", spec, {"action", "threshold"})
                if spec.get("action") not in TRAP_ACTIONS:
                    errors.append(f"traps.{code}.action: must be one of {list(TRAP_ACTIONS)}")
                if spec.get("threshold") is not None:
                    _num(errors, f"traps.{code}.threshold", spec["threshold"], 0.0, 1.0)
            ts = traps.get("TIE_SYMMETRY_FAIL")
            if isinstance(ts, dict) and isinstance(ts.get("threshold"), (int, float)) and ts["threshold"] < 0.80:
                errors.append("traps.TIE_SYMMETRY_FAIL.threshold: mate-tie symmetry gate must be >= 0.80")

    outs = doc.get("outputs")
    if "outputs" in doc:
        if not isinstance(outs, list) or not outs:
            errors.append("outputs: must be a non-empty list")
        else:
            names: set[str] = set()
            for i, o in enumerate(outs):
                w = f"outputs[{i}]"
                if not isinstance(o, dict):
                    errors.append(f"{w}: must be a mapping")
                    continue
                _check_keys(errors, w, o, {"name", "kind", "format"})
                for req in ("name", "kind", "format"):
                    if req not in o:
                        errors.append(f"{w}: missing {req!r}")
                if o.get("kind") is not None and o.get("kind") not in OUTPUT_KINDS:
                    errors.append(f"{w}.kind: {o.get('kind')!r} not in {list(OUTPUT_KINDS)}")
                if o.get("name") in names:
                    errors.append(f"{w}: duplicate output name {o.get('name')!r}")
                names.add(o.get("name"))
            kinds = {o.get("kind") for o in outs if isinstance(o, dict)}
            if "telemetry" not in kinds:
                errors.append("outputs: must declare a 'telemetry' output (traps are reported there)")

    variants = doc.get("variants")
    if variants is not None:
        if modality != "variants":
            errors.append("variants: block only allowed for modality 'variants'")
        elif not isinstance(variants, dict):
            errors.append("variants: must be a mapping")
        else:
            _check_keys(errors, "variants", variants, {"min_alt_reads", "cnv_genes", "deletion_genes",
                                                      "host_decoy", "single_copy_controls", "fws"})
    cites = doc.get("citations")
    if cites is not None and (not isinstance(cites, list) or not all(isinstance(c, str) for c in cites)):
        errors.append("citations: must be a list of DOI strings")
    return errors


def blueprint_from_dict(doc: dict, source: str = "<blueprint>") -> Blueprint:
    errors = validate_blueprint(doc)
    if errors:
        raise BlueprintError(errors, source)
    pool = doc["kernel_pool"]
    t = pool.get("tournament", {}) or {}
    traps = {
        code: Trap(code=code, action=spec["action"], threshold=spec.get("threshold"))
        for code, spec in doc["traps"].items()
    }
    return Blueprint(
        id=doc["id"],
        name=doc["name"],
        version=doc["version"],
        modality=doc["modality"],
        container=doc["container"],
        description=doc.get("description", ""),
        regime_rules=tuple(RegimeRule(r["name"], r["when"], dict(r["set"])) for r in doc["regime_rules"]),
        kernel_pool=KernelPool(tuple(pool.get("kernels", [])), TournamentConfig(**t)),
        conditioner=ConditionerConfig(**doc["conditioner"]),
        stop_gate=StopGateConfig(**doc["stop_gate"]),
        traps=traps,
        outputs=tuple(OutputDecl(o["name"], o["kind"], o["format"]) for o in doc["outputs"]),
        variants=dict(doc.get("variants") or {}),
        citations=tuple(doc.get("citations") or ()),
        source_path=source,
    )


def blueprint_search_path() -> list[Path]:
    env = os.environ.get("FOCALOOP_BLUEPRINT_PATH")
    paths = [Path(p) for p in env.split(os.pathsep) if p] if env else []
    paths.append(Path("/opt/focaloop/blueprints"))
    paths.append(Path(__file__).resolve().parent.parent / "blueprints")
    return paths


def resolve_blueprint_path(ref: str | Path) -> Path:
    """Accept a file path or a builtin blueprint id (looked up on the blueprint search path)."""
    p = Path(ref)
    if p.is_file():
        return p
    if p.is_dir() and (p / "blueprint.focaloop.yml").is_file():
        return p / "blueprint.focaloop.yml"
    if isinstance(ref, str) and ID_RE.match(ref):
        for base in blueprint_search_path():
            cand = base / ref / "blueprint.focaloop.yml"
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"blueprint {ref!r} not found as a file or builtin id")


def load_blueprint(ref: str | Path) -> Blueprint:
    path = resolve_blueprint_path(ref)
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return blueprint_from_dict(doc, source=str(path))


def regime_settings(bp: Blueprint, facts: dict[str, Any]) -> dict[str, Any]:
    """Evaluate `regime_rules` against run facts; later matching rules override earlier ones.

    Predicates are restricted comparisons (`n_assemblies > 1`, `layout == 'PE'`,
    `always`) joined by `and`; anything else is rejected rather than eval'd.
    """
    out: dict[str, Any] = {}
    for rule in bp.regime_rules:
        if _eval_predicate(rule.when, facts):
            out.update(rule.set)
    return out


_CMP_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$")


def _eval_predicate(expr: str, facts: dict[str, Any]) -> bool:
    expr = expr.strip()
    if expr == "always":
        return True
    for clause in expr.split(" and "):
        m = _CMP_RE.match(clause)
        if not m:
            raise ValueError(f"unsupported regime predicate {clause!r}")
        key, op, raw = m.groups()
        lhs = facts.get(key)
        rhs: Any = raw.strip("'\"") if raw[:1] in "'\"" else (float(raw) if re.match(r"^-?\d+(\.\d+)?$", raw) else raw)
        if raw in ("true", "false"):
            rhs = raw == "true"
        if lhs is None:
            return False
        ok = {"==": lhs == rhs, "!=": lhs != rhs}.get(op)
        if ok is None:
            try:
                ok = {">": lhs > rhs, "<": lhs < rhs, ">=": lhs >= rhs, "<=": lhs <= rhs}[op]
            except TypeError:
                return False
        if not ok:
            return False
    return True
