"""Offline structural checks for the dual Galaxy packaging of each blueprint.

* :func:`lint_user_tool` mirrors the ``UserToolSource`` contract (``class: GalaxyUserTool``,
  ``extra="forbid"``, the four semantic validators) and the udt-authoring checklist
  (help object, no XML-isms, no ``min``/``max`` on data inputs, quoted text interpolation).
  When ``galaxy-tool-util>=26.1`` is importable, :func:`official_udt_errors` runs the
  server's own model as well.
* :func:`lint_tool_xml` checks ToolShed XML element order and required sections.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

UDT_TOP = {"class", "id", "version", "name", "description", "container", "shell_command", "inputs", "outputs",
           "requirements", "configfiles", "citations", "license", "help", "tests", "profile", "edam_operations",
           "edam_topics", "xrefs"}
UDT_REQUIRED = ("class", "name", "version", "container", "shell_command")
INPUT_TYPES = {"boolean", "integer", "float", "text", "select", "color", "data", "data_collection",
               "conditional", "repeat", "section"}
INPUT_COMMON = {"name", "label", "help", "optional", "type"}
INPUT_EXTRA = {
    "boolean": {"value"},
    "integer": {"value", "min", "max", "validators"},
    "float": {"value", "min", "max", "validators"},
    "text": {"value", "area", "validators"},
    "select": {"options", "multiple", "validators"},
    "color": {"value"},
    "data": {"format", "multiple"},
    "data_collection": {"collection_type", "format"},
    "conditional": {"test_parameter", "whens"},
    "repeat": {"parameters", "min", "max"},
    "section": {"parameters"},
}
FORBIDDEN_FIELDS = {"truevalue", "falsevalue", "argument", "is_dynamic", "hidden", "parameter_type", "command"}
OUTPUT_FIELDS = {"name", "type", "format", "format_source", "metadata_source", "label", "from_work_dir",
                 "discover_datasets", "collection_type", "hidden"}
ID_RE = re.compile(r"^[a-z][a-z0-9_-]{2,254}$")
REF_RE = re.compile(r"inputs\.([A-Za-z_][A-Za-z0-9_]*)")
EXPR_RE = re.compile(r"(?<!\\)\$\((.*?)\)(?=[\s'\"]|$|\\)", re.S)

XML_ORDER = ["description", "macros", "edam_topics", "edam_operations", "xrefs", "requirements", "version_command",
             "command", "environment_variables", "configfiles", "inputs", "outputs", "tests", "help", "citations"]
XML_REQUIRED = ["description", "requirements", "command", "inputs", "outputs", "tests", "help", "citations"]


def _walk_inputs(inputs: list[dict], errors: list[str], where: str = "inputs") -> list[str]:
    names: list[str] = []
    for i, p in enumerate(inputs or []):
        w = f"{where}[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{w}: must be a mapping")
            continue
        t = p.get("type")
        if "name" not in p:
            errors.append(f"{w}: missing name")
        if t not in INPUT_TYPES:
            errors.append(f"{w}: unsupported input type {t!r}")
            continue
        for k in p:
            if k in FORBIDDEN_FIELDS:
                errors.append(f"{w}: forbidden XML-ism field {k!r}")
            elif k not in INPUT_COMMON | INPUT_EXTRA[t]:
                errors.append(f"{w}: field {k!r} not allowed on {t} input")
        if t == "select":
            opts = p.get("options") or []
            if not opts:
                errors.append(f"{w}: select needs static options")
            for o in opts:
                if not isinstance(o, dict) or set(o) - {"label", "value", "selected"} or "value" not in o:
                    errors.append(f"{w}: select option must be {{label, value, selected}}")
        for v in p.get("validators", []) or []:
            if not isinstance(v, dict) or v.get("type") not in ("regex", "length", "in_range", "empty_field", "no_options"):
                errors.append(f"{w}: bad validator {v!r}")
            elif v["type"] == "regex" and "expression" not in v:
                errors.append(f"{w}: regex validator needs 'expression'")
        if t in ("repeat", "section"):
            _walk_inputs(p.get("parameters", []), errors, f"{w}.parameters")
        names.append(p.get("name"))
    return names


def lint_user_tool(doc: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["root must be a mapping"]
    for k in doc:
        if k not in UDT_TOP:
            errors.append(f"unknown top-level field {k!r} (schema is extra='forbid')")
    for k in UDT_REQUIRED:
        if k not in doc:
            errors.append(f"missing required field {k!r}")
    if doc.get("class") != "GalaxyUserTool":
        errors.append("class must be exactly 'GalaxyUserTool'")
    if "id" in doc and not (isinstance(doc["id"], str) and ID_RE.match(doc["id"])):
        errors.append(f"id {doc.get('id')!r} must match ^[a-z][a-z0-9_-]*$ (3-255 chars)")
    if not isinstance(doc.get("version"), str) or not doc.get("version", "").strip():
        errors.append("version must be a non-blank quoted string")
    if not isinstance(doc.get("name"), str) or len(doc.get("name", "").strip()) < 5:
        errors.append("name must be at least 5 characters")
    cont = doc.get("container")
    if not isinstance(cont, str) or not cont.strip():
        errors.append("container must be a non-blank string (dynamic_tool.blank_container)")
    elif cont.startswith(("docker://", "oras://", "library://")):
        errors.append("container must be a bare image reference; explicit_singularity adds docker:// itself")
    elif cont.endswith(":latest") or ":" not in cont:
        errors.append("container must be pinned to a tag")
    cmd = doc.get("shell_command")
    if not isinstance(cmd, str) or not cmd.strip():
        errors.append("shell_command must be a non-blank string")
        cmd = ""
    names = _walk_inputs(doc.get("inputs", []), errors)
    if len(set(names)) != len(names):
        errors.append("duplicate input names")
    texts = [cmd] + [c.get("content", "") for c in doc.get("configfiles", []) or [] if isinstance(c, dict)]
    for txt in texts:
        for ref in REF_RE.findall(txt):
            if ref not in names:
                errors.append(f"dynamic_tool.undeclared_input_ref: inputs.{ref}")
    for m in re.finditer(r"(?<!\\)\$\((?!\()([^)]*)", cmd):
        body = m.group(1).strip()
        if not re.match(r"^(inputs\b|JSON\.|[A-Za-z_][A-Za-z0-9_]*\()", body) or re.match(r"^(pwd|date|cat|echo|ls|basename|dirname|nproc|which)\b", body):
            errors.append(f"bash command substitution $({body[:20]}...) in shell_command is parsed as a CWL JS "
                          "expression (ReferenceError); use $PWD, shell variables or backticks")
    if "$(outputs." in cmd:
        errors.append("$(outputs.X) is not available in UDT templating")
    # Cheetah params are lowercase ($threads); upper-case shell variables ($PWD, $GALAXY_SLOTS) are fine
    if any(v != v.upper() for v in re.findall(r"(?<![\\(])\$([A-Za-z_][A-Za-z0-9_]*)\b", cmd)):
        errors.append("Cheetah-style $param reference in shell_command (use $(inputs.x))")
    if "#for" in cmd or "#if" in cmd:
        errors.append("Cheetah directives are not supported in shell_command")
    in_types = {p.get("name"): p.get("type") for p in doc.get("inputs", []) or [] if isinstance(p, dict)}
    for m in re.finditer(r"\$\(inputs\.([A-Za-z0-9_]+)\)", cmd):
        if in_types.get(m.group(1)) != "text":
            continue
        line_start = cmd.rfind("\n", 0, m.start()) + 1
        prefix = cmd[line_start:m.start()]
        if prefix.count("'") % 2 == 0 and prefix.count('"') % 2 == 0:
            errors.append(f"text input {m.group(1)!r} interpolated unquoted into shell_command")
    for c in doc.get("configfiles", []) or []:
        if not isinstance(c, dict) or "filename" not in c or "content" not in c:
            errors.append("configfiles entries need filename and content")
    for i, o in enumerate(doc.get("outputs", []) or []):
        w = f"outputs[{i}]"
        if not isinstance(o, dict):
            errors.append(f"{w}: must be a mapping")
            continue
        for k in o:
            if k not in OUTPUT_FIELDS:
                errors.append(f"{w}: unknown field {k!r}")
        if o.get("type") not in ("data", "collection"):
            errors.append(f"{w}: only data/collection outputs are runnable")
        if o.get("type") == "data" and not (o.get("from_work_dir") or o.get("discover_datasets")):
            errors.append(f"{w}: dynamic_tool.output_unclaimed")
        if o.get("type") == "collection" and not o.get("discover_datasets"):
            errors.append(f"{w}: dynamic_tool.output_unclaimed")
        fwd = o.get("from_work_dir")
        if fwd and (fwd.startswith("/") or ".." in fwd):
            errors.append(f"{w}: from_work_dir must be a relative path inside the job directory")
    h = doc.get("help")
    if not isinstance(h, dict) or h.get("format") not in ("markdown", "restructuredtext", "plain_text") \
            or not str(h.get("content", "")).strip() or "TODO" in str(h.get("content", "")):
        errors.append("help must be {format, content} with real content")
    for c in doc.get("citations", []) or []:
        if not isinstance(c, dict) or c.get("type") not in ("doi", "bibtex", "reference") or not c.get("content"):
            errors.append(f"bad citation {c!r}")
    return errors


def load_and_lint_user_tool(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return lint_user_tool(yaml.safe_load(fh))


def official_udt_errors(path: str | Path) -> list[str] | None:
    """Validate with galaxy-tool-util's UserToolSource + lint; ``None`` when not installed."""
    try:
        from galaxy.tool_util.lint import lint_user_tool_source  # type: ignore
        from galaxy.tool_util_models import UserToolSource  # type: ignore
    except ImportError:
        return None
    from pydantic import ValidationError

    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    try:
        model = UserToolSource(**doc)
    except ValidationError as exc:
        return [str(exc)]
    return [str(m) for m in lint_user_tool_source(model)]


def lint_tool_xml(path: str | Path) -> list[str]:
    errors: list[str] = []
    root = ET.parse(path).getroot()
    if root.tag != "tool":
        return ["root element must be <tool>"]
    for attr in ("id", "name", "version", "profile"):
        if not root.get(attr):
            errors.append(f"<tool> missing {attr}")
    if root.get("id") and not re.match(r"^[a-z0-9_]+$", root.get("id", "")):
        errors.append("tool id must use [a-z0-9_]")
    tags = [c.tag for c in root]
    for req in XML_REQUIRED:
        if req not in tags:
            errors.append(f"missing <{req}>")
    ordered = [t for t in tags if t in XML_ORDER]
    if ordered != sorted(ordered, key=XML_ORDER.index):
        errors.append(f"element order {ordered} violates {XML_ORDER}")
    reqs = root.find("requirements")
    if reqs is not None and reqs.find("requirement") is None and reqs.find("container") is None:
        errors.append("<requirements> must declare a requirement or container")
    cmd = root.find("command")
    if cmd is not None and cmd.get("detect_errors") not in ("exit_code", "aggressive"):
        errors.append("<command> needs detect_errors")
    inputs = root.find("inputs")
    params = {p.get("name") or (p.get("argument") or "").lstrip("-").replace("-", "_")
              for p in (inputs.iter("param") if inputs is not None else [])}
    outputs = root.find("outputs")
    out_names = {d.get("name") for d in (outputs if outputs is not None else [])}
    cmd_text = cmd.text if cmd is not None and cmd.text else ""
    for p in params:
        if not re.search(r"\$\{?" + re.escape(p) + r"\b", cmd_text):
            errors.append(f"param {p!r} not used in <command>")
    for o in out_names:
        if not re.search(r"\$\{?" + re.escape(o) + r"\b", cmd_text) and \
                not (outputs.find(f"data[@name='{o}']") is not None and outputs.find(f"data[@name='{o}']").get("from_work_dir")):
            errors.append(f"output {o!r} not written by <command>")
    tests = root.find("tests")
    if tests is None or not list(tests):
        errors.append("<tests> must contain at least one <test>")
    else:
        for t in tests:
            if t.get("expect_num_outputs") is None:
                errors.append("<test> missing expect_num_outputs")
    help_el = root.find("help")
    if help_el is None or not (help_el.text or "").strip():
        errors.append("<help> must not be empty")
    cites = root.find("citations")
    if cites is None or cites.find("citation") is None:
        errors.append("<citations> must contain a citation")
    return errors


def xml_test_data_refs(path: str | Path) -> list[str]:
    """Every test-data file referenced by <param value=...> / <output file=...> in <tests>."""
    root = ET.parse(path).getroot()
    params = {p.get("name"): p for p in root.find("inputs").iter("param")}
    refs: list[str] = []
    for t in root.find("tests"):
        for p in t.iter("param"):
            decl = params.get(p.get("name"))
            if decl is not None and decl.get("type") == "data":
                refs += [v for v in p.get("value", "").split(",") if v]
        for o in t.iter("output"):
            if o.get("file"):
                refs.append(o.get("file"))
    return refs
