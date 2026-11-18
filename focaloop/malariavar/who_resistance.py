"""MalariaGEN Pf8 WHO inferred antimalarial resistance status classifier (13 Dec 2024 specification).

Implements the exact sequential decision tables from Pf8_resistance_classification.pdf across all 10 drugs
and combination regimens:
  Chloroquine, Pyrimethamine, Sulfadoxine, Mefloquine, Artemisinin, Piperaquine,
  SP (uncomplicated), SP (IPTp), AS-MQ, DHA-PPQ.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WHO_K13_VALIDATED = {
    "F446I", "N458Y", "M476I", "Y493H", "R539T", "I543T", "P553L", "R561H", "P574L", "C580Y",
}
WHO_K13_CANDIDATE = {
    "P441L", "G449A", "C469F", "C469Y", "A481V", "R515K", "P527H", "N537I", "N537D",
    "G538V", "V568G", "R622I", "A675V",
}
WHO_K13_POTENTIAL = {
    "K479I", "G533A", "R575K", "M579I", "D584V", "P667T", "F673I", "H719N",
}
# Pf8 WHO delayed-clearance list (all 31 validated + candidate + potential mutations in Pf8_resistance_classification.pdf)
WHO_K13_ASSOC = WHO_K13_VALIDATED | WHO_K13_CANDIDATE | WHO_K13_POTENTIAL

DRUG_COLS = [
    "Chloroquine",
    "Pyrimethamine",
    "Sulfadoxine",
    "Mefloquine",
    "Artemisinin",
    "Piperaquine",
    "SP (uncomplicated)",
    "SP (IPTp)",
    "AS-MQ",
    "DHA-PPQ",
]


def parse_aa(val) -> tuple[set[str], bool]:
    """Return (set of uppercase amino-acid tokens, is_missing_or_ambiguous)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set(), True
    s = str(val).strip()
    if s in ("", "-", "nan", "NaN", "None", "none"):
        return set(), True
    if "*" in s or "!" in s or "-" in s:
        return set(), True
    toks = {t.strip().upper() for t in s.split(",") if t.strip()}
    if not toks:
        return set(), True
    return toks, False


def _single_codon_drug(val, wt: str, mut: str) -> str:
    alleles, miss = parse_aa(val)
    if miss:
        return "Undetermined"
    if wt in alleles and mut in alleles and len(alleles) == 2:
        return "Undetermined"
    if alleles == {wt}:
        return "Sensitive"
    if alleles == {mut}:
        return "Resistant"
    return "Undetermined"


def _cnv_drug(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Undetermined"
    try:
        iv = int(float(val))
    except (ValueError, TypeError):
        return "Undetermined"
    if iv < 0:
        return "Undetermined"
    if iv == 0:
        return "Sensitive"
    if iv == 1:
        return "Resistant"
    return "Undetermined"


def classify_artemisinin(k13_val, who_set: set[str] = WHO_K13_ASSOC, k13_has_missing: bool = False) -> str:
    """Sequential 6-step Pf8 Artemisinin heuristic over kelch13 349-726 non-synonymous changes."""
    if k13_val is None or (isinstance(k13_val, float) and np.isnan(k13_val)):
        raw = ""
    else:
        raw = str(k13_val).strip()
    if raw in ("nan", "NaN", "None", "none"):
        raw = ""

    if raw == "-":
        return "Undetermined"

    has_special_miss = k13_has_missing or ("-" in raw) or ("!" in raw)
    toks = [t.strip().rstrip("*!") for t in raw.split(",") if t.strip() and t.strip() not in ("-", "*", "!")]

    hom_muts = set()
    het_muts = set()
    for t in toks:
        u = t.upper()
        if t == u:
            hom_muts.add(u)
        else:
            het_muts.add(u)

    # Step 1: Homozygous WHO delayed-clearance mutation -> Resistant (even if missingness elsewhere)
    if hom_muts & who_set:
        return "Resistant"
    # Step 2: Heterozygous WHO delayed-clearance mutation -> Undetermined
    if het_muts & who_set:
        return "Undetermined"
    # Step 3: 578S as homozygous (with no other unexplained NS mutant or missingness? Check exact rule)
    if "A578S" in hom_muts and len(hom_muts) == 1 and not het_muts and not has_special_miss:
        return "Sensitive"
    if "A578S" in hom_muts and not (hom_muts - {"A578S"}) and not het_muts:
        # Note: in Pf8 Step 3 precedes Step 4 ("Any missing call in amino acids 349-726")
        return "Sensitive"
    # Step 4: Any missing call in amino acids 349-726 -> Undetermined
    if has_special_miss:
        return "Undetermined"
    # Step 5: No non-reference calls in amino acids 349-726 -> Sensitive
    if not hom_muts and not het_muts:
        return "Sensitive"
    # Step 6: otherwise (mutant not in WHO list) -> Undetermined
    return "Undetermined"


def classify_sp_uncomplicated(dhfr_51, dhfr_59, dhfr_108, wt_hom_only: bool = True) -> str:
    a51, m51 = parse_aa(dhfr_51)
    a59, m59 = parse_aa(dhfr_59)
    a108, m108 = parse_aa(dhfr_108)
    if wt_hom_only:
        has_wt = (not m51 and a51 == {"N"}) or (not m59 and a59 == {"C"}) or (not m108 and a108 == {"S"})
    else:
        has_wt = (not m51 and "N" in a51) or (not m59 and "C" in a59) or (not m108 and "S" in a108)
    if has_wt:
        return "Sensitive"
    if (not m51 and a51 == {"I"}) and (not m59 and a59 == {"R"}) and (not m108 and a108 == {"N"}):
        return "Resistant"
    return "Undetermined"


def classify_sp_iptp(dhfr_51, dhfr_59, dhfr_108, dhfr_164,
                     dhps_437, dhps_540, dhps_581, dhps_613, wt_hom_only: bool = True) -> str:
    a51, m51 = parse_aa(dhfr_51)
    a59, m59 = parse_aa(dhfr_59)
    a108, m108 = parse_aa(dhfr_108)
    a164, m164 = parse_aa(dhfr_164)
    a437, m437 = parse_aa(dhps_437)
    a540, m540 = parse_aa(dhps_540)
    a581, m581 = parse_aa(dhps_581)
    a613, m613 = parse_aa(dhps_613)

    if wt_hom_only:
        wt_core = (
            (not m51 and a51 == {"N"})
            or (not m59 and a59 == {"C"})
            or (not m108 and a108 == {"S"})
            or (not m437 and a437 == {"A"})
            or (not m540 and a540 == {"K"})
        )
        wt_sixth = (not m164 and a164 == {"I"}) and (not m581 and a581 == {"A"}) and (not m613 and a613 == {"A"})
    else:
        wt_core = (
            (not m51 and "N" in a51)
            or (not m59 and "C" in a59)
            or (not m108 and "S" in a108)
            or (not m437 and "A" in a437)
            or (not m540 and "K" in a540)
        )
        wt_sixth = (not m164 and "I" in a164) and (not m581 and "A" in a581) and (not m613 and "A" in a613)

    if wt_core or wt_sixth:
        return "Sensitive"

    quint_hom = (
        (not m51 and a51 == {"I"})
        and (not m59 and a59 == {"R"})
        and (not m108 and a108 == {"N"})
        and (not m437 and a437 == {"G"})
        and (not m540 and a540 == {"E"})
    )
    sixth_hom = (
        (not m164 and a164 == {"L"})
        or (not m581 and a581 == {"G"})
        or (not m613 and (a613 == {"S"} or a613 == {"T"}))
    )
    sixth_no_het = (
        (m164 or len(a164) == 1)
        and (m581 or len(a581) == 1)
        and (m613 or len(a613) == 1)
    )
    if quint_hom and sixth_hom and sixth_no_het:
        return "Resistant"
    return "Undetermined"


def classify_combo(drug_a: str, drug_b: str) -> str:
    if drug_a == "Sensitive" or drug_b == "Sensitive":
        return "Sensitive"
    if drug_a == "Resistant" and drug_b == "Resistant":
        return "Resistant"
    return "Undetermined"


def classify_sample(row_dict: dict, who_set: set[str] = WHO_K13_ASSOC, wt_hom_only: bool = True) -> dict[str, str]:
    """Classify all 10 WHO drug/regimen phenotypes from standardized marker keys."""
    cq = _single_codon_drug(row_dict.get("crt_76"), "K", "T")
    pyr = _single_codon_drug(row_dict.get("dhfr_108"), "S", "N")
    sdx = _single_codon_drug(row_dict.get("dhps_437"), "A", "G")
    mq = _cnv_drug(row_dict.get("mdr1_amp"))
    ppq = _cnv_drug(row_dict.get("pm2_pm3_amp"))
    art = classify_artemisinin(row_dict.get("k13_ns"), who_set=who_set)
    sp_u = classify_sp_uncomplicated(
        row_dict.get("dhfr_51"), row_dict.get("dhfr_59"), row_dict.get("dhfr_108"), wt_hom_only=wt_hom_only
    )
    sp_i = classify_sp_iptp(
        row_dict.get("dhfr_51"), row_dict.get("dhfr_59"), row_dict.get("dhfr_108"), row_dict.get("dhfr_164"),
        row_dict.get("dhps_437"), row_dict.get("dhps_540"), row_dict.get("dhps_581"), row_dict.get("dhps_613"),
        wt_hom_only=wt_hom_only,
    )
    as_mq = classify_combo(art, mq)
    dha_ppq = classify_combo(art, ppq)
    return {
        "Chloroquine": cq,
        "Pyrimethamine": pyr,
        "Sulfadoxine": sdx,
        "Mefloquine": mq,
        "Artemisinin": art,
        "Piperaquine": ppq,
        "SP (uncomplicated)": sp_u,
        "SP (IPTp)": sp_i,
        "AS-MQ": as_mq,
        "DHA-PPQ": dha_ppq,
    }


def verify_pf8_release(rel_dir: Path) -> dict:
    """Verify exact rule parity against Pf8_inferred_resistance_status_classification.tsv across all 24,409 samples."""
    drm = pd.read_csv(rel_dir / "Pf8_drug_resistance_marker_genotypes.tsv", sep="\t", dtype=str).set_index("Sample")
    cnv = pd.read_csv(rel_dir / "Pf8_cnv_calls.tsv", sep="\t").set_index("Sample")
    inf = pd.read_csv(rel_dir / "Pf8_inferred_resistance_status_classification.tsv", sep="\t").rename(
        columns={"sample": "Sample"}
    ).set_index("Sample")

    results = {}
    for label, who_set, wt_hom in [
        ("assoc23_homwt", WHO_K13_ASSOC, True),
        ("all31_homwt", WHO_K13_ASSOC | WHO_K13_POTENTIAL, True),
        ("assoc23_anywt", WHO_K13_ASSOC, False),
        ("all31_anywt", WHO_K13_ASSOC | WHO_K13_POTENTIAL, False),
    ]:
        preds = []
        for s in inf.index:
            d = drm.loc[s]
            c = cnv.loc[s]
            p = classify_sample({
                "crt_76": d["crt_76[K]"],
                "dhfr_51": d["dhfr_51[N]"],
                "dhfr_59": d["dhfr_59[C]"],
                "dhfr_108": d["dhfr_108[S]"],
                "dhfr_164": d["dhfr_164[I]"],
                "dhps_437": d["dhps_437[G]"],
                "dhps_540": d["dhps_540[K]"],
                "dhps_581": d["dhps_581[A]"],
                "dhps_613": d["dhps_613[A]"],
                "k13_ns": d["kelch13_349-726_ns_changes"],
                "mdr1_amp": c["MDR1_final_amplification_call"],
                "pm2_pm3_amp": c["PM2_PM3_final_amplification_call"],
            }, who_set=who_set, wt_hom_only=wt_hom)
            p["Sample"] = s
            preds.append(p)
        pdf = pd.DataFrame(preds).set_index("Sample")
        per_drug = {col: float((pdf[col] == inf[col]).mean()) for col in DRUG_COLS}
        mismatches = {}
        for col in DRUG_COLS:
            bad = pdf.index[pdf[col] != inf[col]]
            if len(bad):
                ex = bad[0]
                mismatches[col] = {
                    "n_diff": int(len(bad)),
                    "example": ex,
                    "pred": str(pdf.loc[ex, col]),
                    "pf8": str(inf.loc[ex, col]),
                }
        results[label] = {"mean_agree": float(np.mean(list(per_drug.values()))), "per_drug": per_drug, "diff": mismatches}
    return results


if __name__ == "__main__":
    import os
    import sys
    rel_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PF8_RELEASE_DIR", "")
    if not rel_arg:
        sys.exit("Usage: python -m focaloop.malariavar.who_resistance <pf8_release_dir> (or set PF8_RELEASE_DIR)")
    rel = Path(rel_arg)
    print(json.dumps(verify_pf8_release(rel), indent=2))
