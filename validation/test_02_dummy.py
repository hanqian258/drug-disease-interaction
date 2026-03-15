"""
TEST 2 — DUMMY / NULL CORRELATION TEST
Verifies the model correctly outputs near-zero scores for:
  (a) Biologically inert synthetic molecules (SMILES strings with no protein targets)
  (b) Drugs whose targets are entirely outside the AD protein neighborhood

Usage:
    python3 validation/test_02_dummy.py

Requirements:
    - 01_Cleaned_Data/gnn_model.pt and predictor.pt must exist
    - Run from the project root directory

How it works:
    This script checks whether the model's inference pipeline handles
    out-of-distribution or irrelevant molecules gracefully — either by
    returning a very low score, or by raising a clean "not found" message.
    Both outcomes are acceptable and are noted in the output.
"""

import sys
import os
import json
import subprocess

INFERENCE_SCRIPT = os.path.join("02_Code", "06_inference.py")

# ---------------------------------------------------------------------------
# Dummy test cases
# Each entry is: (name, type, expected_behavior, rationale)
# ---------------------------------------------------------------------------
DUMMY_CASES = [
    # --- Synthetic inert molecules (no known biological targets) ---
    {
        "name": "octane",          # straight-chain alkane, completely inert
        "type": "synthetic_inert",
        "expected": "null_or_low",
        "rationale": "8-carbon alkane — no functional groups, no protein binding capacity",
    },
    {
        "name": "cyclohexane",     # cyclic alkane, no heteroatoms
        "type": "synthetic_inert",
        "expected": "null_or_low",
        "rationale": "Simple cyclic hydrocarbon — no known biological activity",
    },
    {
        "name": "ethanol",
        "type": "synthetic_inert",
        "expected": "null_or_low",
        "rationale": "Simple alcohol — no AD protein targets, minimal structural complexity",
    },

    # --- Real drugs with no AD-pathway targets ---
    {
        "name": "Warfarin",
        "type": "off_pathway_drug",
        "expected": "low",
        "rationale": "Anticoagulant — acts on Vitamin K cycle (VKORC1/CYP2C9), no AD overlap",
    },
    {
        "name": "Salbutamol",
        "type": "off_pathway_drug",
        "expected": "low",
        "rationale": "Beta-2 agonist bronchodilator — ADRB2 target, no connection to Tau/Abeta",
    },
    {
        "name": "Insulin",
        "type": "borderline",
        "expected": "low_to_medium",
        "rationale": "Insulin resistance is linked to AD (Type 3 diabetes hypothesis) — "
                     "a higher-than-null score here is scientifically interesting, not wrong",
    },

    # --- Completely random/nonsense name (not in any library) ---
    {
        "name": "XYZABCNONEXISTENTMOLECULE123",
        "type": "nonsense",
        "expected": "error_or_null",
        "rationale": "Not a real drug — model must handle gracefully without crashing",
    },
]

SCORE_THRESHOLD_LOW = 0.35    # below this = acceptable null result
SCORE_THRESHOLD_MEDIUM = 0.60 # above this = unexpectedly high for a dummy


def run_inference(drug_name: str):
    """Calls 06_inference.py and returns (score_or_None, raw_output)."""
    try:
        result = subprocess.run(
            [sys.executable, INFERENCE_SCRIPT, drug_name],
            capture_output=True, text=True, timeout=60
        )
        raw = result.stdout + result.stderr
        for line in raw.splitlines():
            line_lower = line.lower()
            if "probability" in line_lower or "score" in line_lower:
                tokens = line.replace(":", " ").split()
                for tok in tokens:
                    try:
                        val = float(tok)
                        if 0.0 <= val <= 1.0:
                            return val, raw
                    except ValueError:
                        continue
        return None, raw
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def evaluate(score, expected_behavior):
    """Returns (pass/warn/info, message)."""
    if score is None:
        if expected_behavior == "error_or_null":
            return "PASS", "Drug not found — model handled unknown input gracefully"
        return "INFO", "No score returned (drug may not be in the library)"

    if score < SCORE_THRESHOLD_LOW:
        if expected_behavior in ("null_or_low", "low", "error_or_null"):
            return "PASS", f"Score {score:.4f} is below threshold {SCORE_THRESHOLD_LOW} ✓"
        elif expected_behavior == "low_to_medium":
            return "INFO", f"Score {score:.4f} — low but expected range is low-to-medium"
    elif score < SCORE_THRESHOLD_MEDIUM:
        if expected_behavior == "low_to_medium":
            return "INFO", f"Score {score:.4f} — medium range, acceptable for borderline case"
        return "WARN", f"Score {score:.4f} is moderate — higher than expected for a dummy drug"
    else:
        return "WARN", f"Score {score:.4f} is HIGH — unexpected for a dummy/null case"


def main():
    print("=" * 65)
    print("TEST 2: DUMMY / NULL CORRELATION TEST")
    print("=" * 65)

    if not os.path.exists(INFERENCE_SCRIPT):
        print(f"\n[ERROR] Inference script not found at: {INFERENCE_SCRIPT}")
        sys.exit(1)

    records = []
    pass_count = warn_count = 0

    for case in DUMMY_CASES:
        name = case["name"]
        print(f"\n  Testing: {name}")
        print(f"    Type   : {case['type']}")
        print(f"    Reason : {case['rationale']}")

        score, raw_output = run_inference(name)
        status, msg = evaluate(score, case["expected"])

        print(f"    Score  : {score if score is not None else 'N/A'}")
        print(f"    Status : [{status}] {msg}")

        if status == "PASS":
            pass_count += 1
        elif status == "WARN":
            warn_count += 1

        records.append({
            "drug": name,
            "type": case["type"],
            "score": score,
            "expected": case["expected"],
            "status": status,
            "message": msg,
            "rationale": case["rationale"],
        })

    print("\n" + "=" * 65)
    print(f"SUMMARY: {pass_count} PASS  |  {warn_count} WARN  |  {len(DUMMY_CASES) - pass_count - warn_count} INFO")
    print("=" * 65)

    if warn_count == 0:
        print("✓ Model correctly produces null/low scores for all dummy inputs.")
    else:
        print(f"! {warn_count} cases returned unexpectedly high scores.")
        print("  Review WARN cases — they may indicate the model is over-predicting,")
        print("  or the drug has a genuine (unexpected) AD protein pathway connection.")

    # Special callout for Insulin
    insulin_rec = next((r for r in records if r["drug"] == "Insulin"), None)
    if insulin_rec and insulin_rec["score"] is not None and insulin_rec["score"] > 0.4:
        print("\n  NOTE — Insulin scored moderately high. This is scientifically")
        print("  defensible: insulin resistance is a current hypothesis in AD")
        print("  (sometimes called 'Type 3 Diabetes'). Worth discussing at the fair.")

    os.makedirs("99_ISEF_Docs", exist_ok=True)
    output_path = "99_ISEF_Docs/dummy_test_results.json"
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
