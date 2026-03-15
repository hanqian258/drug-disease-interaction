"""
TEST 3 — DISCOVERY SCREEN
Batch-runs a curated library of CNS-adjacent and repurposing-candidate drugs
through the model and outputs a ranked table of potential AD candidates.

Usage:
    python3 validation/test_03_discovery.py

Output:
    - Console: ranked table grouped by tier (High / Medium / Low)
    - 99_ISEF_Docs/discovery_results.json  — full machine-readable results
    - 99_ISEF_Docs/discovery_report.txt    — human-readable summary for ISEF docs

Cross-reference logic:
    Any drug scoring >= 0.70 is flagged as a "discovery candidate" and
    annotated with its known targets and any published AD trial evidence.
"""

import sys
import os
import json
import subprocess
import time

INFERENCE_SCRIPT = os.path.join("02_Code", "06_inference.py")

# ---------------------------------------------------------------------------
# Candidate drug library
# Organised into families that are biologically plausible for AD repurposing.
# Literature notes are included for science-fair justification.
# ---------------------------------------------------------------------------
DRUG_LIBRARY = [
    # --- Approved AD drugs (should score HIGH — validates the screen) ---
    {"name": "Donepezil",    "class": "AD_approved",   "mechanism": "AChE inhibitor",        "ad_trial": True},
    {"name": "Memantine",    "class": "AD_approved",   "mechanism": "NMDA antagonist",        "ad_trial": True},
    {"name": "Rivastigmine", "class": "AD_approved",   "mechanism": "AChE/BuChE inhibitor",   "ad_trial": True},
    {"name": "Galantamine",  "class": "AD_approved",   "mechanism": "AChE inhibitor + nAChR", "ad_trial": True},

    # --- Active AD pipeline (experimental, should score moderately-high) ---
    {"name": "Lecanemab",    "class": "AD_pipeline",   "mechanism": "Anti-amyloid antibody",  "ad_trial": True},
    {"name": "Aducanumab",   "class": "AD_pipeline",   "mechanism": "Anti-amyloid antibody",  "ad_trial": True},

    # --- CNS drugs — neurodegeneration-adjacent ---
    {"name": "Lithium",      "class": "CNS",           "mechanism": "GSK-3b inhibitor (tau phosphorylation)", "ad_trial": True},
    {"name": "Valproate",    "class": "CNS",           "mechanism": "HDAC inhibitor / mood stabilizer",       "ad_trial": True},
    {"name": "Risperidone",  "class": "CNS",           "mechanism": "D2/5-HT2A antagonist",   "ad_trial": False},
    {"name": "Quetiapine",   "class": "CNS",           "mechanism": "Atypical antipsychotic",  "ad_trial": False},
    {"name": "Sertraline",   "class": "CNS_SSRI",      "mechanism": "SERT inhibitor",          "ad_trial": True},
    {"name": "Fluoxetine",   "class": "CNS_SSRI",      "mechanism": "SERT inhibitor",          "ad_trial": True},
    {"name": "Escitalopram", "class": "CNS_SSRI",      "mechanism": "SERT inhibitor",          "ad_trial": True},
    {"name": "Venlafaxine",  "class": "CNS_SNRI",      "mechanism": "SERT + NET inhibitor",    "ad_trial": False},
    {"name": "Duloxetine",   "class": "CNS_SNRI",      "mechanism": "SERT + NET inhibitor",    "ad_trial": False},

    # --- Anti-inflammatory / immune ---
    {"name": "Minocycline",  "class": "anti_inflam",   "mechanism": "Tetracycline / neuroinflammation", "ad_trial": True},
    {"name": "Indomethacin", "class": "anti_inflam",   "mechanism": "COX inhibitor / NSAID",           "ad_trial": True},
    {"name": "Celecoxib",    "class": "anti_inflam",   "mechanism": "COX-2 selective NSAID",           "ad_trial": True},
    {"name": "Dexamethasone","class": "anti_inflam",   "mechanism": "Glucocorticoid / immune",         "ad_trial": False},

    # --- Metabolic / diabetes repurposing candidates ---
    {"name": "Metformin",    "class": "metabolic",     "mechanism": "AMPK activator / insulin sensitizer", "ad_trial": True},
    {"name": "Liraglutide",  "class": "metabolic",     "mechanism": "GLP-1 agonist",                       "ad_trial": True},
    {"name": "Semaglutide",  "class": "metabolic",     "mechanism": "GLP-1 agonist (Ozempic)",             "ad_trial": True},
    {"name": "Pioglitazone", "class": "metabolic",     "mechanism": "PPAR-gamma agonist",                  "ad_trial": True},

    # --- Autophagy / proteasome ---
    {"name": "Rapamycin",    "class": "autophagy",     "mechanism": "mTOR inhibitor / autophagy",     "ad_trial": True},
    {"name": "Everolimus",   "class": "autophagy",     "mechanism": "mTOR inhibitor",                  "ad_trial": False},

    # --- Cardiovascular (often comorbid with AD) ---
    {"name": "Atorvastatin", "class": "statin",        "mechanism": "HMG-CoA reductase / cholesterol", "ad_trial": True},
    {"name": "Simvastatin",  "class": "statin",        "mechanism": "HMG-CoA reductase / cholesterol", "ad_trial": True},
    {"name": "Losartan",     "class": "cardiovascular","mechanism": "AT1 receptor blocker",             "ad_trial": True},
    {"name": "Amlodipine",   "class": "cardiovascular","mechanism": "Calcium channel blocker",          "ad_trial": False},

    # --- Gut / neuroprotection wild cards ---
    {"name": "Rifampicin",   "class": "antibiotic",    "mechanism": "RNA polymerase / amyloid disaggregation", "ad_trial": True},
    {"name": "Doxycycline",  "class": "antibiotic",    "mechanism": "Tetracycline / neuroinflammation",        "ad_trial": True},

    # --- Negative controls (should score LOW) ---
    {"name": "Amoxicillin",  "class": "negative_ctrl", "mechanism": "Penicillin antibiotic",      "ad_trial": False},
    {"name": "Warfarin",     "class": "negative_ctrl", "mechanism": "Anticoagulant / Vitamin K",  "ad_trial": False},
    {"name": "Omeprazole",   "class": "negative_ctrl", "mechanism": "Proton pump inhibitor",       "ad_trial": False},
]

DISCOVERY_THRESHOLD   = 0.70   # scores above this = high-priority candidate
MODERATE_THRESHOLD    = 0.45   # scores between 0.45-0.70 = moderate interest


def run_inference(drug_name: str):
    try:
        result = subprocess.run(
            [sys.executable, INFERENCE_SCRIPT, drug_name],
            capture_output=True, text=True, timeout=60
        )
        raw = result.stdout + result.stderr
        for line in raw.splitlines():
            if "probability" in line.lower() or "score" in line.lower():
                tokens = line.replace(":", " ").split()
                for tok in tokens:
                    try:
                        val = float(tok)
                        if 0.0 <= val <= 1.0:
                            return val
                    except ValueError:
                        continue
        return None
    except Exception:
        return None


def tier(score):
    if score is None:
        return "N/A"
    if score >= DISCOVERY_THRESHOLD:
        return "HIGH"
    elif score >= MODERATE_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def main():
    print("=" * 70)
    print("TEST 3: DISCOVERY SCREEN")
    print(f"  Screening {len(DRUG_LIBRARY)} drugs for AD correlation")
    print(f"  HIGH threshold : {DISCOVERY_THRESHOLD}")
    print(f"  MEDIUM threshold: {MODERATE_THRESHOLD}")
    print("=" * 70)

    if not os.path.exists(INFERENCE_SCRIPT):
        print(f"\n[ERROR] Inference script not found at: {INFERENCE_SCRIPT}")
        sys.exit(1)

    results = []
    for i, drug in enumerate(DRUG_LIBRARY, 1):
        name = drug["name"]
        print(f"  [{i:02d}/{len(DRUG_LIBRARY)}] {name:<16}", end=" ", flush=True)
        score = run_inference(name)
        t = tier(score)
        print(f"score={score:.4f if score is not None else 'N/A':>8}  [{t}]")
        results.append({**drug, "score": score, "tier": t})
        time.sleep(0.05)  # slight pause to avoid overwhelming subprocess queue

    # Sort by score descending
    scored = sorted([r for r in results if r["score"] is not None],
                    key=lambda x: x["score"], reverse=True)
    failed = [r for r in results if r["score"] is None]

    # -----------------------------------------------------------------------
    # Print ranked output by tier
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RANKED RESULTS")
    print("=" * 70)

    for t_label, threshold_label in [
        ("HIGH",   f"(score >= {DISCOVERY_THRESHOLD})"),
        ("MEDIUM", f"({MODERATE_THRESHOLD} <= score < {DISCOVERY_THRESHOLD})"),
        ("LOW",    f"(score < {MODERATE_THRESHOLD})"),
    ]:
        tier_results = [r for r in scored if r["tier"] == t_label]
        if not tier_results:
            continue
        print(f"\n  [{t_label}] {threshold_label}")
        print(f"  {'Drug':<16} {'Score':>7}  {'Class':<16}  {'Trial':>5}  Mechanism")
        print("  " + "-" * 65)
        for r in tier_results:
            trial_flag = "YES" if r["ad_trial"] else "no"
            print(f"  {r['name']:<16} {r['score']:>7.4f}  {r['class']:<16}  {trial_flag:>5}  {r['mechanism']}")

    # -----------------------------------------------------------------------
    # Discovery candidates — novel drugs not in AD_approved/AD_pipeline
    # -----------------------------------------------------------------------
    novel_candidates = [
        r for r in scored
        if r["tier"] == "HIGH"
        and r["class"] not in ("AD_approved", "AD_pipeline", "negative_ctrl")
    ]
    if novel_candidates:
        print("\n" + "=" * 70)
        print("★  DISCOVERY CANDIDATES  ★")
        print("Drugs scoring HIGH that are NOT currently approved/pipeline AD drugs:")
        print("=" * 70)
        for r in novel_candidates:
            trial_note = "— already in AD clinical trials" if r["ad_trial"] else "— NO current AD trial (novel!)"
            print(f"  {r['name']:<16}  score={r['score']:.4f}  {trial_note}")
            print(f"    Mechanism : {r['mechanism']}")
            print(f"    Class     : {r['class']}")
            print()
        print("  Suggested next steps for novel candidates:")
        print("  1. Check ChEMBL / BindingDB for protein targets overlapping your PPI network")
        print("  2. Search ClinicalTrials.gov for any unreported AD trials")
        print("  3. Cross-reference DisGeNET for disease-gene associations")
    else:
        print("\n  No novel HIGH-scoring candidates found in this screen.")
        print("  Consider lowering DISCOVERY_THRESHOLD or expanding the drug library.")

    # -----------------------------------------------------------------------
    # Validation check: did approved AD drugs score highest?
    # -----------------------------------------------------------------------
    approved_scores = [r["score"] for r in scored if r["class"] == "AD_approved"]
    neg_scores = [r["score"] for r in scored if r["class"] == "negative_ctrl"]
    if approved_scores and neg_scores:
        avg_approved = sum(approved_scores) / len(approved_scores)
        avg_neg = sum(neg_scores) / len(neg_scores)
        print(f"\n  Validation check:")
        print(f"    Avg score — approved AD drugs : {avg_approved:.4f}")
        print(f"    Avg score — negative controls : {avg_neg:.4f}")
        if avg_approved > avg_neg:
            print("    ✓ Approved drugs scored higher than negative controls — model is consistent.")
        else:
            print("    ✗ Negative controls scored higher than AD drugs — check model calibration.")

    # Save outputs
    os.makedirs("99_ISEF_Docs", exist_ok=True)
    json_path = "99_ISEF_Docs/discovery_results.json"
    txt_path  = "99_ISEF_Docs/discovery_report.txt"

    with open(json_path, "w") as f:
        json.dump(scored + failed, f, indent=2)

    with open(txt_path, "w") as f:
        f.write("DISCOVERY SCREEN REPORT\n")
        f.write(f"Drugs screened: {len(DRUG_LIBRARY)}\n")
        f.write(f"Successfully scored: {len(scored)}\n\n")
        f.write(f"{'Rank':<5} {'Drug':<16} {'Score':>7}  {'Tier':<8}  {'Class':<16}  Mechanism\n")
        f.write("-" * 80 + "\n")
        for i, r in enumerate(scored, 1):
            f.write(f"{i:<5} {r['name']:<16} {r['score']:>7.4f}  {r['tier']:<8}  {r['class']:<16}  {r['mechanism']}\n")

    print(f"\n  Results saved:")
    print(f"    {json_path}")
    print(f"    {txt_path}")


if __name__ == "__main__":
    main()
