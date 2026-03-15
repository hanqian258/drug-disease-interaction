"""
TEST 1 — METRIC TEST
Runs known positive-control (approved AD drugs) and negative-control drugs
through the inference pipeline and reports a ranked table + ROC-AUC score.

Usage:
    python3 validation/test_01_metric.py

Requirements:
    - 01_Cleaned_Data/gnn_model.pt and predictor.pt must exist (run 05_train_gcn.py first)
    - Run from the project root directory
"""

import sys
import os
import json
import subprocess
import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Ground-truth labels for our test drugs
# 1 = known/expected AD association, 0 = no expected AD association
# ---------------------------------------------------------------------------
TEST_DRUGS = {
    # --- POSITIVE CONTROLS (FDA-approved AD drugs) ---
    "Donepezil":    {"label": 1, "reason": "FDA-approved cholinesterase inhibitor (AD)"},
    "Memantine":    {"label": 1, "reason": "FDA-approved NMDA antagonist (AD)"},
    "Rivastigmine": {"label": 1, "reason": "FDA-approved cholinesterase inhibitor (AD)"},
    "Galantamine":  {"label": 1, "reason": "FDA-approved cholinesterase inhibitor (AD)"},
    "Lecanemab":    {"label": 1, "reason": "FDA-approved anti-amyloid antibody (AD, 2023)"},

    # --- BORDERLINE (interesting to discuss at science fair) ---
    "Metformin":    {"label": 1, "reason": "Emerging AD trial evidence (TAME trial); insulin-resistance pathway"},

    # --- NEGATIVE CONTROLS (no known AD mechanism) ---
    "Amoxicillin":  {"label": 0, "reason": "Penicillin antibiotic; no neurological pathway"},
    "Ibuprofen":    {"label": 0, "reason": "NSAID; no direct AD target (neuroinflammation link debated)"},
    "Metoprolol":   {"label": 0, "reason": "Beta-blocker; cardiovascular, not CNS/AD pathway"},
    "Omeprazole":   {"label": 0, "reason": "Proton pump inhibitor; gastrointestinal, no AD pathway"},
}

INFERENCE_SCRIPT = os.path.join("02_Code", "06_inference.py")


def run_inference(drug_name: str) -> float | None:
    """Calls 06_inference.py and parses the probability from stdout."""
    try:
        result = subprocess.run(
            [sys.executable, INFERENCE_SCRIPT, drug_name],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        # Try to find a float probability in the output
        for line in output.splitlines():
            line_lower = line.lower()
            if "probability" in line_lower or "score" in line_lower:
                tokens = line.replace(":", " ").split()
                for tok in tokens:
                    try:
                        val = float(tok)
                        if 0.0 <= val <= 1.0:
                            return val
                    except ValueError:
                        continue
        return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {drug_name}")
        return None
    except Exception as e:
        print(f"  [ERROR] {drug_name}: {e}")
        return None


def main():
    print("=" * 60)
    print("TEST 1: METRIC TEST — Known Drug Benchmark")
    print("=" * 60)

    if not os.path.exists(INFERENCE_SCRIPT):
        print(f"\n[ERROR] Inference script not found at: {INFERENCE_SCRIPT}")
        print("Make sure you are running from the project root directory.")
        sys.exit(1)

    results = []
    for drug, meta in TEST_DRUGS.items():
        print(f"  Running: {drug} ...", end=" ", flush=True)
        prob = run_inference(drug)
        if prob is not None:
            print(f"score={prob:.4f}")
        else:
            print("FAILED (drug may not be in the model's library)")
        results.append({
            "drug": drug,
            "label": meta["label"],
            "score": prob,
            "reason": meta["reason"],
        })

    # Filter out drugs where inference failed
    valid = [r for r in results if r["score"] is not None]

    print("\n" + "=" * 60)
    print("RESULTS — Ranked by Score (highest → lowest)")
    print("=" * 60)
    print(f"{'Drug':<16} {'Score':>7}  {'Expected':>10}  Notes")
    print("-" * 60)
    for r in sorted(valid, key=lambda x: x["score"], reverse=True):
        expected = "POSITIVE" if r["label"] == 1 else "NEGATIVE"
        print(f"{r['drug']:<16} {r['score']:>7.4f}  {expected:>10}  {r['reason']}")

    # ROC-AUC (only meaningful if we have both classes)
    labels = [r["label"] for r in valid]
    scores = [r["score"] for r in valid]
    if len(set(labels)) == 2:
        auc = roc_auc_score(labels, scores)
        print(f"\nROC-AUC Score: {auc:.4f}")
        if auc >= 0.80:
            print("  ✓ Excellent discrimination between known AD drugs and non-AD drugs.")
        elif auc >= 0.65:
            print("  ~ Good — model is directionally correct but not perfectly separating.")
        else:
            print("  ✗ Low AUC — model may need retraining or more positive examples.")
    else:
        print("\n[NOTE] Cannot compute ROC-AUC: need both positive and negative results.")

    # Save results to JSON
    os.makedirs("99_ISEF_Docs", exist_ok=True)
    output_path = "99_ISEF_Docs/metric_test_results.json"
    with open(output_path, "w") as f:
        json.dump({"results": valid, "auc": auc if len(set(labels)) == 2 else None}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
