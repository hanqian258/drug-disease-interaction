"""
09_results_validation.py — Results Section Validation
Updated for 6-disease model (AD, PD, ADHD, Bipolar, ALS, Dementia)
160 drugs, 96 proteins, 105 positive training edges.
"""
import subprocess, sys, os
import numpy as np
from scipy import stats
import pandas as pd

INFERENCE = os.path.join(os.path.dirname(__file__), '07_inference.py')
OUT_DIR   = '99_ISEF_Docs'
os.makedirs(OUT_DIR, exist_ok=True)

def get_score(drug_name):
    try:
        result = subprocess.run([sys.executable, INFERENCE, drug_name],
                                capture_output=True, text=True, timeout=60)
        for line in result.stdout.splitlines():
            if 'Probability' in line:
                try: return float(line.split(':')[-1].strip())
                except ValueError: pass
    except Exception: pass
    return None

def separator(title=''):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        return f"\n{'='*pad} {title} {'='*pad}"
    return '=' * w

POSITIVE_CONTROLS = [
    ("Donepezil",        "AChE inhibitor, FDA-approved AD"),
    ("Memantine",        "NMDA antagonist, FDA-approved AD"),
    ("Rivastigmine",     "AChE/BuChE inhibitor, FDA-approved AD"),
    ("Galantamine",      "AChE inhibitor + nAChR, FDA-approved AD"),
    ("Tacrine",          "AChE inhibitor, first FDA-approved AD drug"),
    ("Riluzole",         "glutamate antagonist, FDA-approved ALS"),
    ("Edaravone",        "free radical scavenger, FDA-approved ALS"),
    ("Lithium",          "GSK-3b inhibitor, Bipolar first-line"),
    ("Haloperidol",      "D2 antagonist, approved Bipolar/psychosis"),
]

NEGATIVE_CONTROLS = [
    ("Amoxicillin",      "Penicillin antibiotic, no CNS pathway"),
    ("Warfarin",         "Anticoagulant, Vitamin K pathway only"),
    ("Omeprazole",       "Proton pump inhibitor, GI only"),
    ("Furosemide",       "Loop diuretic, renal only"),
    ("Cisplatin",        "Platinum chemotherapy, DNA crosslinker"),
    ("Methotrexate",     "Antifolate chemotherapy, no CNS"),
    ("Fluconazole",      "Antifungal, ergosterol synthesis"),
]

def run_metric_test():
    lines = [separator('RESULT 1: METRIC TEST')]
    lines.append("Testing whether model ranks approved disease drugs above non-CNS drugs.\n")
    pos_scores, neg_scores = [], []
    lines.append(f"  {'Drug':<22} {'Score':>7}  {'Expected':>8}  Category")
    lines.append(f"  {'-'*70}")
    for drug, note in POSITIVE_CONTROLS:
        s = get_score(drug)
        pos_scores.append(s)
        status = 'PASS' if s and s >= 0.75 else 'REVIEW'
        lines.append(f"  {drug:<22} {s:>7.4f}  {'HIGH':>8}  {note}  [{status}]")
    lines.append('')
    for drug, note in NEGATIVE_CONTROLS:
        s = get_score(drug)
        neg_scores.append(s)
        status = 'PASS' if s and s < 0.35 else 'REVIEW'
        lines.append(f"  {drug:<22} {s:>7.4f}  {'LOW':>8}  {note}  [{status}]")
    pos_scores = [s for s in pos_scores if s is not None]
    neg_scores = [s for s in neg_scores if s is not None]
    stat, pval = stats.mannwhitneyu(pos_scores, neg_scores, alternative='greater')
    delta = np.mean(pos_scores) - np.mean(neg_scores)
    perfect_sep = min(pos_scores) > max(neg_scores)
    lines.append(f"\n  Approved drugs  — mean={np.mean(pos_scores):.4f}, std={np.std(pos_scores):.4f}")
    lines.append(f"  Non-CNS drugs   — mean={np.mean(neg_scores):.4f}, std={np.std(neg_scores):.4f}")
    lines.append(f"  Score separation — Delta={delta:.4f}")
    lines.append(f"  Mann-Whitney U   — p={pval:.4f} {'(significant, p<0.05)' if pval < 0.05 else '(not significant)'}")
    lines.append(f"  Perfect separation — {'YES' if perfect_sep else 'NO: some overlap'}")
    return '\n'.join(lines), pos_scores, neg_scores

DUMMY_DRUGS = [
    ("Cisplatin",        "platinum chemotherapy, DNA crosslinker"),
    ("Methotrexate",     "antifolate chemotherapy"),
    ("Doxorubicin",      "anthracycline, topoisomerase inhibitor"),
    ("Spironolactone",   "aldosterone antagonist, renal"),
    ("Azithromycin",     "macrolide antibiotic, ribosome inhibitor"),
    ("Ciprofloxacin",    "fluoroquinolone antibiotic"),
    ("Fluconazole",      "antifungal, ergosterol synthesis"),
    ("Baclofen",         "GABA-B agonist, muscle relaxant"),
    ("Bisoprolol",       "beta-1 blocker, cardiovascular only"),
    ("Tamoxifen",        "selective estrogen receptor modulator"),
]

def run_dummy_test(pos_mean):
    lines = [separator('RESULT 2: DUMMY / NULL TEST')]
    lines.append("Off-pathway drugs should score below the approved-drug mean.\n")
    lines.append(f"  Reference — approved drug mean score: {pos_mean:.4f}\n")
    lines.append(f"  {'Drug':<22} {'Score':>7}  {'vs mean':>8}  Notes")
    lines.append(f"  {'-'*65}")
    scores = []
    for drug, note in DUMMY_DRUGS:
        s = get_score(drug)
        if s is None:
            lines.append(f"  {drug:<22} {'N/A':>7}  {'':>8}  {note}")
            continue
        scores.append(s)
        diff = s - pos_mean
        flag = 'PASS' if s < pos_mean else 'FLAG'
        lines.append(f"  {drug:<22} {s:>7.4f}  {diff:>+8.4f}  {note}  [{flag}]")
    if scores:
        flagged = sum(1 for s in scores if s >= pos_mean)
        lines.append(f"\n  {len(scores)-flagged}/{len(scores)} off-pathway drugs scored below approved mean")
        lines.append("  Model correctly suppresses scores for non-CNS drugs." if flagged == 0
                     else f"  {flagged} drug(s) scored unexpectedly high.")
    return '\n'.join(lines)

def run_kfold_summary():
    lines = [separator('RESULT 3: K-FOLD CROSS VALIDATION')]
    kfold_path = os.path.join(OUT_DIR, 'kfold_results.txt')
    if os.path.exists(kfold_path):
        with open(kfold_path) as f:
            content = f.read()
        lines.append(content)
        auc_val = None
        for line in content.splitlines():
            if 'Mean AUC' in line:
                try: auc_val = float(line.split(':')[-1].strip())
                except ValueError: pass
        if auc_val:
            lines.append(f"  Interpretation:")
            lines.append(f"  AUC = {auc_val:.4f}: model ranked a known therapeutic above a")
            lines.append(f"  non-therapeutic in {auc_val*100:.1f}% of held-out comparisons.")
            if auc_val > 0.99:
                lines.append(f"  NOTE: AUC > 0.99 may indicate overfitting given dataset size.")
                lines.append(f"  Recommend validating predictions against independent clinical data.")
            elif auc_val > 0.80:
                lines.append(f"  Exceeds the 0.80 threshold for strong discriminative performance.")
    else:
        lines.append("  kfold_results.txt not found. Run 05b_kfold_eval.py first.")
    return '\n'.join(lines)

REPURPOSING_CANDIDATES = [

    # ── SECTION A: AD TRIAL VALIDATION ──────────────────────────────
    # These drugs have published AD clinical trials — used to show
    # the model independently recovers known candidates.
    ("Resveratrol",      "SIRT1 activator — Phase 2 AD trial completed"),
    ("Berberine",        "GSK-3b/tau + cholinesterase — AD trials published"),
    ("Metformin",        "AMPK activator — active TAME trial, AD arm"),
    ("Sildenafil",       "PDE5/cGMP pathway — retrospective AD cohort data"),
    ("Melatonin",        "antioxidant/circadian — AD trials published"),
    ("Curcumin",         "tau + amyloid aggregation inhibitor — AD trials"),
    ("Nicotine",         "nAChR agonist CHRNA7 — AD patch trials"),
    ("Doxycycline",      "tetracycline — amyloid aggregation inhibitor, AD trial"),
    ("Rapamycin",        "mTOR inhibitor — Phase 1/2 AD trials ongoing"),
    ("Cannabidiol",      "anti-inflammatory/neuroprotective — AD trials"),
    ("Lenalidomide",     "thalidomide analogue — AD neuroinflammation trials"),

    # ── SECTION B: OFF-LABEL DISCOVERY CANDIDATES ───────────────────
    # FDA-approved for other indications, no dedicated AD trial.
    # High scores here represent genuine novel predictions.

    # Cardiovascular / Metabolic
    ("Losartan",         "AT1R blocker, antihypertensive — neuroinflammation hypothesis"),
    ("Empagliflozin",    "SGLT2 inhibitor, T2D — mTOR/autophagy adjacent, no AD trial"),
    ("Canagliflozin",    "SGLT2 inhibitor, T2D — neuroinflammation hypothesis"),
    ("Pravastatin",      "HMG-CoA reductase inhibitor — no dedicated AD trial"),
    ("Pioglitazone",     "PPAR-gamma agonist, T2D — insulin resistance pathway"),

    # Immunology / Autoimmune
    ("Hydroxychloroquine","antimalarial/lupus — lysosomal pathway, no AD trial"),
    ("Tofacitinib",      "JAK1/3 inhibitor, RA — neuroinflammation, no AD trial"),
    ("Baricitinib",      "JAK1/2 inhibitor, RA — neuroinflammation hypothesis"),
    ("Mycophenolate",    "inosine monophosphate dehydrogenase inhibitor, immunosuppressant"),

    # Neurology (non-AD)
    ("Levetiracetam",    "SV2A modulator, epilepsy — synaptic vesicle pathway"),
    ("Fingolimod",       "S1P receptor modulator, MS — blood-brain barrier integrity"),
    ("Imatinib",         "BCR-ABL/c-Abl inhibitor, CML — tau phosphorylation pathway"),
    ("Dasatinib",        "tyrosine kinase inhibitor — senolytic, no AD trial"),

    # Endocrinology
    ("Liraglutide",      "GLP-1 agonist, T2D/obesity — neuroprotection, no AD trial"),
    ("Exenatide",        "GLP-1 agonist, T2D — Parkinson's trial but not AD"),

    # ── SECTION C: REFERENCE NEGATIVES ──────────────────────────────
    # Off-pathway drugs with no biological connection to AD.
    # Should score low — used to anchor the threshold.
    ("Amoxicillin",      "penicillin antibiotic — reference negative"),
    ("Warfarin",         "anticoagulant, Vitamin K pathway — reference negative"),
    ("Cisplatin",        "platinum chemotherapy, DNA crosslinker — reference negative"),
    ("Furosemide",       "loop diuretic, renal only — reference negative"),
    ("Azithromycin",     "macrolide antibiotic — reference negative"),
]


def run_discovery_screen(pos_mean):
    lines = [separator('RESULT 4: DISCOVERY / REPURPOSING SCREEN')]
    lines.append("Candidates split into three sections:\n"
                 "  A) AD trial validation — model should recover known candidates\n"
                 "  B) Off-label discovery — FDA-approved, no AD trial, genuine predictions\n"
                 "  C) Reference negatives — should score below threshold\n")
    lines.append(f"  Approval threshold (approved drug mean): {pos_mean:.4f}\n")

    # ── Tag each drug by section ──────────────────────────────────
    section_map = {}
    for drug, note in REPURPOSING_CANDIDATES:
        if 'reference negative' in note:
            section_map[drug] = 'C'
        elif 'no AD trial' in note or 'not AD' in note:
            section_map[drug] = 'B'
        else:
            section_map[drug] = 'A'

    rows = []
    for drug, note in REPURPOSING_CANDIDATES:
        s = get_score(drug)
        rows.append({
            'Drug': drug,
            'Score': s,
            'Notes': note,
            'Section': section_map[drug]
        })

    for sec, label in [
        ('A', 'SECTION A — AD Trial Validation (model should recover these)'),
        ('B', 'SECTION B — Off-Label Discovery (genuine novel predictions)'),
        ('C', 'SECTION C — Reference Negatives (should score below threshold)'),
    ]:
        lines.append(f"\n  {label}")
        lines.append(f"  {'-'*65}")
        sec_rows = [r for r in rows if r['Section'] == sec]
        for r in sorted(sec_rows, key=lambda x: -(x['Score'] or 0)):
            s = r['Score']
            score_str = f"{s:.4f}" if s is not None else "N/A"
            flag = ''
            if sec == 'A' and s and s >= pos_mean:
                flag = '  ✓ RECOVERED'
            elif sec == 'B' and s and s >= pos_mean:
                flag = '  *** NOVEL CANDIDATE ***'
            elif sec == 'C' and s and s >= pos_mean:
                flag = '  !! UNEXPECTED — review'
            lines.append(f"  {r['Drug']:<28} {score_str:>7}  {r['Notes']}{flag}")

    # ── Summary statistics per section ───────────────────────────
    lines.append(f"\n  {'─'*60}")
    for sec, label in [('A', 'AD Validation'), ('B', 'Off-Label Discovery'), ('C', 'Negatives')]:
        sec_scores = [r['Score'] for r in rows
                      if r['Section'] == sec and r['Score'] is not None]
        if sec_scores:
            above = sum(1 for s in sec_scores if s >= pos_mean)
            lines.append(f"  {label}: {above}/{len(sec_scores)} scored >= threshold "
                         f"(mean={np.mean(sec_scores):.4f})")

    # ── Save CSV with section column ─────────────────────────────
    df = pd.DataFrame(rows).sort_values(['Section', 'Score'], ascending=[True, False])
    df.to_csv(os.path.join(OUT_DIR, 'discovery_candidates.csv'), index=False)
    lines.append(f"\n  Saved -> {OUT_DIR}/discovery_candidates.csv")
    return '\n'.join(lines)

def main():
    try:
        import torch
        data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)
        n_drugs    = data['drug'].x.shape[0]
        n_proteins = data['protein'].x.shape[0]
        n_diseases = data['disease'].x.shape[0]
        n_treats   = data['drug', 'treats', 'disease'].edge_index.shape[1]
    except Exception:
        n_drugs, n_proteins, n_diseases, n_treats = 160, 96, 6, 105

    print(f"Graph: {n_drugs} drugs, {n_proteins} proteins, {n_diseases} diseases, {n_treats} training edges\n")

    print("[1/4] Metric test...")
    metric_text, pos_scores, neg_scores = run_metric_test()
    pos_mean = float(np.mean([s for s in pos_scores if s])) if pos_scores else 0.44

    print("[2/4] Dummy test...")
    dummy_text = run_dummy_test(pos_mean)

    print("[3/4] K-fold summary...")
    kfold_text = run_kfold_summary()

    print("[4/4] Discovery screen...")
    discovery_text = run_discovery_screen(pos_mean)

    auc_str = "see kfold_results.txt"
    try:
        with open(os.path.join(OUT_DIR, 'kfold_results.txt')) as f:
            for line in f:
                if 'Mean AUC' in line: auc_str = line.strip()
    except Exception: pass

    full_report = '\n\n'.join([metric_text, dummy_text, kfold_text, discovery_text])
    full_report += f"\n\n{'='*60}\n  {auc_str}\n"
    full_report += f"  Graph: {n_drugs} drugs, {n_proteins} proteins, {n_diseases} diseases\n"
    full_report += f"  Training edges: {n_treats} positive drug-disease associations\n{'='*60}\n"

    out_path = os.path.join(OUT_DIR, 'results_validation.txt')
    with open(out_path, 'w') as f:
        f.write(full_report)
    print(full_report)
    print(f"\nSaved -> {out_path}")

if __name__ == '__main__':
    main()
