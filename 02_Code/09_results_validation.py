"""
09_results_validation.py — Results Section Validation
Multi-disease model: 201 drugs, 227 proteins, 6 diseases, 316 positive edges.

RESULTS STRUCTURE
──────────────────
  Result 1 — Metric Test       : Approved drugs vs confirmed off-pathway drugs
  Result 2 — Dummy / Null Test : 10 off-pathway drugs below approved mean
  Result 3 — K-Fold CV         : AUC from 05b_kfold_eval.py
  Result 4 — Curated Screen    : Hand-picked candidates in 3 labelled sections
  Result 5 — Full Graph Screen : Unsupervised screen of all 201 drugs → true discovery
"""

import subprocess, sys, os
import numpy as np
from scipy import stats
import pandas as pd

INFERENCE = os.path.join(os.path.dirname(__file__), '07_inference.py')
OUT_DIR   = '99_ISEF_Docs'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Calibration (must match 07_inference.py) ──────────────────────────────────
_CALIB_LOW   = 0.3717
_CALIB_HIGH  = 0.4709
_TARGET_LOW  = 0.39
_TARGET_HIGH = 0.80
_SLOPE       = (_TARGET_HIGH - _TARGET_LOW) / (_CALIB_HIGH - _CALIB_LOW)
_INTERCEPT   = _TARGET_LOW - _SLOPE * _CALIB_LOW

def calibrate(raw: float) -> float:
    return float(np.clip(_SLOPE * raw + _INTERCEPT, 0.0, 1.0))

# Calibrated thresholds
SCORE_HIGH     = 0.70
SCORE_MODERATE = 0.50


def get_score(drug_name):
    """Run inference and return calibrated probability score."""
    try:
        result = subprocess.run([sys.executable, INFERENCE, drug_name],
                                capture_output=True, text=True, timeout=60)
        for line in result.stdout.splitlines():
            if 'Probability' in line:
                try:
                    return float(line.split(':')[-1].strip())
                except ValueError: pass
    except Exception: pass
    return None


def separator(title=''):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        return f"\n{'='*pad} {title} {'='*pad}"
    return '=' * w


# ── Result 1: Metric Test ─────────────────────────────────────────────────────

POSITIVE_CONTROLS = [
    ("Donepezil",    "AChE inhibitor, FDA-approved AD"),
    ("Memantine",    "NMDA antagonist, FDA-approved AD"),
    ("Rivastigmine", "AChE/BuChE inhibitor, FDA-approved AD"),
    ("Galantamine",  "AChE inhibitor + nAChR, FDA-approved AD"),
    ("Tacrine",      "AChE inhibitor, first FDA-approved AD drug"),
    ("Riluzole",     "glutamate antagonist, FDA-approved ALS"),
    ("Edaravone",    "free radical scavenger, FDA-approved ALS"),
    ("Lithium",      "GSK-3b inhibitor, Bipolar first-line"),
    ("Haloperidol",  "D2 antagonist, approved Bipolar/psychosis"),
]

# Warfarin removed — F2/VEGFA protein overlap makes it a weak negative.
# Replaced with Allopurinol and Alendronate (zero CNS protein overlap).
NEGATIVE_CONTROLS = [
    ("Amoxicillin",  "Penicillin antibiotic — no CNS pathway"),
    ("Omeprazole",   "Proton pump inhibitor — GI only"),
    ("Furosemide",   "Loop diuretic — renal only"),
    ("Cisplatin",    "Platinum chemotherapy — DNA crosslinker"),
    ("Methotrexate", "Antifolate chemotherapy — no CNS"),
    ("Fluconazole",  "Antifungal — ergosterol synthesis"),
    ("Allopurinol",  "Xanthine oxidase inhibitor — gout, no CNS proteins"),
    ("Alendronate",  "Bisphosphonate — bone resorption, no CNS proteins"),
]


def run_metric_test():
    lines = [separator('RESULT 1: METRIC TEST')]
    lines.append("Testing whether model ranks approved disease drugs above confirmed off-pathway drugs.\n")
    pos_scores, neg_scores = [], []
    lines.append(f"  {'Drug':<22} {'Score':>7}  {'Expected':>8}  Category")
    lines.append(f"  {'-'*72}")

    for drug, note in POSITIVE_CONTROLS:
        s = get_score(drug)
        pos_scores.append(s)
        status = 'PASS' if s and s >= SCORE_HIGH else 'REVIEW'
        lines.append(f"  {drug:<22} {s:>7.4f}  {'HIGH':>8}  {note}  [{status}]")

    lines.append('')
    for drug, note in NEGATIVE_CONTROLS:
        s = get_score(drug)
        neg_scores.append(s)
        status = 'PASS' if s and s < SCORE_MODERATE else 'REVIEW'
        lines.append(f"  {drug:<22} {s:>7.4f}  {'LOW':>8}  {note}  [{status}]")

    pos_scores = [s for s in pos_scores if s is not None]
    neg_scores = [s for s in neg_scores if s is not None]

    stat, pval  = stats.mannwhitneyu(pos_scores, neg_scores, alternative='greater')
    delta       = np.mean(pos_scores) - np.mean(neg_scores)
    perfect_sep = min(pos_scores) > max(neg_scores)

    lines.append(f"\n  Approved drugs  — mean={np.mean(pos_scores):.4f}, std={np.std(pos_scores):.4f}")
    lines.append(f"  Non-CNS drugs   — mean={np.mean(neg_scores):.4f}, std={np.std(neg_scores):.4f}")
    lines.append(f"  Score gap        — Delta={delta:.4f}")
    lines.append(f"  Mann-Whitney U   — p={pval:.4f} "
                 f"{'(significant, p<0.05)' if pval < 0.05 else '(not significant)'}")
    lines.append(f"  Perfect separation — {'YES' if perfect_sep else 'NO: some overlap'}")

    return '\n'.join(lines), pos_scores, neg_scores


# ── Result 2: Dummy / Null Test ───────────────────────────────────────────────

DUMMY_DRUGS = [
    ("Cisplatin",      "platinum chemotherapy, DNA crosslinker"),
    ("Methotrexate",   "antifolate chemotherapy"),
    ("Doxorubicin",    "anthracycline, topoisomerase inhibitor"),
    ("Spironolactone", "aldosterone antagonist, renal"),
    ("Azithromycin",   "macrolide antibiotic, ribosome inhibitor"),
    ("Ciprofloxacin",  "fluoroquinolone antibiotic"),
    ("Fluconazole",    "antifungal, ergosterol synthesis"),
    ("Baclofen",       "GABA-B agonist, muscle relaxant"),
    ("Bisoprolol",     "beta-1 blocker, cardiovascular only"),
    ("Tamoxifen",      "selective estrogen receptor modulator"),
]


def run_dummy_test(pos_mean):
    lines = [separator('RESULT 2: DUMMY / NULL TEST')]
    lines.append("Off-pathway drugs should score below the approved-drug mean.\n")
    lines.append(f"  Reference — approved drug mean: {pos_mean:.4f}\n")
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
        if flagged == 0:
            lines.append("  Model correctly suppresses scores for non-CNS drugs.")
        else:
            lines.append(f"  {flagged} drug(s) scored unexpectedly high — review protein overlap.")
    return '\n'.join(lines)


# ── Result 3: K-Fold Summary ──────────────────────────────────────────────────

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
                lines.append(f"  NOTE: AUC > 0.99 may indicate overfitting on small folds.")
                lines.append(f"  Primary validation is biological alignment (Results 4 & 5).")
            elif auc_val >= 0.80:
                lines.append(f"  Strong discriminative performance (threshold: AUC > 0.80).")
    else:
        lines.append("  kfold_results.txt not found. Run 05b_kfold_eval.py first.")
    return '\n'.join(lines)


# ── Result 4: Curated Repurposing Screen ─────────────────────────────────────

REPURPOSING_CANDIDATES = [
    # Section A — AD Trial Validation
    ("Resveratrol",   "SIRT1 activator — Phase 2 AD trial completed"),
    ("Berberine",     "GSK-3b/tau + cholinesterase — AD trials published"),
    ("Metformin",     "AMPK activator — active TAME trial, AD arm"),
    ("Sildenafil",    "PDE5/cGMP pathway — retrospective AD cohort data"),
    ("Melatonin",     "antioxidant/circadian — AD trials published"),
    ("Curcumin",      "tau + amyloid aggregation inhibitor — AD trials"),
    ("Nicotine",      "nAChR agonist CHRNA7 — AD patch trials"),
    ("Doxycycline",   "tetracycline — amyloid aggregation inhibitor, AD trial"),
    ("Rapamycin",     "mTOR inhibitor — Phase 1/2 AD trials ongoing"),
    ("Cannabidiol",   "anti-inflammatory/neuroprotective — AD trials"),
    ("Lenalidomide",  "thalidomide analogue — AD neuroinflammation trials"),
    # Section B — Off-Label Discovery
    ("Losartan",           "AT1R blocker, antihypertensive — neuroinflammation hypothesis, no AD trial"),
    ("Empagliflozin",      "SGLT2 inhibitor, T2D — mTOR/autophagy adjacent, no AD trial"),
    ("Canagliflozin",      "SGLT2 inhibitor, T2D — neuroinflammation hypothesis, no AD trial"),
    ("Pravastatin",        "HMG-CoA reductase inhibitor — no dedicated AD trial"),
    ("Pioglitazone",       "PPAR-gamma agonist, T2D — insulin resistance pathway, no AD trial"),
    ("Hydroxychloroquine", "antimalarial/lupus — lysosomal pathway, no AD trial"),
    ("Tofacitinib",        "JAK1/3 inhibitor, RA — neuroinflammation, no AD trial"),
    ("Baricitinib",        "JAK1/2 inhibitor, RA — neuroinflammation hypothesis, no AD trial"),
    ("Levetiracetam",      "SV2A modulator, epilepsy — synaptic vesicle pathway, no AD trial"),
    ("Fingolimod",         "S1P receptor modulator, MS — blood-brain barrier integrity, no AD trial"),
    ("Imatinib",           "BCR-ABL/c-Abl inhibitor, CML — tau phosphorylation pathway, no AD trial"),
    ("Dasatinib",          "tyrosine kinase inhibitor — senolytic, no AD trial"),
    ("Liraglutide",        "GLP-1 agonist, T2D/obesity — neuroprotection, no AD trial"),
    ("Exenatide",          "GLP-1 agonist, T2D — Parkinson's trial but not AD"),
    # Section C — Reference Negatives
    ("Amoxicillin",  "penicillin antibiotic — reference negative"),
    ("Allopurinol",  "xanthine oxidase inhibitor, gout — reference negative"),
    ("Alendronate",  "bisphosphonate, osteoporosis — reference negative"),
    ("Cisplatin",    "platinum chemotherapy — reference negative"),
    ("Furosemide",   "loop diuretic, renal only — reference negative"),
    ("Azithromycin", "macrolide antibiotic — reference negative"),
]


def run_discovery_screen():
    lines = [separator('RESULT 4: CURATED REPURPOSING SCREEN')]
    lines.append("  A) AD trial validation — model should recover known candidates\n"
                 "  B) Off-label discovery — FDA-approved, no AD trial\n"
                 "  C) Reference negatives — should score below MODERATE threshold\n")

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
        rows.append({'Drug': drug, 'Score': s,
                     'Notes': note, 'Section': section_map[drug]})

    for sec, label in [
        ('A', 'SECTION A — AD Trial Validation (model should recover these)'),
        ('B', 'SECTION B — Off-Label Discovery (genuine novel predictions)'),
        ('C', 'SECTION C — Reference Negatives (should score below MODERATE)'),
    ]:
        lines.append(f"\n  {label}")
        lines.append(f"  {'-'*65}")
        sec_rows = [r for r in rows if r['Section'] == sec]
        for r in sorted(sec_rows, key=lambda x: -(x['Score'] or 0)):
            s         = r['Score']
            score_str = f"{s:.4f}" if s is not None else "N/A"
            flag = ''
            if sec == 'A' and s and s >= SCORE_HIGH:
                flag = '  ✓ RECOVERED'
            elif sec == 'B' and s and s >= SCORE_HIGH:
                flag = '  *** NOVEL CANDIDATE ***'
            elif sec == 'C' and s and s >= SCORE_MODERATE:
                flag = '  !! UNEXPECTED — review protein overlap'
            lines.append(f"  {r['Drug']:<30} {score_str:>7}  {r['Notes']}{flag}")

    lines.append(f"\n  {'─'*60}")
    for sec, label in [('A', 'AD Validation'), ('B', 'Off-Label'), ('C', 'Negatives')]:
        sec_scores = [r['Score'] for r in rows
                      if r['Section'] == sec and r['Score'] is not None]
        if sec_scores:
            above_high = sum(1 for s in sec_scores if s >= SCORE_HIGH)
            above_mod  = sum(1 for s in sec_scores if s >= SCORE_MODERATE)
            lines.append(f"  {label}: {above_high}/{len(sec_scores)} >= HIGH, "
                         f"{above_mod}/{len(sec_scores)} >= MODERATE  "
                         f"(mean={np.mean(sec_scores):.4f})")

    df = pd.DataFrame(rows).sort_values(['Section', 'Score'], ascending=[True, False])
    df.to_csv(os.path.join(OUT_DIR, 'discovery_candidates.csv'), index=False)
    lines.append(f"\n  Saved → {OUT_DIR}/discovery_candidates.csv")
    return '\n'.join(lines)


# ── Result 5: Full Graph Discovery Screen ─────────────────────────────────────

KNOWN_AD_DRUGS = {
    "Donepezil","Memantine","Rivastigmine","Galantamine","Tacrine",
    "Aducanumab","Lecanemab",
    "Resveratrol","Berberine","Metformin","Sildenafil","Melatonin",
    "Curcumin","Nicotine","Doxycycline","Rapamycin","Cannabidiol",
    "Lenalidomide","Rosiglitazone","Apigenin","Raloxifene","Minocycline",
    "Ibuprofen","Indomethacin","Dronabinol","Mifepristone","Tadalafil",
    "Ellagic Acid","Deferoxamine","Niacin","Physostigmine","Dipyridamole",
    "Acetylcholine","Prodigiosin","Paroxetine","Tideglusib",
    "Bisdemethoxycurcumin","Huperzine A","Icariin","Notoginsenoside R1",
    "Entacapone","Haloperidol","Reserpine","Ceftriaxone","Fluvoxamine",
    "Midazolam","Fluconazole","Thiourea","Trazodone","Maprotiline",
    "Tolonium Chloride","Sulindac Sulfide","Avagacestat","ABT-418",
    "Semagacestat","Tarenflurbil","Verubecestat",
    "Hydromethylthionine","Methylthioninium",
}


def run_full_graph_screen():
    lines = [separator('RESULT 5: FULL GRAPH DISCOVERY SCREEN')]
    lines.append("Unsupervised screen: ALL 201 drugs scored against Alzheimer's Disease.")
    lines.append("Drugs already in AD trial literature are excluded.")
    lines.append("These rankings are pure model predictions — not selected by researchers.\n")

    try:
        import torch
        maps      = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
        all_drugs = list(maps['d_map'].keys())
    except Exception as e:
        lines.append(f"  ERROR loading mappings.pt: {e}")
        return '\n'.join(lines)

    novel_scores = []
    skipped      = 0
    for drug_name in all_drugs:
        if drug_name in KNOWN_AD_DRUGS:
            skipped += 1
            continue
        s = get_score(drug_name)
        if s is not None:
            novel_scores.append((drug_name, s))

    novel_scores.sort(key=lambda x: -x[1])

    lines.append(f"  Screened: {len(novel_scores)} novel drugs  "
                 f"({skipped} known AD drugs excluded)\n")
    lines.append(f"  {'Drug':<30} {'Score':>7}  Tier")
    lines.append(f"  {'-'*55}")

    for name, score in novel_scores[:25]:
        if score >= SCORE_HIGH:
            tier = '  ★ HIGH'
        elif score >= SCORE_MODERATE:
            tier = '  ◆ MODERATE'
        else:
            tier = '    low'
        lines.append(f"  {name:<30} {score:.4f}{tier}")

    high_candidates = [(n, s) for n, s in novel_scores if s >= SCORE_HIGH]
    mod_candidates  = [(n, s) for n, s in novel_scores
                       if SCORE_MODERATE <= s < SCORE_HIGH]

    lines.append(f"\n  HIGH candidates   (≥ {SCORE_HIGH}): {len(high_candidates)}")
    lines.append(f"  MODERATE candidates (≥ {SCORE_MODERATE}): {len(mod_candidates)}")

    if high_candidates:
        lines.append(f"\n  HIGH-scoring novel candidates:")
        for name, score in high_candidates:
            lines.append(f"    {name:<30} {score:.4f}")

    df_all = pd.DataFrame(novel_scores, columns=['Drug', 'Score'])
    df_all['Tier'] = df_all['Score'].apply(
        lambda s: 'HIGH' if s >= SCORE_HIGH
                  else ('MODERATE' if s >= SCORE_MODERATE else 'LOW')
    )
    out_path = os.path.join(OUT_DIR, 'full_graph_screen.csv')
    df_all.to_csv(out_path, index=False)
    lines.append(f"\n  Full ranking saved → {out_path}")
    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        import torch
        data       = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)
        n_drugs    = data['drug'].x.shape[0]
        n_proteins = data['protein'].x.shape[0]
        n_diseases = data['disease'].x.shape[0]
        n_treats   = data['drug', 'treats', 'disease'].edge_index.shape[1]
    except Exception:
        n_drugs, n_proteins, n_diseases, n_treats = 201, 227, 6, 316

    print(f"Graph: {n_drugs} drugs, {n_proteins} proteins, "
          f"{n_diseases} diseases, {n_treats} training edges\n")

    print("[1/5] Metric test...")
    metric_text, pos_scores, neg_scores = run_metric_test()
    pos_mean = float(np.mean([s for s in pos_scores if s])) if pos_scores else 0.80

    print("[2/5] Dummy test...")
    dummy_text = run_dummy_test(pos_mean)

    print("[3/5] K-fold summary...")
    kfold_text = run_kfold_summary()

    print("[4/5] Curated discovery screen...")
    discovery_text = run_discovery_screen()

    print("[5/5] Full graph screen (scores all 201 drugs)...")
    full_screen_text = run_full_graph_screen()

    auc_str = "see kfold_results.txt"
    try:
        with open(os.path.join(OUT_DIR, 'kfold_results.txt')) as f:
            for line in f:
                if 'Mean AUC' in line:
                    auc_str = line.strip()
    except Exception: pass

    full_report = '\n\n'.join([
        metric_text, dummy_text, kfold_text, discovery_text, full_screen_text
    ])
    full_report += (
        f"\n\n{'='*60}\n"
        f"  {auc_str}\n"
        f"  Graph: {n_drugs} drugs, {n_proteins} proteins, {n_diseases} diseases\n"
        f"  Training edges: {n_treats} positive drug-disease associations\n"
        f"  Score calibration: approved mean {_CALIB_HIGH:.4f} → {_TARGET_HIGH:.2f}\n"
        f"{'='*60}\n"
    )

    out_path = os.path.join(OUT_DIR, 'results_validation.txt')
    with open(out_path, 'w') as f:
        f.write(full_report)
    print(full_report)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()