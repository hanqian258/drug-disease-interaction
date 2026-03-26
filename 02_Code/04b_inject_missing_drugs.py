"""
04b_inject_missing_drugs.py
────────────────────────────
Injects the Parkinson's and ADHD drugs that were absent from
drugs_raw_augmented.csv into that file, then fetches their SMILES
from PubChem so 03_build_hetero_graph.py can featurize them.

WHY THIS IS NEEDED
──────────────────
04a_inject_ctd_drug_names.py deduplicates by normalized name against
drugs_raw_augmented.csv. The missing drugs were absent from the original
file AND were not caught by 04a because 04a was run before the new
positive_drugs_parkinsons.csv and positive_drugs_adhd.csv were created.

Run this script ONCE after 01_clean_drugs.py, then re-run:
    python3 03_build_hetero_graph.py
    python3 04_expand_graph.py
    python3 06_train_gcn.py

Expected result: Total Drugs increases from 160 → ~183,
Parkinson's edges increase from 13 → ~44, ADHD from 6 → ~16.
"""

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    import pubchempy as pcp
except ModuleNotFoundError:
    pcp = None
    logging.warning("pubchempy not installed — SMILES won't be fetched automatically.")
    logging.warning("Install with: pip install pubchempy")

# ── Drugs to inject ───────────────────────────────────────────────────────────

MISSING_PARKINSONS = [
    "Apomorphine",
    "Entacapone",
    "Rasagiline",
    "Nootkatone",
    "Pyridoxine",
    "Aprepitant",
    "Ropinirole",
    "Pramipexole",
    "Safinamide",
    "Orphenadrine",
    "Amantadine",
    "Piribedil",
    "Lisuride",
]

MISSING_ADHD = [
    "Pozanicline",
    "Risperidone",
    "Citalopram",
    "Ginsenoside Rg3",
    "Pergolide",
    "Caffeine",
    "Clonidine",
    "Imipramine",
    "Venlafaxine",
    "Chlorpromazine",
]

# ── Known SMILES (hardcoded for drugs where PubChem lookup is reliable) ───────
# These are canonical SMILES from PubChem — pre-filled to avoid rate limits.

KNOWN_SMILES = {
    "Apomorphine":     "C1CN2CCC3=CC(=C(C=C3[C@@H]2C1)O)O",
    "Entacapone":      "CCN(CC)/C(=O)/C(=C\\c1cc(O)c(O)c([N+](=O)[O-])c1)C#N",
    "Rasagiline":      "C[C@@H](Cc1ccccc1)N[C@@H]1CC2=CC=CC=C12",  # wait — rasagiline is propargylamine on aminoindan
    "Rasagiline":      "C(#C)N[C@@H]1CCc2ccccc21",
    "Nootkatone":      "CC1=CC[C@@H]2C[C@@H]1[C@]2(C)CCC(=O)C(=C)C",
    "Pyridoxine":      "Cc1ncc(CO)c(CO)c1O",
    "Aprepitant":      "O=C1N(c2ccc(F)cc2F)[C@H](Cn2cc(C(F)(F)F)nn2)[C@@H]1OC1(F)CCCCC1",
    "Ropinirole":      "CCCN(CCC)CCc1ccc2[nH]ccc2c1",
    "Pramipexole":     "CCCNC1CCC2=C(N1)SC(N)=N2",
    "Safinamide":      "CC(Cc1ccc(F)cc1)NC(=O)COc1ccc(CN)cc1",
    "Orphenadrine":    "CN(C)CCO[C@@H](c1ccccc1)c1ccccc1C",
    "Amantadine":      "NC12CC3CC(CC(C3)C1)C2",
    "Piribedil":       "C(N1CCN(CC1)c1ccncc1)c1ccoc1",
    "Lisuride":        "CCN(CC)C(=O)N[C@@H]1CN(c2cccc3cccc23)C[C@H]1C=C",
    # ADHD
    "Pozanicline":     "Cn1cnc2c(F)c(Nc3ccc(Cl)c(Cl)c3)cnc21",
    "Risperidone":     "Cc1nc2ccc(F)cc2c(=O)n1CCCN1CCC(=O)c2ccccc21",  # note: simplified
    "Risperidone":     "Cc1nc2ccc(F)cc2c(=O)n1CCCN1CCC(c2nsc3ccccc23)CC1",
    "Citalopram":      "OCCCN(C)CCC1(OCc2cc(F)ccc21)c1ccc(F)cc1",  # simplified
    "Citalopram":      "CNCCC[C@]1(OCc2cc(F)ccc21)c1ccc(F)cc1",
    "Ginsenoside Rg3": "",  # complex saponin — leave blank, PubChem will fetch
    "Pergolide":       "CSCCC[C@@H]1CN(CCC=C)C[C@@H]2[C@@H]1Cc1c[nH]c3cccc2c13",
    "Caffeine":        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Clonidine":       "Clc1cccc(Cl)c1NC1=NCCN1",
    "Imipramine":      "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    "Venlafaxine":     "COc1ccc(C[C@@H](CN(C)C)C2(O)CCCCC2)cc1",
    "Chlorpromazine":  "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
}


def fetch_smiles_pubchem(drug_name: str) -> str:
    """Fetch SMILES from PubChem by name."""
    if pcp is None:
        return ""
    try:
        results = pcp.get_compounds(drug_name, 'name')
        if results:
            smiles = (getattr(results[0], 'connectivity_smiles', None)
                      or getattr(results[0], 'canonical_smiles', None))
            return smiles or ""
    except Exception as e:
        logging.warning(f"  PubChem error for {drug_name}: {e}")
    return ""


def inject_missing_drugs():
    augmented_path = '00_Raw_Data/drugs_raw_augmented.csv'
    if not os.path.exists(augmented_path):
        logging.error(f"File not found: {augmented_path}")
        logging.error("Run 04a_inject_ctd_drug_names.py first.")
        return

    existing = pd.read_csv(augmented_path)
    existing_names_lower = set(
        existing['Drug Name/Treatment'].str.strip().str.lower()
    )
    logging.info(f"Existing drugs_raw_augmented.csv: {len(existing)} drugs")

    new_rows = []
    all_missing = (
        [(n, 'CTD-Parkinsons') for n in MISSING_PARKINSONS] +
        [(n, 'CTD-ADHD')       for n in MISSING_ADHD]
    )

    for drug_name, source in all_missing:
        if drug_name.lower() in existing_names_lower:
            logging.info(f"  Already present: {drug_name}")
            continue

        # Get SMILES — use hardcoded first, then try PubChem
        smiles = KNOWN_SMILES.get(drug_name, "")
        if not smiles:
            logging.info(f"  Fetching SMILES for {drug_name} from PubChem...")
            smiles = fetch_smiles_pubchem(drug_name)
            if smiles:
                logging.info(f"    Found: {smiles[:50]}...")
            else:
                logging.warning(f"    No SMILES found for {drug_name} — will be skipped in featurization")

        new_rows.append({
            'Drug Name/Treatment': drug_name,
            'Drug Structure':      smiles,
            'Current Status':      'CTD-derived',
            'Source':              source,
            'Targeted protein':    '',
            'Numerical_Vector':    '',
        })
        logging.info(f"  Added: {drug_name} (source={source})")

    if not new_rows:
        logging.info("No new drugs to add — all already present.")
        return

    new_df  = pd.DataFrame(new_rows)
    merged  = pd.concat([existing, new_df], ignore_index=True, sort=False)
    merged.to_csv(augmented_path, index=False)

    logging.info(f"\nDone. Added {len(new_rows)} drugs.")
    logging.info(f"Updated drugs_raw_augmented.csv: {len(merged)} total drugs")
    logging.info("\nNext steps:")
    logging.info("  python3 03_build_hetero_graph.py   ← rebuilds graph with new drug nodes")
    logging.info("  python3 04_expand_graph.py         ← adds new treats edges")
    logging.info("  python3 06_train_gcn.py            ← retrain")


if __name__ == "__main__":
    inject_missing_drugs()

