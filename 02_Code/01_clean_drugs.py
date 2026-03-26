import pandas as pd
import os
import logging

try:
    import pubchempy as pcp
except ModuleNotFoundError:
    pcp = None

_MISSING_PUBCHEMPY_WARNED = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ── Name corrections applied before PubChem lookup ───────────────────────────
# Maps CTD chemical names → PubChem-searchable names.
# Add new entries here whenever a new CTD file introduces unusual names.
NAME_CORRECTIONS = {
    # Alzheimer's (original)
    "N2-((2S)-2-(3,5-difluorophenyl)-2-hydroxyethanoyl)-N1-((7S)-5-methyl-6-oxo-6,7-dihydro-5H-dibenzo(b,d)azepin-7-yl)-L-alaninamide": "Semagacestat",
    "3-methyl-5-(1-methyl-2-pyrrolidinyl)isoxazole": "ABT-418",
    "BMS 708163": "Avagacestat",
    "Vitamin D": "Cholecalciferol",
    "Raloxifene Hydrochloride": "Raloxifene",
    # ALS
    "Thioctic Acid": "Lipoic acid",
    "Vitamin E": "Tocopherol",
    "2-(4-morpholino)ethyl-1-phenylcyclohexane-1-carboxylate": "PRE-084",
    "kenpaullone": "Kenpaullone",
    # Bipolar
    "Lithium Chloride": "Lithium",
    "Lithium carbonate": "Lithium",
    "Quetiapine Fumarate": "Quetiapine",
    "Acetylcysteine": "N-Acetylcysteine",
    # Dementia
    "ginsenoside Rg1": "Ginsenoside Rg1",
    "Aspirin": "Acetylsalicylic acid",
    "Acetylsalicylic acid (aspirin)": "Acetylsalicylic acid",

    # AD (new CTD file)
    "6,7-dihydroxyflavone":               "6,7-Dihydroxyflavone",
    "bisdemethoxycurcumin":               "Bisdemethoxycurcumin",
    "huperzine A":                        "Huperzine A",
    "icariin":                            "Icariin",
    "notoginsenoside R1":                 "Notoginsenoside R1",
    "puag-haad":                          "Puag-haad",
    "sulindac sulfide":                   "Sulindac Sulfide",
    "tideglusib":                         "Tideglusib",
    "entacapone":                         "Entacapone",
    # Parkinson's (new CTD file)
    "4-phenylbutyric acid":               "4-Phenylbutyric Acid",
    "mangiferin":                         "Mangiferin",
    "nardosinone":                        "Nardosinone",
    "nootkatone":                         "Nootkatone",
    "rasagiline":                         "Rasagiline",
    "ropinirole":                         "Ropinirole",
    "rotigotine":                         "Rotigotine",
    "safinamide":                         "Safinamide",
    # ADHD (new CTD file)
    "Venlafaxine Hydrochloride":          "Venlafaxine",
    "ginsenoside Rg3":                    "Ginsenoside Rg3",
    "pozanicline":                        "Pozanicline",
}


def get_smiles(drug_name: str) -> str | None:
    """Fetch canonical SMILES from PubChem by drug name."""
    global _MISSING_PUBCHEMPY_WARNED
    if pcp is None:
        if not _MISSING_PUBCHEMPY_WARNED:
            logging.warning(
                "pubchempy is not installed; skipping PubChem lookup for missing SMILES. "
                "Install with: pip install pubchempy"
            )
            _MISSING_PUBCHEMPY_WARNED = True
        return None
    try:
        results = pcp.get_compounds(drug_name, 'name')
        if results:
            # connectivity_smiles is the new canonical; fall back to canonical_smiles
            smiles = getattr(results[0], 'connectivity_smiles', None) \
                  or getattr(results[0], 'canonical_smiles', None)
            return smiles
    except Exception as e:
        logging.error(f"Error fetching SMILES for {drug_name}: {e}")
    return None


def clean_drug_list(input_path: str, output_path: str) -> None:
    """
    Read a drug CSV, apply name corrections, fetch missing SMILES from PubChem,
    and write the cleaned file to output_path.

    Supported input column layouts:
      - 'name' + optional 'smiles'        (positive_drugs_ctd.csv style)
      - 'Drug Name/Treatment' + 'Drug Structure'  (drugs_raw_augmented.csv style)
    """
    logging.info(f"Processing {input_path}...")
    df = pd.read_csv(input_path)

    # Detect which columns hold the drug name and SMILES
    if 'name' in df.columns:
        name_col  = 'name'
        smiles_col = 'smiles'
    elif 'Drug Name/Treatment' in df.columns:
        name_col  = 'Drug Name/Treatment'
        smiles_col = 'Drug Structure'
    else:
        logging.error(f"Cannot find name column in {input_path}. Skipping.")
        return

    # Ensure smiles column exists
    if smiles_col not in df.columns:
        df[smiles_col] = None
    # pandas may infer float dtype for mostly-missing columns; force object for SMILES strings
    df[smiles_col] = df[smiles_col].astype(object)

    for idx, row in df.iterrows():
        # Apply name correction
        raw_name    = str(row[name_col]).strip()
        lookup_name = NAME_CORRECTIONS.get(raw_name, raw_name)

        current_smiles = str(row.get(smiles_col, '')).strip()
        if current_smiles in ('', 'nan', 'None'):
            logging.info(f"Fetching SMILES for {lookup_name}...")
            smiles = get_smiles(lookup_name)
            if smiles:
                df.at[idx, smiles_col] = smiles
                logging.info(f"  Found SMILES for {lookup_name}")
            else:
                logging.warning(f"  Could not find SMILES for {lookup_name}")

    df.to_csv(output_path, index=False)
    logging.info(f"Saved → {output_path}")


def merge_drug_links() -> None:
    """
    Combine drug_links files for all diseases into a single
    01_Cleaned_Data/drug_links.csv.
    Deduplicates by (drug_name, protein_target) keeping the highest score.
    """
    link_files = [
        '00_Raw_Data/drug_links.csv',          # original AD file
        '00_Raw_Data/drug_links_als.csv',
        '00_Raw_Data/drug_links_bipolar.csv',
        '00_Raw_Data/drug_links_dementia.csv',
        '00_Raw_Data/drug_links_ad.csv',       # new full AD CTD file
        '00_Raw_Data/drug_links_adhd.csv',     # new ADHD CTD file
        '00_Raw_Data/drug_links_parkinsons.csv', # new Parkinson's CTD file
    ]

    frames = []
    for path in link_files:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
            logging.info(f"Loaded drug links: {path}")
        else:
            logging.warning(f"Drug links file not found, skipping: {path}")

    if not frames:
        logging.error("No drug_links files found.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Normalize drug names using NAME_CORRECTIONS
    combined['drug_name'] = combined['drug_name'].apply(
        lambda n: NAME_CORRECTIONS.get(str(n).strip(), str(n).strip())
    )

    # Deduplicate: keep highest inference_score per (drug, protein) pair
    combined = (combined
                .sort_values('inference_score', ascending=False)
                .drop_duplicates(subset=['drug_name', 'protein_target'])
                .reset_index(drop=True))

    out_path = '01_Cleaned_Data/drug_links.csv'
    combined.to_csv(out_path, index=False)
    logging.info(f"Merged drug links: {len(combined)} rows → {out_path}")


def merge_positive_drugs() -> None:
    """
    Combine positive drug files for all diseases into a single
    01_Cleaned_Data/positive_drugs.csv.
    Each file must have columns: name, label, smiles (optional).
    """
    positive_files = [
        '01_Cleaned_Data/positive_drugs.csv',           # original AD (already cleaned)
        '01_Cleaned_Data/positive_drugs_als.csv',
        '01_Cleaned_Data/positive_drugs_bipolar.csv',
        '01_Cleaned_Data/positive_drugs_dementia.csv',
        '01_Cleaned_Data/positive_drugs_ad.csv',        # new full AD CTD file
        '01_Cleaned_Data/positive_drugs_adhd.csv',      # new ADHD CTD file
        '01_Cleaned_Data/positive_drugs_parkinsons.csv', # new Parkinson's CTD file
    ]

    frames = []
    for path in positive_files:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
            logging.info(f"Loaded positive drugs: {path}")
        else:
            logging.warning(f"Positive drugs file not found, skipping: {path}")

    if not frames:
        logging.error("No positive drug files found.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Normalize names
    combined['name'] = combined['name'].apply(
        lambda n: NAME_CORRECTIONS.get(str(n).strip(), str(n).strip())
    )

    # Deduplicate by name — keep first occurrence (preserves SMILES if already fetched)
    combined = combined.drop_duplicates(subset=['name']).reset_index(drop=True)

    out_path = '01_Cleaned_Data/positive_drugs.csv'
    combined.to_csv(out_path, index=False)
    logging.info(f"Merged positive drugs: {len(combined)} entries → {out_path}")


def _has_supported_drug_columns(csv_path: str) -> bool:
    """Check whether CSV has one of the supported name/smiles column layouts."""
    try:
        cols = set(pd.read_csv(csv_path, nrows=0).columns)
    except Exception:
        return False
    return ('name' in cols) or ('Drug Name/Treatment' in cols)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs('01_Cleaned_Data', exist_ok=True)

    ctd_primary = '00_Raw_Data/positive_drugs_ctd.csv'
    ctd_fallback = '00_Raw_Data/postive_drugs_ctd.csv'
    ctd_input = ctd_primary if os.path.exists(ctd_primary) else ctd_fallback
    if ctd_input == ctd_fallback and os.path.exists(ctd_fallback):
        logging.warning(
            "Using fallback file with historical typo: "
            "00_Raw_Data/postive_drugs_ctd.csv"
        )

    # ── Step 1: Clean original Alzheimer's positive drugs ────────────────────
    if os.path.exists(ctd_input):
        if _has_supported_drug_columns(ctd_input):
            clean_drug_list(
                ctd_input,
                '01_Cleaned_Data/positive_drugs.csv'
            )
        else:
            logging.warning(
                f"Skipping {ctd_input}: unsupported columns for drug cleaning."
            )
    else:
        logging.warning(
            "CTD positive drugs file not found; expected one of: "
            "00_Raw_Data/positive_drugs_ctd.csv or "
            "00_Raw_Data/postive_drugs_ctd.csv"
        )

    # ── Step 2: Clean all disease-specific drug files ───────────────────────
    for disease in ['als', 'bipolar', 'dementia', 'ad', 'adhd', 'parkinsons']:
        raw_path = f'00_Raw_Data/positive_drugs_{disease}.csv'
        out_path = f'01_Cleaned_Data/positive_drugs_{disease}.csv'
        if os.path.exists(raw_path):
            clean_drug_list(raw_path, out_path)
        else:
            logging.warning(f"File not found, skipping: {raw_path}")

    # ── Step 3: Merge all positive drug files into one ───────────────────────
    merge_positive_drugs()

    # ── Step 4: Clean negative controls ──────────────────────────────────────
    clean_drug_list(
        '00_Raw_Data/negative_controls.csv',
        '01_Cleaned_Data/negative_controls.csv'
    )

    # ── Step 5: Fill SMILES for augmented raw drug file ──────────────────────
    if os.path.exists('00_Raw_Data/drugs_raw_augmented.csv'):
        clean_drug_list(
            '00_Raw_Data/drugs_raw_augmented.csv',
            '00_Raw_Data/drugs_raw_augmented.csv'   # overwrite in place
        )
    else:
        logging.warning("drugs_raw_augmented.csv not found — run 04a first.")

    # ── Step 6: Merge all drug_links files into one ───────────────────────────
    merge_drug_links()

    logging.info("\nAll done. Files written to 01_Cleaned_Data/")
    logging.info("Next: run 04a_inject_ctd_drug_names.py for each new disease file,")
    logging.info("      then run 03_build_hetero_graph.py → 04_expand_graph.py")

