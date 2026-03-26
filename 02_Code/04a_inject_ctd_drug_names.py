import pandas as pd
import os
import re
from pathlib import Path

# ── Name corrections — must match 01_clean_drugs.py exactly ──────────────────
NAME_CORRECTIONS = {
    "N2-((2S)-2-(3,5-difluorophenyl)-2-hydroxyethanoyl)-N1-((7S)-5-methyl-6-oxo-6,7-dihydro-5H-dibenzo(b,d)azepin-7-yl)-L-alaninamide": "Semagacestat",
    "3-methyl-5-(1-methyl-2-pyrrolidinyl)isoxazole": "ABT-418",
    "BMS 708163": "Avagacestat",
    "Vitamin D": "Cholecalciferol",
    "Raloxifene Hydrochloride": "Raloxifene",
    "Thioctic Acid": "Lipoic acid",
    "Vitamin E": "Tocopherol",
    "2-(4-morpholino)ethyl-1-phenylcyclohexane-1-carboxylate": "PRE-084",
    "kenpaullone": "Kenpaullone",
    "Lithium Chloride": "Lithium",
    "Lithium carbonate": "Lithium",
    "Quetiapine Fumarate": "Quetiapine",
    "Acetylcysteine": "N-Acetylcysteine",
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


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for dedup matching."""
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z0-9\s\-\+]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def apply_corrections(name: str) -> str:
    """Apply NAME_CORRECTIONS then normalize."""
    corrected = NAME_CORRECTIONS.get(str(name).strip(), str(name).strip())
    return corrected


def merge_one_ctd_file(existing_names: set, ctd_path: str,
                       name_col: str, disease_label: str) -> pd.DataFrame:
    """
    Read one CTD file and return only the rows whose drug names
    are not already in existing_names.

    name_col     — column in the CTD file that holds the drug name
    disease_label — e.g. 'ALS', 'Bipolar', 'Dementia'
    """
    if not os.path.exists(ctd_path):
        print(f"  WARNING: {ctd_path} not found — skipping.")
        return pd.DataFrame()

    df = pd.read_csv(ctd_path)

    if name_col not in df.columns:
        # Fall back: try the first column
        name_col = df.columns[0]
        print(f"  WARNING: expected name column not found, using '{name_col}'")

    # Apply corrections then normalize for dedup check
    df['_corrected'] = df[name_col].apply(apply_corrections)
    df['_norm']      = df['_corrected'].apply(normalize_name)

    # Drop entries already in the augmented file
    new_df = df[~df['_norm'].isin(existing_names)].copy()

    if new_df.empty:
        print(f"  {disease_label}: all drugs already present — nothing to add.")
        return pd.DataFrame()

    # Build rows in drugs_raw schema
    out = pd.DataFrame()
    out['Drug Name/Treatment'] = new_df['_corrected'].values
    out['Drug Structure']      = ''          # SMILES filled later by 01_clean_drugs.py
    out['Current Status']      = 'CTD-derived'
    out['Source']              = f'CTD-{disease_label}'
    out['Targeted protein']    = ''
    out['Numerical_Vector']    = ''

    print(f"  {disease_label}: {len(new_df)} new drugs added")
    return out


def build_augmented_file(raw_path: str, ctd_files: list, output_path: str) -> None:
    """
    Start from raw_path (original drugs_raw.csv or existing augmented file),
    then append new drugs from each CTD file in ctd_files.

    ctd_files format:
        [
            (path, name_column, disease_label),
            ...
        ]
    """
    # Load the base raw file
    raw_df = pd.read_csv(raw_path)
    raw_df['_norm'] = raw_df['Drug Name/Treatment'].apply(
        lambda n: normalize_name(apply_corrections(n))
    )
    existing_names = set(raw_df['_norm'].dropna())
    print(f"Base file: {raw_path} — {len(raw_df)} drugs")

    # Merge each CTD file
    all_new = [raw_df]
    for ctd_path, name_col, label in ctd_files:
        new_rows = merge_one_ctd_file(existing_names, ctd_path, name_col, label)
        if not new_rows.empty:
            # Update existing_names so subsequent files don't re-add the same drug
            new_norms = new_rows['Drug Name/Treatment'].apply(
                lambda n: normalize_name(str(n))
            )
            existing_names.update(new_norms)
            all_new.append(new_rows)

    merged = pd.concat(all_new, ignore_index=True, sort=False)
    merged.drop(columns=['_norm'], errors='ignore', inplace=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    n_added = len(merged) - len(raw_df)
    print(f"\nSaved → {output_path}")
    print(f"Original: {len(raw_df)} drugs")
    print(f"Added:    {n_added} new drugs")
    print(f"Total:    {len(merged)} drugs")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # The base file to start from.
    # Use drugs_raw_augmented.csv if it already exists (from a previous run),
    # otherwise start fresh from drugs_raw.csv.
    base = ('00_Raw_Data/drugs_raw_augmented.csv'
            if os.path.exists('00_Raw_Data/drugs_raw_augmented.csv')
            else '00_Raw_Data/drugs_raw.csv')

    # List every CTD source file here.
    # Format: (file_path, drug_name_column, disease_label)
    ctd_sources = [
        # Original Alzheimer's file
        (
            '01_Cleaned_Data/CTD_D000544_chemicals_20260315024131.csv',
            'ChemicalName',
            'AD'
        ),
        # Existing disease files
        (
            '00_Raw_Data/positive_drugs_als.csv',
            'name',
            'ALS',
        ),
        (
            '00_Raw_Data/positive_drugs_bipolar.csv',
            'name',
            'Bipolar',
        ),
        (
            '00_Raw_Data/positive_drugs_dementia.csv',
            'name',
            'Dementia',
        ),
        # New disease files from full CTD chemical query
        (
            '00_Raw_Data/positive_drugs_ad.csv',
            'name',
            'AD-full',
        ),
        (
            '00_Raw_Data/positive_drugs_adhd.csv',
            'name',
            'ADHD',
        ),
        (
            '00_Raw_Data/positive_drugs_parkinsons.csv',
            'name',
            'Parkinsons',
        ),
    ]

    build_augmented_file(
        raw_path    = base,
        ctd_files   = ctd_sources,
        output_path = '00_Raw_Data/drugs_raw_augmented.csv',
    )

    print("\nNext step: run 01_clean_drugs.py to fetch SMILES for all new drugs.")