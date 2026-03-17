import pandas as pd
import os
import re
from pathlib import Path

def normalize_name(name):
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z0-9\\s\\-\\+]', '', name)
    name = re.sub(r'\\s+', ' ', name)
    return name

def merge_ctd_into_raw(raw_path, ctd_path, output_path):
    raw_df = pd.read_csv(raw_path)
    ctd_df = pd.read_csv(ctd_path)

    # Identify which column holds the drug name in the CTD file
    # CTD downloads use 'ChemicalName'; your local file uses 'Drug Name'
    name_col = 'ChemicalName' if 'ChemicalName' in ctd_df.columns else ('Chemical Name' if 'Chemical Name' in ctd_df.columns else 'Drug Name')

    raw_df['norm_name'] = raw_df['Drug Name/Treatment'].apply(normalize_name)
    ctd_df['norm_name'] = ctd_df[name_col].apply(normalize_name)

    # Remove CTD drugs already present in raw
    raw_names = set(raw_df['norm_name'].dropna().unique())
    ctd_df = ctd_df[~ctd_df['norm_name'].isin(raw_names)].copy()

    # STEP 1: rename the name column FIRST, before adding any new columns
    ctd_df = ctd_df.rename(columns={name_col: 'Drug Name/Treatment'})

    # STEP 2: now safely add any missing schema columns
    if 'Drug Structure' not in ctd_df.columns:
        smiles_col = 'SMILES' if 'SMILES' in ctd_df.columns else None
        ctd_df['Drug Structure'] = ctd_df[smiles_col] if smiles_col else ''
    ctd_df['Current Status'] = 'CTD-derived'
    ctd_df['Source'] = 'CTD'

    # STEP 3: keep only the columns that match drugs_raw.csv schema
    keep_cols = ['Drug Name/Treatment', 'Drug Structure', 'Current Status', 'norm_name']
    ctd_df = ctd_df[[c for c in keep_cols if c in ctd_df.columns]]

    # Align columns before concat so missing ones are filled with NaN
    merged = pd.concat([raw_df, ctd_df], ignore_index=True, sort=False)
    merged.drop(columns=['norm_name'], inplace=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged drug file: {output_path}")
    print(f"Raw drugs: {len(raw_df)}, new CTD drugs added: {len(ctd_df)}")

if __name__ == '__main__':
    merge_ctd_into_raw(
        '00_Raw_Data/drugs_raw.csv',
        '01_Cleaned_Data/CTD_D000544_chemicals_20260315024131.csv',
        '00_Raw_Data/drugs_raw_augmented.csv'
    )