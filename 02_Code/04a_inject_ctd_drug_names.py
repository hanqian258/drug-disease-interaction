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

    raw_df['norm_name'] = raw_df['Drug Name/Treatment'].apply(normalize_name)
    ctd_df['norm_name'] = ctd_df['Chemical Name'].apply(normalize_name)

    raw_names = set(raw_df['norm_name'].dropna().unique())
    ctd_df = ctd_df[~ctd_df['norm_name'].isin(raw_names)].copy()

    # Keep required schema columns; fill missing values
    for col in ['Drug Name/Treatment', 'Drug Structure', 'Current Status', 'Source']:
        if col not in ctd_df.columns:
            ctd_df[col] = ''

    ctd_df = ctd_df.rename(columns={'Chemical Name': 'Drug Name/Treatment'})
    ctd_df['Drug Structure'] = ctd_df.get('SMILES', '') if 'SMILES' in ctd_df.columns else ''
    ctd_df['Current Status'] = 'CTD-derived'
    ctd_df['Source'] = 'CTD'
    ctd_df = ctd_df[['Drug Name/Treatment', 'Drug Structure', 'Current Status', 'Source', 'norm_name']]

    merged = pd.concat([raw_df, ctd_df], ignore_index=True, sort=False)
    merged.drop(columns=['norm_name'], inplace=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged drug file: {output_path}")
    print(f"raw drugs: {len(raw_df)}, new CTD drugs added: {len(ctd_df)}")

if __name__ == '__main__':
    merge_ctd_into_raw(
        '00_Raw_Data/drugs_raw.csv',
        '01_Cleaned_Data/CTD_D000544_chemicals_20260315024131.csv',
        '00_Raw_Data/drugs_raw_augmented.csv'
    )