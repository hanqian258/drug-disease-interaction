import pandas as pd
import pubchempy as pcp
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_smiles(drug_name):
    try:
        results = pcp.get_compounds(drug_name, 'name')
        if results:
            return results[0].canonical_smiles
    except Exception as e:
        logging.error(f"Error fetching SMILES for {drug_name}: {e}")
    return None

def clean_drug_list(input_path, output_path):
    logging.info(f"Processing {input_path}...")
    df = pd.read_csv(input_path)

    # Ensure 'smiles' column exists
    if 'smiles' not in df.columns:
        df['smiles'] = None

    for idx, row in df.iterrows():
        if pd.isna(row['smiles']) or row['smiles'] == '':
            logging.info(f"Fetching SMILES for {row['name']}...")
            smiles = get_smiles(row['name'])
            if smiles:
                df.at[idx, 'smiles'] = smiles
                logging.info(f"Found SMILES for {row['name']}")
            else:
                logging.warning(f"Could not find SMILES for {row['name']}")

    df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    os.makedirs('01_Cleaned_Data', exist_ok=True)

    # Process positive drugs
    clean_drug_list('00_Raw_Data/positive_drugs.csv', '01_Cleaned_Data/positive_drugs.csv')

    # Process negative controls
    clean_drug_list('00_Raw_Data/negative_controls.csv', '01_Cleaned_Data/negative_controls.csv')

    # Copy drug_links as it doesn't need SMILES (it uses drug names)
    links = pd.read_csv('00_Raw_Data/drug_links.csv')
    links.to_csv('01_Cleaned_Data/drug_links.csv', index=False)
