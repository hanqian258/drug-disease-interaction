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
    NAME_CORRECTIONS = {
    "N2-((2S)-2-(3,5-difluorophenyl)-2-hydroxyethanoyl)-N1-((7S)-5-methyl-6-oxo-6,7-dihydro-5H-dibenzo(b,d)azepin-7-yl)-L-alaninamide": "Semagacestat",
    "3-methyl-5-(1-methyl-2-pyrrolidinyl)isoxazole": "ABT-418",
    "BMS 708163": "Avagacestat",
    "Vitamin D": "Cholecalciferol",
    "Raloxifene Hydrochloride": "Raloxifene",
}

# Apply before PubChem lookup
drug_name = NAME_CORRECTIONS.get(drug_name, drug_name)
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
