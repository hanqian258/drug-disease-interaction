import requests
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fetch_string_interactions(proteins, required_score=400):
    string_api_url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "%0d".join(proteins),
        "species": 9606,  # Human
        "required_score": required_score,
        "caller_identity": "isef_project"
    }

    response = requests.post(string_api_url, data=params)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Error fetching data from STRING DB: {response.status_code}")
        return []

def main():
    proteins = [
    # Direct drug targets (your current drugs)
    "ACHE", "BACE1", "GRIN1", "GRIN2B", "CHRNA7",
    # Core AD genetic risk genes
    "MAPT", "APP", "APOE", "PSEN1", "PSEN2",
    "TREM2", "CLU", "PICALM", "BIN1", "CR1",
    # PPI bridge proteins (connect targets to disease)
    "GSK3B", "CDK5", "ADAM10", "IDE", "MMP9",
    "IL6", "TNF", "CASP3", "BECN1", "SQSTM1",
    # For multi-disease expansion (Parkinson's)
    "SNCA", "LRRK2", "PRKN", "PINK1", "UCHL1",
    "A2M", "ADAMTS1", "AMFR", "APOC1", "ARC", "ATP5F1A",
    "BAX", "BCHE", "BCL2", "CALM1", "CASP3", "CHRNB2",
    "CRH", "CST3", "CYP2D6", "DHCR24", "EIF2S1", "ENO1",
    "ESR1", "F2", "HMOX1", "IDE", "IGF1", "IGF1R",
    "IGF2", "IGF2R", "INS", "INSR", "IQCK", "LEP",
    "MAOB", "MPO", "NCSTN", "NOS3", "NPY", "PLAU",
    "PPARG", "PYY", "RELN", "SLC2A4", "SOD2", "TF",
    "TFAM", "TPI1", "VEGFA", "VSNL1",
    # Bipolar Disorder
    "ANK3", "CACNA1C", "DISC1", "BDNF", "GSK3B", "DTNBP1",
    "NRG1", "DAOA", "SLC6A4", "TPH2", "COMT", "DRD2", "HTR2A",
    # Dementia / FTD
    "APP", "PSEN1", "PSEN2", "GRN", "FUS", "VCP", "SQSTM1",
    "UBQLN2", "C9orf72",
    # ALS
    "SOD1", "OPTN", "TBK1", "NEK1", "SETX",
    # AD new proteins (from AD_research_-_AD_Chemicals.csv)
    "ABCA7", "ABI3", "ACE", "ADAM10", "CYP46A1",
    "CD2AP", "CD33", "DPYSL2", "EPHA1", "GAPDHS",
    "HFE", "INPP5D", "IQCK", "MMP9", "MTHFR",
    "NCSTN", "NECTIN2", "PGRMC1", "PILRA", "PLCG2",
    "PRNP", "PTK2B", "SLC30A6", "TOMM40", "TPP1",
    "VSNL1", "WWOX", "HLA-DRB5", "MIR100",
    # ADHD new proteins (from AD_research_-_ADHD_Chemicals.csv)
    "AS3MT", "CHRNA4", "CIC", "CNR1", "DRD4", "DRD5",
    "FGD1", "GIT1", "GRM1", "GRM7", "GRM8", "SLC6A3",
    "STS", "TPH2",
    # Parkinson's new proteins (from AD_research_-_Parkinson_s_Chemicals.csv)
    "ABCB1", "ADARB2", "AIF1", "ALDH2", "BAG5", "BST1",
    "CEACAM6", "CP", "CYP2E1", "DDC", "DDIT4", "DNM1L",
    "DRD1", "DRD2", "EDN1", "ENO2", "FBP1", "FCER2",
    "FGB", "GDNF", "GFAP", "GPDH1", "GSTA4", "GSTM1",
    "GSTP1", "HBG1", "HGF", "HLA-DRA", "HMOX1",
    "HSPA1A", "HSPA9", "IL1B", "MAG", "MAOA", "MAP2",
    "MAP3K5", "MTA1", "NCAPG2", "NGF", "NOS1", "NQO1",
    "PARK7", "PPARGC1A", "RAB32", "RPL14", "RPL6", "RPS8",
    "SLC18A2", "SLC2A14", "SLC30A10", "SLC38A2",
    "TALDO1", "TCL1B", "TH", "TMEM230", "TRPM2", "VPS35",
    # DisGeNET proteins missing from STRING graph (from protein_disease_weights.csv)
    # AD / dementia
    "SORL1", "AGER", "TTR", "TYROBP", "CTSD", "TARDBP",
    "TMEM106B", "GBA1", "DCTN1", "NEFH", "NEFL", "INA", "PRPH",
    "FBXO7", "SYNJ1", "ATP13A2", "SNCB", "UNC13A",
    # Bipolar / ADHD
    "ANK3", "CACNA1C", "ADGRL1", "ADGRL3", "FMR1", "MECP2",
    "NTRK2", "GAD1", "GRIN2A", "NCAN", "CLOCK", "CRY1", "FKBP5",
    "GRIA3", "IMPA2", "NCAM1", "NR4A2", "POMC", "RXRA",
    "SLC18A1", "FOS", "GABRA1", "DBH", "DRD3", "GNB5",
    # Parkinson's
    "LRRK2", "PINK1", "PARK7", "VPS35", "FBXO7", "NR4A2",
    "BCL2L1", "MAP1B", "PON1", "SLC2A9",
]
    logging.info(f"Fetching interactions for {len(proteins)} proteins from STRING DB...")
    interactions = fetch_string_interactions(proteins)

    if interactions:
        df = pd.DataFrame(interactions)
        # Keep only relevant columns
        cols = ['preferredName_A', 'preferredName_B', 'score',
                'experimentally_determined_interaction',
                'database_annotated', 'coexpression',
                'automated_textmining', 'homology']
        df = df[[c for c in cols if c in df.columns]]

        output_path = '01_Cleaned_Data/ppi_interactions.csv'
        df.to_csv(output_path, index=False)
        logging.info(f"Successfully fetched {len(df)} interactions and saved to {output_path}")

        # Verify internal controls
        controls = [('APP', 'BACE1'), ('MAPT', 'GSK3B'), ('SNCA', 'LRRK2'), ('DRD2', 'COMT'), ('TH', 'DDC')]
        for p1, p2 in controls:
            match = df[((df['preferredName_A'] == p1) & (df['preferredName_B'] == p2)) |
                       ((df['preferredName_A'] == p2) & (df['preferredName_B'] == p1))]
            if not match.empty:
                logging.info(f"Internal Control Verified: {p1}-{p2} found with score {match['score'].values[0]}")
            else:
                logging.warning(f"Internal Control Missing: {p1}-{p2} not found in high-confidence network")
    else:
        logging.warning("No interactions found.")

if __name__ == "__main__":
    main()