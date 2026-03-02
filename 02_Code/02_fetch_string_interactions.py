import requests
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fetch_string_interactions(proteins, required_score=700):
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
        "MAPT", "APP", "APOE", "BACE1", "PSEN1",
        "PSEN2", "TREM2", "CLU", "PICALM", "CR1", "GSK3B"
    ]

    logging.info(f"Fetching interactions for {len(proteins)} proteins from STRING DB...")
    interactions = fetch_string_interactions(proteins)

    if interactions:
        df = pd.DataFrame(interactions)
        # Keep only relevant columns
        df = df[['preferredName_A', 'preferredName_B', 'score']]

        output_path = '01_Cleaned_Data/ppi_interactions.csv'
        df.to_csv(output_path, index=False)
        logging.info(f"Successfully fetched {len(df)} interactions and saved to {output_path}")

        # Verify internal controls
        controls = [('APP', 'BACE1'), ('MAPT', 'GSK3B')]
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
