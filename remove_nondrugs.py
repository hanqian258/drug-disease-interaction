"""
remove_nondrugs.py
──────────────────
Run from your project root. Removes non-drug category entries
from drugs_raw_augmented.csv that have no SMILES and should
never have been in the drug graph.

Usage:
    python3 remove_nondrugs.py
"""

import pandas as pd

TO_REMOVE = {
    "Plant Preparations",
    "Biological Products",
    "Nanotubes, Carbon",
    "Plant Extracts",
    "Androgens",
    "Heparin, Low-Molecular-Weight",
    "Ginkgo biloba extract",
    "Cholinesterase Inhibitors",
    "Kai-Xin-San",
    "Anti-Inflammatory Agents, Non-Steroidal",
    "Lecithins",
}

path = '00_Raw_Data/drugs_raw_augmented.csv'

df     = pd.read_csv(path)
before = len(df)

mask   = df['Drug Name/Treatment'].str.strip().isin(TO_REMOVE)
removed = df[mask]['Drug Name/Treatment'].tolist()

df = df[~mask].reset_index(drop=True)
df.to_csv(path, index=False)

print(f"Removed {len(removed)} entries: {before} -> {len(df)} drugs")
print("Removed:")
for name in removed:
    print(f"  - {name}")

print("\nNext steps:")
print("  python3 02_Code/03_build_hetero_graph.py")
print("  python3 02_Code/04_expand_graph.py")
print("  python3 02_Code/06_train_gcn.py")
