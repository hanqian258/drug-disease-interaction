#!/usr/bin/env bash
# rename_pipeline.sh
# Run once from your project root to apply the new file numbering.
# Usage: bash rename_pipeline.sh

set -e

echo "Renaming pipeline files..."

cd 02_Code

# The old 04_validate_data.py clashed with 04_expand_graph.py
# New numbering inserts it after expand as step 05
mv 04_validate_data.py  05_validate_graph.py  2>/dev/null && echo "  04_validate_data.py  → 05_validate_graph.py"  || echo "  (05_validate_graph.py already exists or source missing)"
mv 05_train_gcn.py      06_train_gcn.py        2>/dev/null && echo "  05_train_gcn.py       → 06_train_gcn.py"       || echo "  (06_train_gcn.py already exists or source missing)"
mv 06_inference.py      07_inference.py        2>/dev/null && echo "  06_inference.py       → 07_inference.py"       || echo "  (07_inference.py already exists or source missing)"
mv 07_visualize_graph.py 08_visualize_graph.py 2>/dev/null && echo "  07_visualize_graph.py → 08_visualize_graph.py" || echo "  (08_visualize_graph.py already exists or source missing)"

cd ..

echo ""
echo "Final pipeline order:"
echo "  01_clean_drugs.py          — fetch SMILES from PubChem"
echo "  02_fetch_string_ppi.py     — fetch PPI from STRING"
echo "  03_build_hetero_graph.py   — build graph with CTD weights"
echo "  04_expand_graph.py         — add disease nodes, DisGeNET weights"
echo "  05_validate_graph.py       — print graph stats"
echo "  06_train_gcn.py            — train GNN (use fixed version)"
echo "  07_inference.py            — predict drug-disease scores"
echo "  08_visualize_graph.py      — render network plot"
echo ""
echo "Also update these internal references after renaming:"
echo "  07_inference.py line ~10:  model path still reads gnn_model.pt (OK)"
echo "  README.md:                 update script names in run instructions"
echo "  Drug_Discovery_GNN_Demo.ipynb: update !python3 paths in cells"
