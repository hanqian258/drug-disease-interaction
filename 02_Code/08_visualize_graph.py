import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

def visualize_graph():
    print("--- Visualizing Heterogeneous Graph ---")

    if not os.path.exists('01_Cleaned_Data/expanded_graph.pt'):
        print("Error: expanded_graph.pt not found.")
        return

    data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)

    drug_names   = maps['drug_names']
    all_proteins = maps['all_proteins']
    d_map        = maps['d_map']
    idx_to_drug  = {v: k for k, v in d_map.items()}

    COLOR_DRUG_APPROVED = '#1A6FBF'
    COLOR_DRUG_CTD      = '#6EB5E8'
    COLOR_DRUG_OTHER    = '#C2DFF5'
    COLOR_PROT_CORE     = '#1A8C4E'
    COLOR_PROT_OTHER    = '#7DCB9F'
    COLOR_DIS_AD        = '#C0392B'
    COLOR_DIS_OTHER     = '#E8837C'

    CORE_AD_PROTEINS = {
        'MAPT','APP','APOE','BACE1','PSEN1','PSEN2',
        'TREM2','CLU','BIN1','PICALM','GSK3B','CDK5',
        'CASP3','IL1B','TNF','ACHE','GRIN1','CHRNA7'
    }

    approved_ad = set()
    ctd_drugs   = set()
    try:
        raw = pd.read_csv('00_Raw_Data/drugs_raw_augmented.csv')
        for _, row in raw.iterrows():
            name   = str(row['Drug Name/Treatment']).strip()
            status = str(row.get('Current Status', '')).strip()
            if status == 'Approved':
                approved_ad.add(name)
            elif status == 'CTD-derived':
                ctd_drugs.add(name)
    except Exception as e:
        print(f"  Warning: {e}")

    approved_indices = {d_map[n] for n in approved_ad if n in d_map}

    G = nx.Graph()

    all_drug_names_ordered = [idx_to_drug[i] for i in sorted(idx_to_drug.keys())]
    for name in all_drug_names_ordered:
        if name in approved_ad:
            color = COLOR_DRUG_APPROVED
        elif name in ctd_drugs:
            color = COLOR_DRUG_CTD
        else:
            color = COLOR_DRUG_OTHER
        G.add_node(f"D:{name}", layer='drug', color=color, label=name)

    for name in all_proteins:
        color = COLOR_PROT_CORE if name in CORE_AD_PROTEINS else COLOR_PROT_OTHER
        G.add_node(f"P:{name}", layer='protein', color=color, label=name)

    diseases = ["Alzheimer's Disease", "Parkinson's Disease", "ADHD"]
    for name in diseases:
        color = COLOR_DIS_AD if 'Alzheimer' in name else COLOR_DIS_OTHER
        G.add_node(f"DIS:{name}", layer='disease', color=color, label=name)

    def get_weights(etype):
        ea = getattr(data[etype], 'edge_attr', None)
        return ea.view(-1).numpy() if ea is not None else None

    # Drug-protein (approved only)
    ei = data['drug', 'binds', 'protein'].edge_index
    ew = get_weights(('drug', 'binds', 'protein'))
    for i in range(ei.shape[1]):
        d_idx = ei[0, i].item()
        if d_idx not in approved_indices:
            continue
        p_idx  = ei[1, i].item()
        d_name = idx_to_drug.get(d_idx, str(d_idx))
        p_name = all_proteins[p_idx]
        w = float(ew[i]) if ew is not None else 0.5
        G.add_edge(f"D:{d_name}", f"P:{p_name}",
                   weight=w, etype='binds', color='#5B9BD5')

    # PPI high-confidence sampled
    ei = data['protein', 'interacts_with', 'protein'].edge_index
    ew = get_weights(('protein', 'interacts_with', 'protein'))
    for i in range(0, ei.shape[1], 6):
        w = float(ew[i]) if ew is not None else 0.5
        if w < 0.65:
            continue
        p1 = all_proteins[ei[0, i].item()]
        p2 = all_proteins[ei[1, i].item()]
        G.add_edge(f"P:{p1}", f"P:{p2}",
                   weight=w, etype='ppi', color='#95D5B2')

    # Protein-disease
    ei = data['protein', 'associated_with', 'disease'].edge_index
    ew = get_weights(('protein', 'associated_with', 'disease'))
    for i in range(ei.shape[1]):
        p_name   = all_proteins[ei[0, i].item()]
        dis_idx  = ei[1, i].item()
        dis_name = diseases[dis_idx] if dis_idx < len(diseases) else f"Disease_{dis_idx}"
        w = float(ew[i]) if ew is not None else 0.5
        G.add_edge(f"P:{p_name}", f"DIS:{dis_name}",
                   weight=w, etype='associated', color='#F4A261')

    # Drug-disease treats
    ei = data['drug', 'treats', 'disease'].edge_index
    for i in range(ei.shape[1]):
        d_idx   = ei[0, i].item()
        dis_idx = ei[1, i].item()
        d_name  = idx_to_drug.get(d_idx, str(d_idx))
        dis_name = diseases[dis_idx] if dis_idx < len(diseases) else f"Disease_{dis_idx}"
        G.add_edge(f"D:{d_name}", f"DIS:{dis_name}",
                   weight=1.0, etype='treats', color='#E63946')

    # ── FIXED HIERARCHICAL LAYOUT ─────────────────────────────────────────────
    pos = {}

    drug_nodes    = [n for n in G.nodes if G.nodes[n]['layer'] == 'drug']
    protein_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == 'protein']
    disease_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == 'disease']

    def drug_sort_key(n):
        c = G.nodes[n]['color']
        if c == COLOR_DRUG_APPROVED: return 0
        if c == COLOR_DRUG_CTD:      return 1
        return 2
    drug_nodes    = sorted(drug_nodes, key=drug_sort_key)
    protein_nodes = sorted(protein_nodes,
                           key=lambda n: 0 if G.nodes[n]['color'] == COLOR_PROT_CORE else 1)

    def spread(nodes, y, margin=0.02):
        n = len(nodes)
        if n == 0:
            return
        xs = np.linspace(margin, 1 - margin, n)
        for node, x in zip(nodes, xs):
            pos[node] = (x, y)

    spread(drug_nodes,    y=2.0)
    spread(protein_nodes, y=1.0)
    spread(disease_nodes, y=0.0)

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(28, 12))
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    node_sizes  = []
    for n in G.nodes:
        layer = G.nodes[n]['layer']
        color = G.nodes[n]['color']
        if layer == 'disease':
            node_sizes.append(3500 if color == COLOR_DIS_AD else 2000)
        elif layer == 'drug' and color == COLOR_DRUG_APPROVED:
            node_sizes.append(1200)
        elif layer == 'protein' and color == COLOR_PROT_CORE:
            node_sizes.append(900)
        elif layer == 'drug' and color == COLOR_DRUG_CTD:
            node_sizes.append(500)
        else:
            node_sizes.append(250)

    for etype, base_w, alpha in [
        ('treats',     3.5, 0.85),
        ('associated', 2.0, 0.65),
        ('binds',      1.8, 0.60),
        ('ppi',        0.8, 0.35),
    ]:
        edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == etype]
        colors = [G[u][v]['color'] for u, v in edges]
        widths = [base_w * G[u][v].get('weight', 0.5) for u, v in edges]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges,
                                   edge_color=colors, width=widths,
                                   alpha=alpha, ax=ax)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92, ax=ax)

    important = (
        {n for n in G.nodes if G.nodes[n]['layer'] == 'disease'} |
        {n for n in G.nodes if G.nodes[n]['color'] == COLOR_DRUG_APPROVED} |
        {n for n in G.nodes if G.nodes[n]['color'] == COLOR_PROT_CORE}
    )
    labels = {n: G.nodes[n]['label'] for n in important}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=6.5, font_weight='bold', ax=ax)

    for y, label in [(2.0, 'Layer 1 — Drugs'),
                     (1.0, 'Layer 2 — Proteins'),
                     (0.0, 'Layer 3 — Diseases')]:
        ax.text(-0.02, y, label, transform=ax.transData,
                fontsize=9, color='#555', va='center', ha='right', style='italic')

    for y in [0.5, 1.5]:
        ax.axhline(y, color='#CCCCCC', linewidth=0.8, linestyle='--', alpha=0.5)

    legend_elements = [
        mpatches.Patch(color=COLOR_DRUG_APPROVED, label='FDA-approved AD drug (Layer 1)'),
        mpatches.Patch(color=COLOR_DRUG_CTD,      label='CTD therapeutic drug (Layer 1)'),
        mpatches.Patch(color=COLOR_DRUG_OTHER,    label='Other drug (Layer 1)'),
        mpatches.Patch(color=COLOR_PROT_CORE,     label='Core AD protein (Layer 2)'),
        mpatches.Patch(color=COLOR_PROT_OTHER,    label='Other PPI protein (Layer 2)'),
        mpatches.Patch(color=COLOR_DIS_AD,        label="Alzheimer's Disease (Layer 3)"),
        mpatches.Patch(color=COLOR_DIS_OTHER,     label='Other disease (Layer 3)'),
        plt.Line2D([0],[0], color='#E63946', linewidth=3,   label='Drug treats disease'),
        plt.Line2D([0],[0], color='#F4A261', linewidth=2,   label='Protein–disease association'),
        plt.Line2D([0],[0], color='#5B9BD5', linewidth=1.5, label='Drug–protein binding'),
        plt.Line2D([0],[0], color='#95D5B2', linewidth=1,   label='Protein–protein interaction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=8, framealpha=0.9, ncol=2)

    ax.set_title(
        "Heterogeneous Drug–Protein–Disease Interaction Network\n"
        "Edge thickness ∝ interaction weight  |  Node color = biological layer",
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlim(-0.12, 1.05)
    ax.set_ylim(-0.5, 2.6)
    ax.axis('off')
    plt.tight_layout()

    out = 'network_visualization.png'
    plt.savefig(out, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved → {out}")
    plt.close()

if __name__ == "__main__":
    visualize_graph()