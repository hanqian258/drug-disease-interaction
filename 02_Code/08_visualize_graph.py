"""
08_visualize_graph.py — Dynamic community-style network visualization

Design principles:
  - Force-directed spring layout (no linear rows)
  - Representative nodes: full size, full opacity, labeled
  - Background nodes: heavily sampled down, drawn UNDER rep nodes, alpha=0.20
  - Edge thickness proportional to interaction weight
  - Exports PNG + Cytoscape GraphML
"""

import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

# ── Representative nodes ──────────────────────────────────────────────────────
REPRESENTATIVE_DRUGS = {
    # FDA-approved AD
    'Donepezil', 'Memantine', 'Galantamine', 'Rivastigmine', 'Tacrine',
    # FDA-approved ALS
    'Riluzole', 'Edaravone',
    # Bipolar first-line
    'Lithium', 'Haloperidol', 'Quetiapine',
    # Top repurposing candidates
    'Doxycycline', 'Metformin', 'Melatonin', 'Curcumin', 'Berberine',
    'Minocycline', 'Cannabidiol', 'Nicotine', 'Resveratrol',
    'Trehalose', 'Sirolimus',
}

REPRESENTATIVE_PROTEINS = {
    'MAPT', 'APP', 'APOE', 'BACE1', 'PSEN1', 'TREM2',
    'GSK3B', 'CDK5', 'ACHE', 'GRIN1', 'CHRNA7',
    'IL1B', 'TNF', 'CASP3', 'SOD1', 'TARDBP',
    'BDNF', 'SLC6A4', 'DRD2', 'FUS',
}

# ── Colors ────────────────────────────────────────────────────────────────────
C_DRUG_APPROVED = '#1565C0'   # deep blue   — FDA-approved
C_DRUG_CTD      = '#42A5F5'   # mid blue    — CTD therapeutic
C_DRUG_BG       = '#BBDEFB'   # pale blue   — background drugs
C_PROT_CORE     = '#2E7D32'   # deep green  — core AD/ALS/BD proteins
C_PROT_BG       = '#C8E6C9'   # pale green  — background proteins
C_DIS_AD        = '#B71C1C'   # deep red    — Alzheimer's
C_DIS_OTHER     = '#EF9A9A'   # light red   — other diseases

# Max background nodes to show (keeps graph readable)
MAX_BG_DRUGS    = 20
MAX_BG_PROTEINS = 15


def visualize_graph():
    print("--- Visualizing Heterogeneous Graph (Community Layout) ---")

    for path in ['01_Cleaned_Data/expanded_graph.pt', '01_Cleaned_Data/mappings.pt']:
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return

    data = torch.load('01_Cleaned_Data/expanded_graph.pt', weights_only=False)
    maps = torch.load('01_Cleaned_Data/mappings.pt', weights_only=False)
    all_proteins = maps['all_proteins']
    d_map        = maps['d_map']
    idx_to_drug  = {v: k for k, v in d_map.items()}

    # Disease names
    diseases = ["Alzheimer's Disease", "Parkinson's Disease", "ADHD",
                "Bipolar Disorder", "ALS", "Dementia"]
    n_dis    = data['disease'].x.shape[0]
    diseases = diseases[:n_dis]

    # Drug status
    approved_ad, ctd_drugs = set(), set()
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

    # ── Decide which background nodes to include ──────────────────────────────
    all_drug_names  = [idx_to_drug[i] for i in sorted(idx_to_drug)]
    bg_drug_names   = [n for n in all_drug_names if n not in REPRESENTATIVE_DRUGS]
    bg_prot_names   = [p for p in all_proteins   if p not in REPRESENTATIVE_PROTEINS]

    # Sample down background nodes
    rng = np.random.RandomState(42)
    bg_drug_sample = rng.choice(bg_drug_names,
                                size=min(MAX_BG_DRUGS, len(bg_drug_names)),
                                replace=False).tolist()
    bg_prot_sample = rng.choice(bg_prot_names,
                                size=min(MAX_BG_PROTEINS, len(bg_prot_names)),
                                replace=False).tolist()

    include_drugs = set(REPRESENTATIVE_DRUGS) | set(bg_drug_sample)
    include_prots = set(REPRESENTATIVE_PROTEINS) | set(bg_prot_sample)

    # ── Build graph ───────────────────────────────────────────────────────────
    G = nx.Graph()

    # Drug nodes
    for name in include_drugs:
        if name not in d_map:
            continue
        is_rep = name in REPRESENTATIVE_DRUGS
        if name in approved_ad:
            color = C_DRUG_APPROVED
        elif name in ctd_drugs:
            color = C_DRUG_CTD
        else:
            color = C_DRUG_BG
        G.add_node(f"D:{name}", layer='drug', color=color,
                   label=name, representative=is_rep)

    # Protein nodes
    for name in include_prots:
        is_rep = name in REPRESENTATIVE_PROTEINS
        color  = C_PROT_CORE if is_rep else C_PROT_BG
        G.add_node(f"P:{name}", layer='protein', color=color,
                   label=name, representative=is_rep)

    # Disease nodes (always included, always representative)
    for name in diseases:
        color = C_DIS_AD if 'Alzheimer' in name else C_DIS_OTHER
        G.add_node(f"DIS:{name}", layer='disease', color=color,
                   label=name, representative=True)

    def get_w(etype):
        ea = getattr(data[etype], 'edge_attr', None)
        return ea.view(-1).numpy() if ea is not None else None

    # Drug-protein edges (representative drugs only → cleaner)
    ei = data['drug', 'binds', 'protein'].edge_index
    ew = get_w(('drug', 'binds', 'protein'))
    for i in range(ei.shape[1]):
        d_idx = ei[0, i].item()
        p_idx = ei[1, i].item()
        d_name = idx_to_drug.get(d_idx, '')
        p_name = all_proteins[p_idx]
        if d_name not in REPRESENTATIVE_DRUGS or p_name not in include_prots:
            continue
        w = float(ew[i]) if ew is not None else 0.5
        G.add_edge(f"D:{d_name}", f"P:{p_name}",
                   weight=w, etype='binds', color='#5B9BD5')

    # PPI — only between included proteins, high confidence
    ei = data['protein', 'interacts_with', 'protein'].edge_index
    ew = get_w(('protein', 'interacts_with', 'protein'))
    for i in range(0, ei.shape[1], 4):
        p1 = all_proteins[ei[0, i].item()]
        p2 = all_proteins[ei[1, i].item()]
        if p1 not in include_prots or p2 not in include_prots:
            continue
        w = float(ew[i]) if ew is not None else 0.5
        if w < 0.60:
            continue
        G.add_edge(f"P:{p1}", f"P:{p2}",
                   weight=w, etype='ppi', color='#81C784')

    # Protein-disease
    ei = data['protein', 'associated_with', 'disease'].edge_index
    ew = get_w(('protein', 'associated_with', 'disease'))
    for i in range(ei.shape[1]):
        p_name  = all_proteins[ei[0, i].item()]
        dis_idx = ei[1, i].item()
        if p_name not in include_prots or dis_idx >= len(diseases):
            continue
        w = float(ew[i]) if ew is not None else 0.5
        G.add_edge(f"P:{p_name}", f"DIS:{diseases[dis_idx]}",
                   weight=w, etype='associated', color='#FFA726')

    # Drug-disease treats (representative drugs only)
    ei = data['drug', 'treats', 'disease'].edge_index
    for i in range(ei.shape[1]):
        d_idx   = ei[0, i].item()
        dis_idx = ei[1, i].item()
        d_name  = idx_to_drug.get(d_idx, '')
        if d_name not in REPRESENTATIVE_DRUGS or dis_idx >= len(diseases):
            continue
        G.add_edge(f"D:{d_name}", f"DIS:{diseases[dis_idx]}",
                   weight=1.0, etype='treats', color='#E53935')

    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    # ── Spring layout — community style ───────────────────────────────────────
    # Seed positions: diseases in center, proteins around them,
    # drugs on the outer ring — spring layout then pulls everything naturally
    seed_pos = {}
    angles_dis = np.linspace(0, 2*np.pi, len(diseases), endpoint=False)
    for i, name in enumerate(diseases):
        r = 0.25
        seed_pos[f"DIS:{name}"] = (r * np.cos(angles_dis[i]),
                                    r * np.sin(angles_dis[i]))

    prot_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == 'protein']
    angles_p   = np.linspace(0, 2*np.pi, len(prot_nodes), endpoint=False)
    for i, n in enumerate(prot_nodes):
        r = 0.55 + rng.uniform(-0.08, 0.08)
        seed_pos[n] = (r * np.cos(angles_p[i]), r * np.sin(angles_p[i]))

    drug_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == 'drug']
    angles_d   = np.linspace(0, 2*np.pi, len(drug_nodes), endpoint=False)
    for i, n in enumerate(drug_nodes):
        r = 0.90 + rng.uniform(-0.08, 0.08)
        seed_pos[n] = (r * np.cos(angles_d[i]), r * np.sin(angles_d[i]))

    pos = nx.spring_layout(G, pos=seed_pos, k=0.18,
                           iterations=120, seed=42)

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_facecolor('#F5F5F5')
    fig.patch.set_facecolor('#F5F5F5')

    rep_nodes  = [n for n in G.nodes if G.nodes[n].get('representative')]
    back_nodes = [n for n in G.nodes if not G.nodes[n].get('representative')]

    # ── Draw background nodes FIRST (underneath) ─────────────────────────────
    if back_nodes:
        nx.draw_networkx_nodes(G, pos,
                               nodelist=back_nodes,
                               node_color=[G.nodes[n]['color'] for n in back_nodes],
                               node_size=120,
                               alpha=0.20,
                               ax=ax)

    # ── Draw edges (between background draw and rep node draw) ───────────────
    for etype, base_w, alpha in [
        ('treats',     4.0, 0.85),
        ('associated', 2.5, 0.65),
        ('binds',      2.0, 0.60),
        ('ppi',        1.0, 0.40),
    ]:
        edges  = [(u, v) for u, v, d in G.edges(data=True)
                  if d.get('etype') == etype]
        colors = [G[u][v]['color'] for u, v in edges]
        widths = [base_w * G[u][v].get('weight', 0.5) for u, v in edges]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges,
                                   edge_color=colors, width=widths,
                                   alpha=alpha, ax=ax)

    # ── Draw representative nodes ON TOP ──────────────────────────────────────
    def rep_size(n):
        layer = G.nodes[n]['layer']
        color = G.nodes[n]['color']
        if layer == 'disease':
            return 5000 if color == C_DIS_AD else 3000
        if layer == 'drug' and color == C_DRUG_APPROVED:
            return 2200
        if layer == 'protein' and color == C_PROT_CORE:
            return 1400
        return 900

    if rep_nodes:
        nx.draw_networkx_nodes(G, pos,
                               nodelist=rep_nodes,
                               node_color=[G.nodes[n]['color'] for n in rep_nodes],
                               node_size=[rep_size(n) for n in rep_nodes],
                               alpha=0.95,
                               ax=ax)

    # Labels on representative nodes only
    labels = {n: G.nodes[n]['label'] for n in rep_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=6.5, font_weight='bold', ax=ax)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=C_DRUG_APPROVED, label='FDA-approved drug (representative)'),
        mpatches.Patch(color=C_DRUG_CTD,      label='CTD therapeutic drug (representative)'),
        mpatches.Patch(color=C_DRUG_BG,       label='Other drug (background, α=0.20)'),
        mpatches.Patch(color=C_PROT_CORE,     label='Core disease protein (representative)'),
        mpatches.Patch(color=C_PROT_BG,       label='Other PPI protein (background, α=0.20)'),
        mpatches.Patch(color=C_DIS_AD,        label="Alzheimer's Disease"),
        mpatches.Patch(color=C_DIS_OTHER,     label='Other disease'),
        plt.Line2D([0],[0], color='#E53935', lw=3,   label='Drug treats disease'),
        plt.Line2D([0],[0], color='#FFA726', lw=2,   label='Protein–disease association'),
        plt.Line2D([0],[0], color='#5B9BD5', lw=1.5, label='Drug–protein binding'),
        plt.Line2D([0],[0], color='#81C784', lw=1,   label='Protein–protein interaction'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=8, framealpha=0.92, ncol=2)

    ax.set_title(
        "Heterogeneous Drug–Protein–Disease Interaction Network\n"
        "Force-directed layout  |  Edge thickness ∝ weight  |  "
        "Background nodes semi-transparent",
        fontsize=12, fontweight='bold', pad=12
    )
    ax.axis('off')
    plt.tight_layout()

    png_out = 'network_visualization.png'
    plt.savefig(png_out, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved PNG → {png_out}")
    plt.close()

    # ── Cytoscape GraphML export ───────────────────────────────────────────────
    G_exp = nx.Graph()
    for n, attrs in G.nodes(data=True):
        x, y = pos.get(n, (0, 0))
        G_exp.add_node(n,
            label          = attrs.get('label', n),
            layer          = attrs.get('layer', ''),
            color          = attrs.get('color', '#CCCCCC'),
            representative = str(attrs.get('representative', False)),
            pos_x          = float(x * 800),
            pos_y          = float(y * 800),
        )
    for u, v, attrs in G.edges(data=True):
        G_exp.add_edge(u, v,
            weight     = float(attrs.get('weight', 0.5)),
            etype      = attrs.get('etype', ''),
            color      = attrs.get('color', '#CCCCCC'),
        )
    graphml_out = 'network_visualization.graphml'
    nx.write_graphml(G_exp, graphml_out)
    print(f"  Saved GraphML → {graphml_out}")
    print("\n  Cytoscape: File → Import → Network from File → select .graphml")
    print("  Style → Node Fill Color → map to 'color'")
    print("  Style → Edge Width      → map to 'weight'")


if __name__ == "__main__":
    visualize_graph()