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

    # ── Node color scheme (one color per layer) ───────────────────────────────
    # Layer 1 — drugs
    COLOR_DRUG_APPROVED  = '#4A90D9'   # strong blue   — FDA-approved AD drugs
    COLOR_DRUG_CTD       = '#A8C8F0'   # light blue    — CTD therapeutic drugs
    COLOR_DRUG_OTHER     = '#D0E8FF'   # very light    — other drugs in library
    # Layer 2 — proteins
    COLOR_PROTEIN_CORE   = '#2ECC71'   # strong green  — core AD proteins (MAPT, APP, APOE...)
    COLOR_PROTEIN_OTHER  = '#A8E6C4'   # light green   — other PPI proteins
    # Layer 3 — diseases
    COLOR_DISEASE_AD     = '#E74C3C'   # red           — Alzheimer's Disease
    COLOR_DISEASE_OTHER  = '#F1948A'   # light red     — other diseases

    CORE_AD_PROTEINS = {
        'MAPT', 'APP', 'APOE', 'BACE1', 'PSEN1', 'PSEN2',
        'TREM2', 'CLU', 'BIN1', 'PICALM', 'GSK3B', 'CDK5'
    }

    # ── Load drug status for color differentiation ────────────────────────────
    approved_ad_drugs = set()
    ctd_drugs         = set()
    try:
        raw = pd.read_csv('00_Raw_Data/drugs_raw_augmented.csv')
        for _, row in raw.iterrows():
            name   = str(row['Drug Name/Treatment']).strip()
            status = str(row.get('Current Status', '')).strip()
            if status == 'Approved':
                approved_ad_drugs.add(name)
            elif status == 'CTD-derived':
                ctd_drugs.add(name)
    except Exception as e:
        print(f"  Warning: could not load drug status — {e}")

    # ── Build NetworkX graph ──────────────────────────────────────────────────
    G = nx.Graph()

    # Reverse d_map: index → name
    idx_to_drug = {v: k for k, v in d_map.items()}

    # Add drug nodes
    for idx, name in idx_to_drug.items():
        if name in approved_ad_drugs:
            color = COLOR_DRUG_APPROVED
        elif name in ctd_drugs:
            color = COLOR_DRUG_CTD
        else:
            color = COLOR_DRUG_OTHER
        G.add_node(f"D:{name}", layer='drug', color=color,
                   label=name, node_idx=idx)

    # Add protein nodes
    for i, name in enumerate(all_proteins):
        color = COLOR_PROTEIN_CORE if name in CORE_AD_PROTEINS else COLOR_PROTEIN_OTHER
        G.add_node(f"P:{name}", layer='protein', color=color, label=name)

    # Add disease nodes
    diseases = ['Alzheimer\'s Disease', 'Parkinson\'s Disease', 'ADHD']
    for i, name in enumerate(diseases):
        color = COLOR_DISEASE_AD if 'Alzheimer' in name else COLOR_DISEASE_OTHER
        G.add_node(f"DIS:{name}", layer='disease', color=color, label=name)

    # ── Helper: get edge attribute weights ───────────────────────────────────
    def get_weights(etype):
        ea = getattr(data[etype], 'edge_attr', None)
        if ea is None:
            return None
        w = ea.view(-1).numpy()
        return w

    # ── Drug → Protein edges (weighted by CTD inference score) ───────────────
    ei = data['drug', 'binds', 'protein'].edge_index
    ew = get_weights(('drug', 'binds', 'protein'))

    # Only show edges for approved AD drugs to keep visualization clean
    approved_indices = {d_map[n] for n in approved_ad_drugs if n in d_map}

    for i in range(ei.shape[1]):
        d_idx = ei[0, i].item()
        p_idx = ei[1, i].item()
        if d_idx not in approved_indices:
            continue
        d_name = idx_to_drug.get(d_idx, str(d_idx))
        p_name = all_proteins[p_idx]
        w = float(ew[i]) if ew is not None else 0.5
        G.add_edge(f"D:{d_name}", f"P:{p_name}",
                   weight=w, etype='binds', color='#5B9BD5')

    # ── Protein → Protein edges (weighted by STRING combined_score) ───────────
    ei  = data['protein', 'interacts_with', 'protein'].edge_index
    ew  = get_weights(('protein', 'interacts_with', 'protein'))
    # Sample every 8th edge to avoid hairball; keep high-confidence ones
    for i in range(0, ei.shape[1], 8):
        p1 = all_proteins[ei[0, i].item()]
        p2 = all_proteins[ei[1, i].item()]
        w  = float(ew[i]) if ew is not None else 0.5
        if w < 0.6:          # only show high-confidence PPI
            continue
        G.add_edge(f"P:{p1}", f"P:{p2}",
                   weight=w, etype='ppi', color='#95D5B2')

    # ── Protein → Disease edges (weighted by DisGeNET DPI score) ─────────────
    ei = data['protein', 'associated_with', 'disease'].edge_index
    ew = get_weights(('protein', 'associated_with', 'disease'))
    for i in range(ei.shape[1]):
        p_name   = all_proteins[ei[0, i].item()]
        dis_idx  = ei[1, i].item()
        dis_name = diseases[dis_idx] if dis_idx < len(diseases) else f"Disease_{dis_idx}"
        w = float(ew[i]) if ew is not None else 0.5
        G.add_edge(f"P:{p_name}", f"DIS:{dis_name}",
                   weight=w, etype='associated', color='#F4A261')

    # ── Drug → Disease edges (training labels, weight=1.0) ───────────────────
    ei = data['drug', 'treats', 'disease'].edge_index
    for i in range(ei.shape[1]):
        d_idx   = ei[0, i].item()
        dis_idx = ei[1, i].item()
        d_name  = idx_to_drug.get(d_idx, str(d_idx))
        dis_name = diseases[dis_idx] if dis_idx < len(diseases) else f"Disease_{dis_idx}"
        G.add_edge(f"D:{d_name}", f"DIS:{dis_name}",
                   weight=1.0, etype='treats', color='#E63946')

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


    # ── Layout: separate layers vertically ────────────────────────────────────
    # Collect node lists by layer for layout
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'drug']
    protein_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'protein']
    disease_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'disease']

    # Use spring layout but seed positions by layer for better separation
    # Compute initial positions — layer-based seeding
    pos = {}
    rng = np.random.RandomState(42)

    # Position drugs closer to the disease cluster, not spread in a ring
    for i, n in enumerate(drug_nodes):
        x = 0.3 + 0.7 * (i / max(len(drug_nodes) - 1, 1))
        pos[n] = (x, 1.5 + rng.uniform(-0.3, 0.3))

    for i, n in enumerate(protein_nodes):
        x = 0.2 + 0.6 * (i / max(len(protein_nodes) - 1, 1))
        pos[n] = (x, 0.8 + rng.uniform(-0.2, 0.2))

    for i, n in enumerate(disease_nodes):
        pos[n] = (0.3 + 0.4 * i, 0.0)


    # Use stronger spring force and more iterations to pull connected nodes together
    pos = nx.spring_layout(G, pos=pos, k=0.15, iterations=120, seed=42)

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    node_sizes  = []
    for n in G.nodes:
        layer = G.nodes[n]['layer']
        if layer == 'disease':
            node_sizes.append(3000)
        elif layer == 'drug' and G.nodes[n]['color'] == COLOR_DRUG_APPROVED:
            node_sizes.append(1800)
        elif layer == 'protein' and G.nodes[n]['color'] == COLOR_PROTEIN_CORE:
            node_sizes.append(1400)
        else:
            node_sizes.append(800)

    # Draw edges grouped by type with different thickness and color
    for etype, base_width, alpha in [
        ('treats',     4.0, 0.90),
        ('associated', 2.5, 0.70),
        ('binds',      2.0, 0.65),
        ('ppi',        1.0, 0.40),
    ]:
        edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == etype]
        colors = [G[u][v]['color'] for u, v in edges]
        widths = [base_width * G[u][v].get('weight', 0.5) for u, v in edges]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges,
                                   edge_color=colors, width=widths,
                                   alpha=alpha, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.95, ax=ax)

    # Labels — only for important nodes to avoid clutter
    important = (
        {n for n in G.nodes if G.nodes[n]['layer'] == 'disease'} |
        {n for n in G.nodes if G.nodes[n]['color'] == COLOR_DRUG_APPROVED} |
        {n for n in G.nodes if G.nodes[n]['color'] == COLOR_PROTEIN_CORE}
    )
    labels = {n: G.nodes[n]['label'] for n in important}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=7, font_weight='bold', ax=ax)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=COLOR_DRUG_APPROVED,  label='FDA-approved AD drug (Layer 1)'),
        mpatches.Patch(color=COLOR_DRUG_CTD,       label='CTD therapeutic drug (Layer 1)'),
        mpatches.Patch(color=COLOR_DRUG_OTHER,     label='Other drug (Layer 1)'),
        mpatches.Patch(color=COLOR_PROTEIN_CORE,   label='Core AD protein (Layer 2)'),
        mpatches.Patch(color=COLOR_PROTEIN_OTHER,  label='Other PPI protein (Layer 2)'),
        mpatches.Patch(color=COLOR_DISEASE_AD,     label="Alzheimer's Disease (Layer 3)"),
        mpatches.Patch(color=COLOR_DISEASE_OTHER,  label='Other disease (Layer 3)'),
        plt.Line2D([0],[0], color='#E63946', linewidth=3, label='Drug treats disease'),
        plt.Line2D([0],[0], color='#F4A261', linewidth=2, label='Protein–disease association'),
        plt.Line2D([0],[0], color='#5B9BD5', linewidth=2, label='Drug–protein binding'),
        plt.Line2D([0],[0], color='#95D5B2', linewidth=1, label='Protein–protein interaction'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=8, framealpha=0.9, ncol=2)

    ax.set_title(
        "Heterogeneous Drug–Protein–Disease Interaction Network\n"
        "Edge thickness ∝ interaction weight  |  "
        "Node color = biological layer",
        fontsize=13, fontweight='bold', pad=15
    )
    ax.axis('off')
    plt.tight_layout()

    out_path = 'network_visualization.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved → {out_path}")
    plt.close()

if __name__ == "__main__":
    visualize_graph()
