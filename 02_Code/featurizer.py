import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from rdkit import Chem
from torch_geometric.data import Data

class DrugFeaturizer:
    def __init__(self):
        self.allowed_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']

    def get_atom_features(self, atom):
        symbol = atom.GetSymbol()
        atom_idx = self.allowed_atoms.index(symbol) if symbol in self.allowed_atoms else len(self.allowed_atoms)
        return [atom_idx, atom.GetDegree(), int(atom.GetHybridization())]

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        node_features = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(node_features, dtype=torch.float)
        
        edge_indices = []
        for bond in mol.GetBonds():
            s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(s, e), (e, s)])
            
        if not edge_indices: return None
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
