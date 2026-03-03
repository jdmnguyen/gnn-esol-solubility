# src/featurization.py
from rdkit import Chem
import torch
from torch_geometric.data import Data

# Simple atom feature helper
ATOM_LIST = [1, 6, 7, 8, 9, 16, 17, 35, 53]  # H, C, N, O, F, S, Cl, Br, I

def atom_to_feature_vector(atom):
    """Return a simple feature vector for an RDKit atom."""
    atomic_num = atom.GetAtomicNum()
    atom_type = [int(atomic_num == a) for a in ATOM_LIST]
    degree = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge()
    hybridization = atom.GetHybridization()

    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ]
    hybridization_feat = [int(hybridization == h) for h in hybridization_types]

    is_aromatic = int(atom.GetIsAromatic())

    return torch.tensor(
        atom_type
        + [degree, formal_charge]
        + hybridization_feat
        + [is_aromatic],
        dtype=torch.float,
    )

def smiles_to_mol(smiles: str):
    """Convert SMILES to an RDKit Mol, handling failures."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol

def mol_to_graph_data_obj(mol, y_value):
    """Convert RDKit Mol into a PyG Data object with node features and edges."""
    # Node features
    x = torch.stack([atom_to_feature_vector(atom) for atom in mol.GetAtoms()], dim=0)

    # Edge index (bonds)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # undirected graph as two directed edges

    if len(edge_index) == 0:
        # Handle molecules with no bonds (rare, but good to be safe)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Label
    y = torch.tensor([y_value], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data
