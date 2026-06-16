import pandas as pd
import numpy as np

import torch
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# DATA: https://www.nature.com/articles/s41597-020-00634-8

chromophore_df = pd.read_csv('/data/chromophores.csv')

column_headers = chromophore_df.columns.to_list()
chromophore_df = chromophore_df.drop(columns=[header for header in column_headers if header not in ("Chromophore", "Solvent", "Lifetime (ns)")])

# https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/

def get_atom_features(atom):
  permitted_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'Se', 'Te', 'Si', 'P', 'B', 'Sn', 'Ge']
  atom_type = [1 if atom.GetSymbol == x else 0 for x in permitted_atoms]

  hybridization = [
      1 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP else 0,
      1 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2 else 0,
      1 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3 else 0
  ]

  features = atom_type + hybridization + [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        1 if atom.GetIsAromatic() else 0
  ]

  return features

def get_bond_features(bond):
  bond_type = [
      1 if bond.GetBondType() == Chem.rdchem.BondType.SINGLE else 0,
      1 if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE else 0,
      1 if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE else 0,
      1 if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC else 0
  ]

  features = bond_type

  return features

def smiles_to_graph(smiles):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  
  node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
  x = torch.tensor(node_feats, dtype=torch.float)

  bond_indices = []
  bond_attrs = []

  for bond in mol.GetBonds():
    start_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()

    attr = get_bond_features(bond)

    bond_indices.append([start_idx, end_idx])
    bond_indices.append([end_idx, start_idx])

    bond_attrs.append(attr)
    bond_attrs.append(attr)

  edge_indices = torch.tensor(bond_indices, dtype=torch.float)
  edge_attrs = torch.tensor(bond_attrs, dtype=torch.float)

  data = Data(x=x, edge_index=edge_indices, edge_attr=edge_attrs)

  return data

  
