#this is the main place we code; we can change the name of it later depending on the structure of the project I guess
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