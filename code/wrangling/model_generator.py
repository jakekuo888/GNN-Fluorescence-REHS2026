import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdPartialCharges
import numpy as np
import cirpy
from rdkit import rdBase

import json
import os

#https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/


def get_atom_features(atom):
  permitted_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'Se', 'Te', 'Si', 'P', 'B', 'Sn', 'Ge']
  #one-hot everything
  atom_type = [int(atom.GetSymbol() == x) for x in permitted_atoms]

  atomH = atom.GetHybridization()
  hybridization = [
      int(atomH == Chem.rdchem.HybridizationType.SP),
      int(atomH == Chem.rdchem.HybridizationType.SP2),
      int(atomH == Chem.rdchem.HybridizationType.SP3)
  ]

  charge = float(atom.GetDoubleProp('_GasteigerCharge'))
  if not np.isfinite(charge):
      charge = 0.0

  chirality_options = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
  ]
  chirality = [int(atom.GetChiralTag() == c) for c in chirality_options]

  features = atom_type + hybridization + chirality + [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        charge
  ]

  return features

def get_bond_features(bond):
  bondGBT = bond.GetBondType()

  bond_type = [
      int(bondGBT == Chem.rdchem.BondType.SINGLE),
      int(bondGBT == Chem.rdchem.BondType.DOUBLE),
      int(bondGBT == Chem.rdchem.BondType.TRIPLE),
      int(bondGBT == Chem.rdchem.BondType.AROMATIC)
  ]

  features = bond_type + [
      int(bond.IsInRing()),
      int(bond.GetIsConjugated())
  ]

  return features

def resolve_smiles(name, dictionary, file):
    blocker = rdBase.BlockLogs()

    SOLVENT_MANUAL_MAP = {
        # spacing variants of DCM
        'CH2Cl2': 'ClCCl',
        'CH 2 Cl 2': 'ClCCl',
        'CH 2 Cl': 'ClCCl',
        # acetonitrile
        'CH3CN': 'CC#N',
        'CH 3 CN': 'CC#N',
        # methanol
        'CH3OH': 'CO',
        'CH 3 OH': 'CO',
        # benzene
        'C6H6': 'c1ccccc1',
        # cyclohexane
        'C6H12': 'C1CCCCC1',
        # ethyl acetate
        'EtOAc': 'CCOC(C)=O',
        # MTHF (2-methyltetrahydrofuran)
        'MTHF': 'C1CCC(C)O1',
        # benzonitrile
        'PhCN': 'N#Cc1ccccc1',
        # DMSO typo
        'dimethylsufoxide': 'CS(C)=O',
        # water
        'H 2 O': 'O',
    }

    mol = Chem.MolFromSmiles(str(name))
    if mol is None:
        if name in SOLVENT_MANUAL_MAP:
            return SOLVENT_MANUAL_MAP[name]
        if name in dictionary:
            return dictionary[name]
        else:
            print(f"Fetching SMILES for: {name}")
            result = cirpy.resolve(name, 'smiles')
            dictionary[name] = result  # cache even if None
            
            # save updated cache to disk
            with open(file, 'w') as f:
                json.dump(dictionary, f)
            
            return result
    else:
        return name

def smiles_to_graph(smiles):
  blocker = rdBase.BlockLogs()

  CACHE_FILE = './data/solvent_cache.json'
  if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        SOLVENT_SMILES = json.load(f)
  else:
      SOLVENT_SMILES = {}
  
  NUM_NODE_FEATURES = 27
  NUM_EDGE_FEATURES = 6

  empty_graph = Data(
            x=torch.zeros((1, NUM_NODE_FEATURES), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, NUM_EDGE_FEATURES), dtype=torch.float)
  )

  mol = Chem.MolFromSmiles(str(smiles))
  if mol is None:
      resolved = resolve_smiles(str(smiles), SOLVENT_SMILES, CACHE_FILE)
      if resolved is None:
          return None
      mol = Chem.MolFromSmiles(resolved)
      if mol is None:
          return None

  mol = Chem.AddHs(mol)

  rdPartialCharges.ComputeGasteigerCharges(mol)
  
  node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
  x = torch.tensor(node_feats, dtype=torch.float)

  bond_indices = []
  bond_attrs = []

  for bond in mol.GetBonds():
    start_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()

    attr = get_bond_features(bond)

    #do twice so it's treated like an undirected graph
    bond_indices.append([start_idx, end_idx])
    bond_indices.append([end_idx, start_idx])

    bond_attrs.append(attr)
    bond_attrs.append(attr)

  edge_indices = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
  edge_attrs = torch.tensor(bond_attrs, dtype=torch.float)

  data = Data(x=x, edge_index=edge_indices, edge_attr=edge_attrs)

  return data

def smiles_to_morgan_fp(fp_gen, smiles):
	mol = Chem.MolFromSmiles(smiles)
	bit_vect = fp_gen.GetFingerprint(mol)
	fp_array = np.zeros((2048,), dtype=np.int8)
	Chem.DataStructs.ConvertToNumpyArray(bit_vect, fp_array)

	return fp_array