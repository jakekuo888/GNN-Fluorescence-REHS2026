import torch
from torch_geometric.data import Data
from rdkit import Chem

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

  features = atom_type + hybridization + [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())
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

  features = bond_type

  return features

def smiles_to_graph(smiles):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    print(f"ERR: {smiles} could not be parsed")
    return None
  
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