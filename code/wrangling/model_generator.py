import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
import numpy as np
import cirpy
from rdkit import rdBase
from rdkit.Chem import BRICS, rdmolops
from torch_geometric.utils import subgraph

import json
import os

# https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/


def get_atom_features(atom):
    permitted_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl',
                       'Br', 'I', 'Se', 'Te', 'Si', 'P', 'B', 'Sn', 'Ge']
    # one-hot everything
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
        'H2O': 'O'
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


def return_frags(mol, graph):
    bonds_to_break = [b[0] for b in BRICS.FindBRICSBonds(mol)]
    bond_indices = [mol.GetBondBetweenAtoms(
        i, j).GetIdx() for i, j in bonds_to_break]

    frag_mol = rdmolops.FragmentOnBonds(mol, bond_indices, addDummies=False)
    atom_groups = Chem.GetMolFrags(frag_mol, asMols=False)

    fragment_graphs = []

    for atom_idx_group in atom_groups:
        subset = torch.tensor(atom_idx_group, dtype=torch.long)
        sub_edge_index, sub_edge_attr = subgraph(
            subset, graph.edge_index, graph.edge_attr,
            relabel_nodes=True, num_nodes=graph.num_nodes
        )
        frag_data = Data(
            x=graph.x[subset],
            pos=graph.pos[subset],          # original global coords, untouched
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
        )
        fragment_graphs.append(frag_data)

    atom_to_frag = {}
    for frag_id, atom_idx_group in enumerate(atom_groups):
        for atom_idx in atom_idx_group:
            atom_to_frag[atom_idx] = frag_id

    gs_edges = set()
    for atom_i, atom_j in bonds_to_break:
        frag_a, frag_b = atom_to_frag[atom_i], atom_to_frag[atom_j]
        gs_edges.add((min(frag_a, frag_b), max(frag_a, frag_b)))

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    frag_fps = [
        fp_gen.GetFingerprint(mol=mol, fromAtoms=list(g)) for g in atom_groups
    ]

    fragmentation_output = {
        "frag_graphs": fragment_graphs,
        "atom_groups": atom_groups,
        "atom_to_frag_map": atom_to_frag,
        "cut_bonds": gs_edges,
        "frag_fps": frag_fps,
        "entire_graph": graph,
        "smiles": graph.smiles
    }

    return fragmentation_output

def gen_data(dict_, frag):
    #objective: return a bigger dict to be make one-liner GNN
    n_frags = len(frag)
    x_ = torch.tensor(frag, dtype = torch.float)

    entire_graph = dict_['entire_graph']
    atom_to_frag = dict_['atom_to_frag_map']
    cut_bonds = dict_['cut_bonds']
    atom_groups = dict_['atom_groups']

    pair_to_attr = {}
    src_all, dst_all = entire_graph.edge_index
    for k in range(entire_graph.edge_index.size(1)):
        a1, a2 = src_all[k].item(), dst_all[k].item()
        f1, f2 = atom_to_frag[a1], atom_to_frag[a2]
        if f1 != f2:
            key = (min(f1, f2), max(f1, f2))
            if key not in pair_to_attr:
                pair_to_attr[key] = entire_graph.edge_attr[k]

    src, dst, edge_attr_list = [], [], []
    for (f1, f2) in cut_bonds:
        attr = pair_to_attr[(min(f1, f2), max(f1, f2))]
        src += [f1, f2]
        dst += [f2, f1]
        edge_attr_list += [attr, attr]

    if edge_attr_list:
        e_idx = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.stack(edge_attr_list)
    else:
        e_idx = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, entire_graph.edge_attr.size(-1)), dtype=torch.float)

    entire_pos = entire_graph.pos
    if not torch.is_tensor(entire_pos):
        entire_pos = torch.tensor(entire_pos, dtype = torch.float)

    pos = torch.stack([
        entire_pos[list(atom_groups[f_idx])].mean(dim=0)
        for f_idx in range(n_frags)
    ])

    dict_['x'] = x_
    dict_['edge_index'] = e_idx
    dict_['edge_attr'] = edge_attr
    dict_['pos'] = pos
    dict_['y'] = dict_['y_norm']
    dict_['n_frags'] = n_frags

    return dict_

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

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        resolved = resolve_smiles(str(smiles), SOLVENT_SMILES, CACHE_FILE)
        if resolved is None:
            return None
        mol = Chem.MolFromSmiles(resolved)
        if mol is None:
            return None

    mol = Chem.AddHs(mol)
    rdDistGeom.EmbedMolecule(mol)
    rdForceFieldHelpers.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    positions = conf.GetPositions()

    rdPartialCharges.ComputeGasteigerCharges(mol)

    node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    bond_indices = []
    bond_attrs = []

    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        attr = get_bond_features(bond)

        # do twice so it's treated like an undirected graph
        bond_indices.append([start_idx, end_idx])
        bond_indices.append([end_idx, start_idx])

        bond_attrs.append(attr)
        bond_attrs.append(attr)

    edge_indices = torch.tensor(
        bond_indices, dtype=torch.long).t().contiguous()
    edge_attrs = torch.tensor(bond_attrs, dtype=torch.float)

    data = Data(x=x, pos=positions, edge_index=edge_indices,
                edge_attr=edge_attrs)

    data.smiles = smiles

    return return_frags(mol, data)


def smiles_to_morgan_fp(fp_gen, smiles):
    mol = Chem.MolFromSmiles(smiles)
    bit_vect = fp_gen.GetFingerprint(mol)
    fp_array = np.zeros((2048,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(bit_vect, fp_array)

    return fp_array
