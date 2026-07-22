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
from random import randint

import datamol as dm

import json
import os

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

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


def get_fallback_bond_indices(mol: Chem.Mol):
    bonds_to_break = [b[0] for b in BRICS.FindBRICSBonds(mol)]

    if bonds_to_break:
        # Extract the bond IDs for BRICS
        bond_indices = [mol.GetBondBetweenAtoms(
            i, j).GetIdx() for i, j in bonds_to_break]
        return bonds_to_break, bond_indices, "brics"

    # 2. Fallback to Fraggle style if BRICS found 0 bonds
    fraggle_bond_indices = []
    fraggle_bonds_to_break = []

    # FraggleSim relies primarily on breaking non-ring (acyclic) bonds
    for bond in mol.GetBonds():
        # Fraggle looks for single acyclic cuts that don't isolate single atoms
        # (like cutting a terminal methyl off). We mimic that logic here:
        if not bond.IsInRing():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            # Ensure we aren't just chopping off a terminal heavy atom (e.g., -CH3, -F, -OH)
            if begin_atom.GetDegree() > 1 and end_atom.GetDegree() > 1:
                atom_pair = (begin_atom.GetIdx(), end_atom.GetIdx())
                fraggle_bonds_to_break.append(atom_pair)
                fraggle_bond_indices.append(bond.GetIdx())

    return fraggle_bonds_to_break, fraggle_bond_indices, "fraggle"


def return_frags(mol, graph):
    bric_bonds = list(BRICS.FindBRICSBonds(mol))
    bonds_to_break = [b[0] for b in bric_bonds]
    bonds_name = [b[1] for b in bric_bonds]

    bond_indices = [mol.GetBondBetweenAtoms(
        i, j).GetIdx() for i, j in bonds_to_break]

    if len(bond_indices) > 0:
        frag_mol = rdmolops.FragmentOnBonds(
            mol, bond_indices, addDummies=False)
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
                # original global coords, untouched
                pos=graph.pos[subset],
                edge_index=sub_edge_index,
                edge_attr=sub_edge_attr,
            )
            fragment_graphs.append(frag_data)
    else:
        frag_data = Data(
            x=graph.x,
            pos=graph.pos,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
        )
        fragment_graphs = [frag_data]
        atom_groups = Chem.GetMolFrags(mol, asMols=False)

    atom_to_frag = {}
    for frag_id, atom_idx_group in enumerate(atom_groups):
        for atom_idx in atom_idx_group:
            atom_to_frag[atom_idx] = frag_id

    gs_edges = set()
    for atom_i, atom_j in bonds_to_break:
        frag_a, frag_b = atom_to_frag[atom_i], atom_to_frag[atom_j]
        if frag_a != frag_b:
            gs_edges.add((min(frag_a, frag_b), max(frag_a, frag_b)))

    frag_brics_types = {frag_id: set() for frag_id in range(len(atom_groups))}
    for (atom_i, atom_j), (label_i, label_j) in zip(bonds_to_break, bonds_name):
        frag_a = atom_to_frag[atom_i]
        frag_b = atom_to_frag[atom_j]
        frag_brics_types[frag_a].add(label_i)
        frag_brics_types[frag_b].add(label_j)


    frag_fps = [
        fp_gen.GetFingerprint(mol=mol, fromAtoms=list(g)) for g in atom_groups
    ]

    fragmentation_output = {
        "frag_graphs": fragment_graphs,
        "atom_groups": atom_groups,
        "atom_to_frag_map": atom_to_frag,
        "gs_edges": gs_edges,
        "frag_fps": frag_fps,
        "entire_graph": graph,
        "smiles": graph.smiles,
        "frag_brics_types": frag_brics_types,
    }

    final_output = gen_data(fragmentation_output)

    return final_output


def gen_data(dict_):
    # objective: return a bigger dict to be make one-liner GNN
    n_frags = len(dict_["frag_graphs"])
    frag_graphs = dict_["frag_graphs"]

    entire_graph = dict_['entire_graph']
    atom_to_frag = dict_['atom_to_frag_map']
    cut_bonds = dict_['gs_edges']
    atom_groups = dict_['atom_groups']
    smiles = dict_['smiles']

    pair_to_attr = {}
    try:
        src_all, dst_all = entire_graph.edge_index
    except Exception as e:
        print(f"CRITICAL ERROR IN SRC, DST: {smiles}")
        return None

    for k in range(entire_graph.edge_index.size(1)):
        a1, a2 = src_all[k].item(), dst_all[k].item()
        f1, f2 = atom_to_frag[a1], atom_to_frag[a2]
        if f1 != f2:
            key = (min(f1, f2), max(f1, f2))
            if key not in pair_to_attr:
                pair_to_attr[key] = entire_graph.edge_attr[k]

    src, dst, edge_attr_list = [], [], []
    for (f1, f2) in cut_bonds:
        try:
            attr = pair_to_attr[(min(f1, f2), max(f1, f2))]
        except Exception as e:
            print(f"CRITICAL ERROR: MAX(f1, f2) FAILED-- " + dict_["smiles"])
            return None
        src += [f1, f2]
        dst += [f2, f1]
        edge_attr_list += [attr, attr]

    if edge_attr_list:
        e_idx = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.stack(edge_attr_list)
    else:
        e_idx = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(
            (0, entire_graph.edge_attr.size(-1)), dtype=torch.float)

    entire_pos = entire_graph.pos
    if not torch.is_tensor(entire_pos):
        entire_pos = torch.tensor(entire_pos, dtype=torch.float)

    pos = torch.stack([
        entire_pos[list(atom_groups[f_idx])].mean(dim=0)
        for f_idx in range(n_frags)
    ])

    dict_['frag_graphs'] = frag_graphs
    dict_['edge_index'] = e_idx
    dict_['edge_attr'] = edge_attr
    dict_['pos'] = pos
    dict_['n_frags'] = n_frags

    return dict_


def optimize_conformer(mol, max_iters=2000):
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is not None:
        result = rdForceFieldHelpers.MMFFOptimizeMolecule(
            mol, maxIters=max_iters)
        if result == 0:
            return mol, "MMFF"
        elif result == 1:
            # ran out of iterations, not a real failure -- geometry is still meaningfully relaxed
            return mol, "MMFF_partial"

    # only reach here on a genuine parameter failure (-1), not a slow-converging molecule
    result = rdForceFieldHelpers.UFFOptimizeMolecule(mol, maxIters=max_iters)
    if result == 0:
        return mol, "UFF"
    elif result == 1:
        return mol, "UFF_partial"

    return mol, "unoptimized"


def embed_conformer(mol):
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.maxIterations = 1000
    params.ignoreSmoothingFailures = True

    conf_id = rdDistGeom.EmbedMolecule(mol, params)
    if conf_id == -1:
        # last resort -- pure random-distance-geometry, no torsion knowledge
        conf_id = rdDistGeom.EmbedMolecule(
            mol, useRandomCoords=True, maxAttempts=2000)

    return conf_id  # still -1 if truly unembeddable -- caller must check


CACHE_FILE = './data/solvent_cache.json'

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        SOLVENT_SMILES = json.load(f)
else:
    SOLVENT_SMILES = {}


def smiles_to_graph(smiles):
    blocker = rdBase.BlockLogs()

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

    conf_id = embed_conformer(mol)
    if conf_id == -1:
        print(f"\n Error: Conformer failed - {smiles}")
        return None

    try:
        mol, method = optimize_conformer(mol)
        if method == "unoptimized":
            print(f"\n Warning: Unoptimized molecule, can proceed safely - {smiles}")
    except Exception as e:
        print(f"\n Error: MMFF failed - {smiles}")
        return None

    try:
        conf = mol.GetConformer()
    except Exception as e:
        print(f"n Error: Conformer failed - {smiles}")
        return None
    positions = conf.GetPositions()
    positions = torch.tensor(positions, dtype=torch.float)

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
