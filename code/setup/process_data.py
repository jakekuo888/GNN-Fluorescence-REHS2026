from wrangling.data_conversion import generate_and_export_data
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader

re_generate_data = False  # use this to toggle whether want to regenerate the data


class PredOption():
    def __init__(self, dataset, pred_label, out_folder, out_file):
        self.dataset = dataset

        if (self.dataset == "d4c"):
            self.mol_label = "Chromophore"
            self.sol_label = "Solvent"
        elif (self.dataset == "qmwf"):
            self.mol_label = "SMI"
            self.sol_label = "solvent"

        self.pred_label = pred_label
        self.out_folder = out_folder
        self.out_file = out_file


class FragmentDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


d4c_lifetime = PredOption("d4c", "Lifetime (ns)",
                          "lifetime-data", "lifetime-d4c.txt")
d4c_absorption = PredOption(
    "d4c", "Absorption max (nm)", "absorption-data", "absorption-d4c.txt")
qmwf_absorption = PredOption(
    "qmwf", "lambda_max (Exp nm)", "absorption-data", "absorption-qmwf.txt")

absorption_data_options = [d4c_absorption, qmwf_absorption]


def generate_graphs_labels(chosen_option, generate_data=False, y_mean=None, y_std=None, normalize=True):
    if generate_data:
        generate_and_export_data(chosen_option.dataset, chosen_option.mol_label, chosen_option.sol_label,
                                 chosen_option.pred_label, chosen_option.out_folder, chosen_option.out_file)

    # Load the lists of Data (graph) objects to process
    molecules_dicts = torch.load(
        f'./data/{chosen_option.out_folder}/molecularGraphs-{chosen_option.dataset}.pt', weights_only=False)  # list of PyG Data objects
    solvent_archive = np.load(
        f'./data/{chosen_option.out_folder}/solventFingerprints-{chosen_option.dataset}.npz')

    matrix_key = list(solvent_archive.keys())[0]
    solvent_matrix = solvent_archive[matrix_key]

    solvents_list = [row for row in solvent_matrix]

    # edge_features = molecules_list[0].num_edge_features
    solv_features = solvents_list[0].size

    # Fill the y-label list with the fluorescence times
    with open(f'./data/{chosen_option.out_folder}/{chosen_option.out_file}', 'r') as f:
        y_labels = [float(line)
                    for line in f.read().splitlines() if line.strip()]

    y_tensor = torch.tensor(y_labels, dtype=torch.float)

    # filter out zero or negative fluorescence times
    valid_mask = y_tensor > 0
    y_tensor = y_tensor[valid_mask]

    # filter molecules and solvents to match
    molecules_dicts = [mol for mol, valid in zip(
        molecules_dicts, valid_mask) if valid]
    # solvents_list = [sol for sol, valid in zip(solvents_list, valid_mask) if valid]

    for d, label in zip(molecules_dicts, y_tensor):
        # View(-1,1) returns size [1,1] because y_normalized is a tensor
        d["y_real"] = label.view(-1, 1)

    # Z-Score Standardization to fix Large Data Scale
    y_log = torch.log(y_tensor)

    if normalize:
        y_mean = y_log.mean()
        y_std = y_log.std()

    y_normalized = (y_log - y_mean) / y_std

    # Attach the y-labels to the x values
    for d, label in zip(molecules_dicts, y_normalized):
        # View(-1,1) returns size [1,1] because y_normalized is a tensor
        d["y_normalized"] = label.view(-1, 1)

    for data, fp in zip(molecules_dicts, solvents_list):
        d["sol_fp"] = fp

    smiles_for_similarity = []

    for d in molecules_dicts:
        smiles_for_similarity.append(d["smiles"])

    return molecules_dicts, y_mean, y_std, smiles_for_similarity, solv_features


def collate_fn(mol_dicts):
    # mol_dicts: whatever molecules the DataLoader's sampler selected
    # for this minibatch, in whatever (possibly shuffled) order it picked.

    all_fragments = []
    frags_per_mol = []

    for mol_dict in mol_dicts:
        frags = mol_dict["frag_graphs"]
        all_fragments.extend(frags)      # dump this molecule's fragments in
        # remember how many, in this exact order
        frags_per_mol.append(len(frags))

    atom_batch = Batch.from_data_list(all_fragments)
    return atom_batch, frags_per_mol, mol_dicts


def generate_GOMS(dicts, frags):
    GOMS = []

    for d, fs in zip(dicts, frags):
        n_frags = len(fs)
        x_ = torch.tensor(fs, dtype = torch.float)

        cut_bonds = []

        e_idx = 

        #each d is dict for mol x
        #fs is frags for mol x

        for f in fs:
            f
        
        graph = Data(
            x = x_,
            edge_index = e_idx,
            edge_attr = edge_attr,
            pos = pos,
            y = y,
        )

        GOMS.append(graph)

    return GOMS


if re_generate_data:
    for option in absorption_data_options:
        generate_graphs_labels(option, generate_data=True)
