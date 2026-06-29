import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

import os
import sys

re_generate_data = False #use this to toggle whether want to regenerate the data


# Finds the root directory (one level up from main_script.py)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(root_dir, 'data-wrangling'))

from data_conversion import generate_and_export_data

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

d4c_lifetime = PredOption("d4c", "Lifetime (ns)", "lifetime-data", "lifetime-d4c.txt")
d4c_absorption = PredOption("d4c", "Absorption max (nm)", "absorption-data", "absorption-d4c.txt")
qmwf_absorption = PredOption("qmwf", "lambda_max (Exp nm)", "absorption-data", "absorption-qmwf.txt")

absorption_data_options = [d4c_absorption, qmwf_absorption]

def generate_graphs_labels(chosen_option, generate_data=False, y_mean=None, y_std=None, normalize=True):
    if generate_data:
        generate_and_export_data(chosen_option.dataset, chosen_option.mol_label, chosen_option.sol_label, chosen_option.pred_label, chosen_option.out_folder, chosen_option.out_file)

    # Load the lists of Data (graph) objects to process
    molecules_list = torch.load(f'./data/{chosen_option.out_folder}/molecularGraphs-{chosen_option.dataset}.pt', weights_only=False) # list of PyG Data objects
    # solvents_list = torch.load(f'./data/{chosen_option.out_folder}/solventGraphs-{chosen_option.dataset}.pt', weights_only=False)
    solvent_archive = np.load(f'./data/{chosen_option.out_folder}/solventFingerprints-{chosen_option.dataset}.npz')

    matrix_key = list(solvent_archive.keys())[0] 
    solvent_matrix = solvent_archive[matrix_key]

    solvents_list = [row for row in solvent_matrix]

    edge_features = molecules_list[0].num_edge_features
    solv_features = solvents_list[0].size

    # Fill the y-label list with the fluorescence times
    with open(f'./data/{chosen_option.out_folder}/{chosen_option.out_file}', 'r') as f:
        y_labels = [float(line) for line in f.read().splitlines() if line.strip()]

    y_tensor = torch.tensor(y_labels, dtype=torch.float)

    # filter out zero or negative fluorescence times
    valid_mask = y_tensor > 0
    y_tensor = y_tensor[valid_mask]

    # filter molecules and solvents to match
    molecules_list = [mol for mol, valid in zip(molecules_list, valid_mask) if valid]
    # solvents_list = [sol for sol, valid in zip(solvents_list, valid_mask) if valid]

    # Z-Score Standardization to fix Large Data Scale
    y_log = torch.log(y_tensor)

    if normalize:
        y_mean = y_log.mean()
        y_std = y_log.std()
    
    y_normalized = (y_log - y_mean) / y_std

    # Attach the y-labels to the x values
    for data, label in zip(molecules_list, y_normalized):
        data.y = label.view(-1, 1) # View(-1,1) returns size [1,1] because y_normalized is a tensor

    for data, fp in zip(molecules_list, solvents_list):
        data.sol_fp = fp

    smiles_for_similarity = []

    for data in molecules_list:
        smiles_for_similarity.append(data.smiles)

    return molecules_list, y_mean, y_std, smiles_for_similarity, solv_features
