import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

import os
import sys

# Finds the root directory (one level up from main_script.py)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(root_dir, 'data-wrangling'))

from data_conversion import generate_and_export_data

class PredOption():
    def __init__(self, dataset, pred_label, out_folder, out_file):
        self.dataset = dataset

        if (self.dataset == "d4c"):
            self.csv = "d4c.csv"
            self.mol_label = "Chromophore"
            self.sol_label = "Solvent"
        elif (self.dataset == "qmwf"):
            self.csv = "qmwf.csv"
            self.mol_label = "SMI"
            self.sol_label = "solvent"

        self.pred_label = pred_label
        self.out_folder = out_folder
        self.out_file = out_file

d4c_lifetime = PredOption("d4c", "Lifetime (ns)", "lifetime-data", "lifetime-d4c.txt")
d4c_absorption = PredOption("d4c", "Absorption max (nm)", "absorption-data", "absorption-d4c.txt")
qmwf_absorption = PredOption("qmwf", "lambda_max (Exp,  nm)", "absorption-data", "absorption-qmwf.txt")

chosen_option = qmwf_absorption

option = 0
re_generate_data = True
if re_generate_data:
   generate_and_export_data(chosen_option.csv, chosen_option.mol_label, chosen_option.sol_label, chosen_option.pred_label, chosen_option.out_folder)

# Load the lists of Data (graph) objects to process
molecules_list = torch.load(f'./data/{chosen_option.out_folder}/molecularGraphs.pt', weights_only=False) # list of PyG Data objects
solvents_list = torch.load(f'./data/{chosen_option.out_folder}/solventGraphs.pt', weights_only=False)

edge_features = molecules_list[0].num_edge_features

for sol in solvents_list:
    if sol.edge_attr.dim() == 1 and sol.edge_attr.shape[0] == 0:
        sol.edge_attr = torch.zeros((0, edge_features), dtype=torch.float)
    if sol.edge_index.dim() == 1 and sol.edge_index.shape[0] == 0:
        sol.edge_index = torch.zeros((2, 0), dtype=torch.long)

# Fill the y-label list with the fluorescence times
y_labels = []

with open(f'./data/{chosen_option.out_folder}/{chosen_option.out_file}', 'r') as f:
  for line in f:
    line.strip()
    y_labels.append(float(line))

y_tensor = torch.tensor(y_labels, dtype=torch.float)

# Z-Score Standardization to fix Large Data Scale
y_log = torch.log(y_tensor)
y_mean = y_log.mean()
y_std = y_log.std()
y_normalized = (y_log - y_mean) / y_std

# Attach the y-labels to the x values
for data, label in zip(molecules_list, y_normalized):
    data.y = label.view(-1, 1) # View(-1,1) returns size [1,1] because y_normalized is a tensor