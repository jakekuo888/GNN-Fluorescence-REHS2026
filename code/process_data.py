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

prediction_options = ["Absorption max (nm)", "Lifetime (ns)"]
prediction_folders = ["absorption-data", "lifetime-data"]
prediction_files = ["absorption.txt", "lifetime.txt"]

option = 0
re_generate_data = False
if re_generate_data:
   generate_and_export_data(prediction_options[option], prediction_folders[option])

# Load the lists of Data (graph) objects to process
molecules_list = torch.load(f'./data/{prediction_folders[option]}/molecularGraphs.pt', weights_only=False) # list of PyG Data objects
solvents_list = torch.load(f'./data/{prediction_folders[option]}/solventGraphs.pt', weights_only=False)

edge_features = molecules_list[0].num_edge_features

for sol in solvents_list:
    if sol.edge_attr.dim() == 1 and sol.edge_attr.shape[0] == 0:
        sol.edge_attr = torch.zeros((0, edge_features), dtype=torch.float)
    if sol.edge_index.dim() == 1 and sol.edge_index.shape[0] == 0:
        sol.edge_index = torch.zeros((2, 0), dtype=torch.long)

# Fill the y-label list with the fluorescence times
fluorescence_times = []

with open(f'./data/{prediction_folders[option]}/{prediction_files[option]}', 'r') as f:
  for line in f:
    line.strip()
    fluorescence_times.append(float(line))

y_tensor = torch.tensor(fluorescence_times, dtype=torch.float)

# Z-Score Standardization to fix Large Data Scale
y_log = torch.log(y_tensor)
y_mean = y_log.mean()
y_std = y_log.std()
y_normalized = (y_log - y_mean) / y_std

# Attach the y-labels to the x values
for data, label in zip(molecules_list, y_normalized):
    data.y = label.view(-1, 1) # View(-1,1) returns size [1,1] because y_normalized is a tensor