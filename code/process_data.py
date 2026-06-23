import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

molecules_list = torch.load('./data/lifetime-data/molecularGraphs.pt', weights_only=False) # list of PyG Data objects
solvents_list = torch.load('./data/lifetime-data/solventGraphs.pt', weights_only=False)

fluorescence_times = []

with open('./data/lifetime-data/lifetime.txt', 'r') as f:
  for line in f:
    line.strip()
    fluorescence_times.append(float(line))

y_tensor = torch.tensor(fluorescence_times, dtype=torch.float)

# Z-Score Standardization to fix Large Data Scale
y_log = torch.log(y_tensor)
y_mean = y_log.mean()
y_std = y_log.std()
y_normalized = (y_log - y_mean) / y_std

for data, label in zip(molecules_list, y_normalized):
    data.y = label.view(-1, 1)