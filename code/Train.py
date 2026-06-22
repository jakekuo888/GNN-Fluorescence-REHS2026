import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from NeuralNet import Model

molecules_list = torch.load('data-wrangling/lifetime-data/molecularGraphs.pt', weights_only=False) # list of PyG Data objects
loader = DataLoader(molecules_list, batch_size=32, shuffle=True) # mini-batching

with open('./data-wrangling/lifetime-data/lifetime.txt', 'r') as f:
  fluorescence_times = f.read().splitlines()

y_tensor = torch.tensor(fluorescence_times, dtype=torch.float)

for data_obj, label in zip(loader, y_tensor):
  data_obj.y = label.view(-1, 1)

loader = DataLoader(molecules_list, batch_size=32, shuffle=True) # mini-batching

node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

model = Model(node_features, edge_features, 64, [64, 64, 64])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
  model.train()

  for data in loader:
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.zero_grad()

def test(loader):
  model.eval()

  total_mse = 0.0
  total_graphs = 0

  for data in loader:
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    loss = criterion(out, data.y)

    num_graphs = data.num_graphs

    total_mse += loss * num_graphs
    total_graphs += num_graphs

  avg_mse = total_mse / total_graphs

  return avg_mse

for epoch in range(1,101):
  train()
  test_acc = 0 # implement train split loader
  train_acc = 0 # implement test split loader
  print(f"Epoch #{epoch} | Test Accuracy: {test_acc:.4f} | Train Accuracy: {train_acc:.4f}")