import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from NeuralNet import Model

molecules_list = torch.load('data-wrangling/lifetime-data/molecularGraphs.pt', weights_only=False) # list of PyG Data objects
loader = DataLoader(molecules_list, batch_size=32, shuffle=True) # mini-batching

with open('./data-wrangling/lifetime-data/lifetime.txt', 'r') as f:
  fluorescence_times = f.read().splitlines()

y_tensor = torch.tensor(fluorescence_times, dtype=torch.float)

processed_dataset = []

for data_obj, label in zip(loader, y_tensor):
  data_obj.y = label.view(-1, 1)
  processed_dataset.append(data_obj)

train_dataset, split_dataset = train_test_split(processed_dataset, test_size=0.2, random_state=42)
test_dataset, sample_dataset = train_test_split(split_dataset, test_size=0.01, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
sample_loader = DataLoader(sample_dataset, batch_size=4, shuffle=True)

node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

model = Model(node_features, edge_features, 64, [64, 64, 64])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train(loader):
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
  train(sample_loader)
  sample_acc = test(sample_loader)
  print(f"Epoch #{epoch} | Test Accuracy: {sample_acc:.4f}")
  #print(f"Epoch #{epoch} | Test Accuracy: {test_acc:.4f} | Train Accuracy: {train_acc:.4f}")