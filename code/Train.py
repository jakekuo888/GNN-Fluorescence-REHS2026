import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from NeuralNet import Model

molecules_list = torch.load('./code/data-wrangling/lifetime-data/molecularGraphs.pt', weights_only=False) # list of PyG Data objects

fluorescence_times = []

with open('./code/data-wrangling/lifetime-data/lifetime.txt', 'r') as f:
  for line in f:
    line.strip()
    fluorescence_times.append(float(line))

y_tensor = torch.tensor(fluorescence_times, dtype=torch.float)

# Z-Score Standardization to fix Large Data Scale
y_log = torch.log(y_tensor)
y_mean = y_log.mean()
y_std = y_log.std()
y_normalized = (y_log - y_mean) / y_std

processed_dataset = []

for data_obj, label in zip(molecules_list, y_tensor):
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
    loss = criterion(out, data.y.view(-1, 1))
    optimizer.zero_grad()
    loss.backward()

def test(loader):
  model.eval()

  total_mse = 0.0
  total_graphs = 0

  for data in loader:
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    loss = criterion(out, data.y.view(-1, 1))

    num_graphs = data.num_graphs

    total_mse += loss * num_graphs
    total_graphs += num_graphs

  avg_mse = total_mse / total_graphs

  return avg_mse

for epoch in range(1,101):
  train(sample_loader)
  sample_avg_mse = test(sample_loader)
  print(f"Epoch #{epoch} | Sample Average MSE: {sample_avg_mse:.4f}")
  #print(f"Epoch #{epoch} | Test Accuracy: {test_acc:.4f} | Train Accuracy: {train_acc:.4f}")