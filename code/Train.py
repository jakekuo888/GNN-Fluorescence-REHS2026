import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from NeuralNet import Model

molecules_list = torch.load('./code/data-wrangling/lifetime-data/molecularGraphs.pt', weights_only=False) # list of PyG Data objects
solvents_list = torch.load('./code/data-wrangling/lifetime-data/solventGraphs.pt', weights_only=False)

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

mol_processed_dataset = []
sol_processed_dataset = []

for data_obj, label in zip(molecules_list, y_normalized):
  data_obj.y = label.view(-1, 1)
  mol_processed_dataset.append(data_obj)

for data_obj, label in zip(solvents_list, y_normalized):
  data_obj.y = label.view(-1, 1)
  sol_processed_dataset.append(data_obj)

real_mol_train_dataset, mol_split_dataset, real_sol_train_dataset, sol_split_dataset = train_test_split(mol_processed_dataset, sol_processed_dataset, test_size=0.2, random_state=42, shuffle=True)
mol_train_dataset, mol_test_dataset, sol_train_dataset, sol_test_dataset = train_test_split(mol_split_dataset, sol_split_dataset, test_size=0.2, random_state=42)

mol_real_train_loader = DataLoader(real_mol_train_dataset, batch_size=128, shuffle=False)
mol_train_loader = DataLoader(mol_train_dataset, batch_size=128, shuffle=False)
mol_test_loader = DataLoader(mol_test_dataset, batch_size=128, shuffle=False)

sol_real_train_loader = DataLoader(real_sol_train_dataset, batch_size=128, shuffle=False)
sol_train_loader = DataLoader(sol_train_dataset, batch_size=128, shuffle=False)
sol_test_loader = DataLoader(sol_test_dataset, batch_size=128, shuffle=False)

node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

model = Model(node_features, edge_features, 128, [128, 128, 128])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train(mol_loader, sol_loader):
  model.train()

  for mol_data, sol_data in zip(mol_loader, sol_loader):
    out = model(mol_data.x, mol_data.edge_index, mol_data.edge_attr, mol_data.batch, sol_data.x, sol_data.edge_index, sol_data.edge_attr, sol_data.batch)
    loss = criterion(out, mol_data.y.view(-1, 1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test(mol_loader, sol_loader):
  model.eval()

  total_mse = 0.0
  total_graphs = 0

  with torch.no_grad():

    for mol_data, sol_data in zip(mol_loader, sol_loader):
      out = model(mol_data.x, mol_data.edge_index, mol_data.edge_attr, mol_data.batch, sol_data.x, sol_data.edge_index, sol_data.edge_attr, sol_data.batch)
      loss = criterion(out, mol_data.y.view(-1, 1))

      num_graphs = mol_data.num_graphs

      total_mse += loss * num_graphs
      total_graphs += num_graphs

  avg_mse = total_mse / total_graphs

  return avg_mse

for epoch in range(1,301):
  train(mol_train_loader, sol_train_loader)
  train_avg_mse = test(mol_train_loader, sol_train_loader)
  sample_avg_mse = test(mol_test_loader, sol_test_loader)
  print(f"Epoch #{epoch} | Train Average MSE: {train_avg_mse:.4f} | Test Average MSE: {sample_avg_mse:.4f}")
  #print(f"Epoch #{epoch} | Test Accuracy: {test_acc:.4f} | Train Accuracy: {train_acc:.4f}")

def evaluate(mol_loader, sol_loader):
    model.eval()
    with torch.no_grad():
      for mol_data, sol_data in zip(mol_loader, sol_loader):
        out = model(mol_data.x, mol_data.edge_index, mol_data.edge_attr, mol_data.batch, sol_data.x, sol_data.edge_index, sol_data.edge_attr, sol_data.batch)
        # Reverse normalization for prediction
        pred_log = out.squeeze() * y_std + y_mean
        pred_actual = torch.exp(pred_log)
        pred_actual = pred_actual.flatten()

        # Reverse normalization for target too (this was missing)
        target_log = mol_data.y.flatten() * y_std + y_mean
        target_actual = torch.exp(target_log)

        for pred, target in zip(pred_actual, target_actual):
            print(f"Predicted: {pred.item():.4f}")
            print(f"Actual:    {target.item():.4f}")
            print("-" * 30)

print("UNDERGOING TESTING\n-----------\n")

evaluate(mol_test_loader, sol_test_loader)