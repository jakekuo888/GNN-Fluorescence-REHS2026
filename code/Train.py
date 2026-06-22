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

for data_obj, label in zip(molecules_list, y_normalized):
  data_obj.y = label.view(-1, 1)
  processed_dataset.append(data_obj)

train_dataset, split_dataset = train_test_split(processed_dataset, test_size=0.2, random_state=42)
test_dataset, sample_dataset = train_test_split(split_dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
sample_loader = DataLoader(sample_dataset, batch_size=128, shuffle=False)

node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

model = Model(node_features, edge_features, 128, [128, 128, 128])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train(loader):
  model.train()

  for data in loader:
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    loss = criterion(out, data.y.view(-1, 1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test(loader):
  model.eval()

  total_mse = 0.0
  total_graphs = 0

  with torch.no_grad():

    for data in loader:
      out = model(data.x, data.edge_index, data.edge_attr, data.batch)
      loss = criterion(out, data.y.view(-1, 1))

      num_graphs = data.num_graphs

      total_mse += loss * num_graphs
      total_graphs += num_graphs

  avg_mse = total_mse / total_graphs

  return avg_mse

for epoch in range(1,11):
  train(test_loader)
  train_avg_mse = test(test_loader)
  sample_avg_mse = test(sample_loader)
  print(f"Epoch #{epoch} | Train Average MSE: {train_avg_mse:.4f} | Test Average MSE: {sample_avg_mse:.4f}")
  #print(f"Epoch #{epoch} | Test Accuracy: {test_acc:.4f} | Train Accuracy: {train_acc:.4f}")

def evaluate(loader):
    model.eval()

    with torch.no_grad():

        for data in loader:
            out = model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch
            )

            # Reverse normalization ONLY for prediction
            pred_log = out.squeeze() * y_std + y_mean
            pred_actual = torch.exp(pred_log)

            pred_actual = pred_actual.flatten()
            target_actual = data.y.flatten()

            for pred, target in zip(pred_actual, target_actual):
                print(f"Predicted: {pred.item():.4f}")
                print(f"Actual:    {target.item():.4f}")
                print("-" * 30)


print("UNDERGOING TESTING\n-----------\n")

eval_loader = DataLoader(
    sample_dataset,
    batch_size=128,
    shuffle=False
)

evaluate(eval_loader)