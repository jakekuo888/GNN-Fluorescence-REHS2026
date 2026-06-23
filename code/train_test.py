import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from neural_networks import Model
from process_data import molecules_list, solvents_list
from process_data import y_std, y_mean

#EASY CONTROLS vvv
n_epochs = 100
collect_data = True
#EASY CONTROLS ^^^

# Split intro 3 datasets: the real training dataset for the supercomputer, and the sample train/test datasets for testing the initial model
mol_real_train_dataset, mol_split_dataset, sol_real_train_dataset, sol_split_dataset = train_test_split(
    molecules_list, solvents_list, test_size=0.8, random_state=42
)

mol_sample_train_dataset, mol_sample_test_dataset, sol_sample_train_dataset, sol_sample_test_dataset = train_test_split(
    mol_split_dataset, sol_split_dataset, test_size=0.2, random_state=42
)

# Data Loaders for each of the three for molecules and solvents
real_mol_train_loader = DataLoader(mol_real_train_dataset, batch_size=64, shuffle=False)
sample_mol_train_loader = DataLoader(mol_sample_train_dataset, batch_size=64, shuffle=False)
sample_mol_test_loader = DataLoader(mol_sample_test_dataset, batch_size=128, shuffle=False)

real_sol_train_loader = DataLoader(sol_real_train_dataset, batch_size=64, shuffle=False)
sample_sol_train_loader = DataLoader(sol_sample_train_dataset, batch_size=64, shuffle=False)
sample_sol_test_loader = DataLoader(sol_sample_test_dataset, batch_size=128, shuffle=False)

# Set up the Model class (GNN/FFNN), AdamW optimizer, and MSE Loss function
node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

model = Model(node_features, edge_features, 128, [128, 128, 128])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2.5e-4)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Train takes in mol & sol loader, zips them to return a forward pass through the model, loss, backprop, repeat
def train(mol_loader, sol_loader):
  model.train()

  for mol_data, sol_data in zip(mol_loader, sol_loader):
    out = model(
      mol_data.x, mol_data.edge_index, mol_data.edge_attr, mol_data.batch,
      sol_data.x, sol_data.edge_index, sol_data.edge_attr, sol_data.batch
    )
    loss = criterion(out, mol_data.y.view(-1,1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Train has the evaluation mode (output actual vs predicted for sample) and the non-evaluation mode (just avg MSE output)
def test(mol_loader, sol_loader, no_eval=True):
  model.eval()

  total_mse = 0.0
  total_graphs = 0

  with torch.no_grad():

    for mol_data, sol_data in zip(mol_loader, sol_loader):
      out = model(
        mol_data.x, mol_data.edge_index, mol_data.edge_attr, mol_data.batch,
        sol_data.x, sol_data.edge_index, sol_data.edge_attr, sol_data.batch
      )
      loss = criterion(out, mol_data.y.view(-1,1))

      if no_eval:
        num_graphs = mol_data.num_graphs

        total_mse += loss * num_graphs
        total_graphs += num_graphs
      else:
        # Reverse normalization for prediction
        pred_log = out.squeeze() * y_std + y_mean
        pred_actual = torch.exp(pred_log)
        pred_actual = pred_actual.flatten()

        # Reverse normalization for target too
        target_log = mol_data.y.flatten() * y_std + y_mean
        target_actual = torch.exp(target_log)

        with open("./data/plot-data/pred-act.txt", "w") as f_:
          for pred, target in zip(pred_actual, target_actual):
              print(f"Predicted: {pred.item():.4f}")
              print(f"Actual:    {target.item():.4f}")
              print("-" * 30)
              if(collect_data): print(f"{pred.item():.4f},{target.item():.4f}", file=f_)
              #load data so it can be used for plotting

  if no_eval:
    avg_mse = total_mse / total_graphs

    return avg_mse
  else:
    return 0.0

# Train & Test the Model
with open("./data/plot-data/MSE.txt", "w") as f_:
  for epoch in range(1,n_epochs+1):
    train(sample_mol_train_loader, sample_sol_train_loader)

    train_avg_mse = test(sample_mol_train_loader, sample_sol_train_loader)
    sample_avg_mse = test(sample_mol_test_loader, sample_sol_test_loader)
    scheduler.step(float(sample_avg_mse))

    print(f"Epoch #{epoch} | Train Average MSE: {train_avg_mse:.4f} | Test Average MSE: {sample_avg_mse:.4f}")
    if(collect_data): print(f"{train_avg_mse:.4f}, {sample_avg_mse:.4f}", file=f_) #loading data for plotting (train, test)

print("\nUNDERGOING TESTING\n-----------\n")
test(sample_mol_test_loader, sample_sol_test_loader, no_eval=False)