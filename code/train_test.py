import torch
import sys
import subprocess
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from neural_networks import Model
from early_stop import EarlyStop

from process_data import molecules_list, solvents_list
from process_data import y_std, y_mean

#EASY CONTROLS vvv
n_epochs = 100
collect_data = True
early_stopper = EarlyStop(9, 0.005)
#EASY CONTROLS ^^^

# Split intro 3 datasets: the real training dataset for the supercomputer, and the sample train/test datasets for testing the initial model
mol_train_dataset, mol_split_dataset, sol_train_dataset, sol_split_dataset = train_test_split(
    molecules_list, solvents_list, test_size=0.2, random_state=42
)

mol_validate_dataset, mol_test_dataset, sol_validate_dataset, sol_test_dataset = train_test_split(
    mol_split_dataset, sol_split_dataset, test_size=0.5, random_state=42
)

# Data Loaders for each of the three for molecules and solvents
mol_train_loader = DataLoader(mol_train_dataset, batch_size=64, shuffle=False)
mol_validate_loader = DataLoader(mol_validate_dataset, batch_size=64, shuffle=False)
mol_test_loader = DataLoader(mol_test_dataset, batch_size=128, shuffle=False)

sol_train_loader = DataLoader(sol_train_dataset, batch_size=64, shuffle=False)
sol_validate_loader = DataLoader(sol_validate_dataset, batch_size=64, shuffle=False)
sol_test_loader = DataLoader(sol_test_dataset, batch_size=128, shuffle=False)

# Set up the Model class (GNN/FFNN), AdamW optimizer, and MSE Loss function
node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

model = Model(node_features, edge_features, 128, [128, 128, 128])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
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
def test(mol_loader, sol_loader, no_eval=True, print_diff=False):
  model.eval()

  total_mse = 0.0
  total_graphs = 0
  f_ = open("./data/plot-data/pred-act.txt", "w") if not no_eval else None

  with torch.no_grad():

    for mol_data, sol_data in zip(mol_loader, sol_loader):
      out = model(
        mol_data.x, mol_data.edge_index, mol_data.edge_attr, mol_data.batch,
        sol_data.x, sol_data.edge_index, sol_data.edge_attr, sol_data.batch
      )
      
      # Reverse normalization for prediction
      pred_log = out.squeeze() * y_std + y_mean
      pred_actual = torch.exp(pred_log)
      pred_actual = pred_actual.flatten()

      # Reverse normalization for target too
      target_log = mol_data.y.flatten() * y_std + y_mean
      target_actual = torch.exp(target_log)

      loss = torch.mean(torch.abs(pred_actual - target_actual))

      if no_eval:
        num_graphs = mol_data.num_graphs

        total_mse += loss * num_graphs
        total_graphs += num_graphs
      if print_diff:
        # Reverse normalization for prediction
        pred_log = out.squeeze() * y_std + y_mean
        pred_actual = torch.exp(pred_log)
        pred_actual = pred_actual.flatten()

        # Reverse normalization for target too
        target_log = mol_data.y.flatten() * y_std + y_mean
        target_actual = torch.exp(target_log)

        for pred, target in zip(pred_actual, target_actual):
          print(f"Predicted: {pred.item():.4f}")
          print(f"Actual:    {target.item():.4f}")
          print("-" * 30)
          if(collect_data): print(f"{pred.item():.4f},{target.item():.4f}", file=f_)
          #load data so it can be used for plotting

  if f_ is not None:
    f_.close()
  
  if no_eval:
    avg_mse = total_mse / total_graphs

    return avg_mse
  else:
    return 0.0

# Train & Test the Model
with open("./data/plot-data/MSE.txt", "w") as f_:
  for epoch in range(1,n_epochs+1):
    train(mol_train_loader, sol_train_loader)

    train_avg_mse = test(mol_train_loader, sol_train_loader)
    sample_avg_mse = test(mol_validate_loader, sol_validate_loader)
    scheduler.step(float(sample_avg_mse))

    if early_stopper.stop_early(sample_avg_mse, model):
      print(f'Early stop has been initiated on Epoch #{epoch}')
      early_stopper.restore_best(model)
      break

    print(f"Epoch #{epoch} | Train Average MAE: {train_avg_mse:.4f} | Test Average MAE: {sample_avg_mse:.4f} | Early stopper count: {early_stopper.count}")
    if(collect_data): print(f"{train_avg_mse:.4f}, {sample_avg_mse:.4f}", file=f_) #loading data for plotting (train, test)

print("\nUNDERGOING TESTING\n-----------\n")
test_avg_mse = test(mol_test_loader, sol_test_loader, no_eval=True, print_diff=True)
print(f"\n------------------------------\nTEST AVERAGE MAE (FINAL RESULTS): {test_avg_mse}\n------------------------------")

if(collect_data):
  want_visuals = input("\n Do you want to create Visuals (Y/N): ").lower()
  if(want_visuals == 'y'):
    subprocess.run([sys.executable, "./plots-visuals/plot-mse.py"])
    print("Visuals sucessfully created!\n Check Plots-Visuals/Visuals.")
