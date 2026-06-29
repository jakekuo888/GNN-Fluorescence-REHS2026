import torch
import sys
import subprocess
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

import sys
import os

from neural_networks import Model
from early_stop import EarlyStop

from process_data import train_molecules_list as molecules_list
from process_data import train_solvents_list as solvents_list
from process_data import train_y_mean as y_mean
from process_data import train_y_std as y_std

from process_data import test_molecules_list, test_solvents_list, test_y_mean, test_y_std, train_smiles_for_similarity, test_smiles_for_similarity

from process_data import absorption_data, PredOption, generate_graphs_labels

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(root_dir, 'data-wrangling'))
sys.path.append(os.path.join(root_dir, 'plots-visuals'))

from plot_similarity_error import plot_vector_similarity_loss_graph, plot_smiles_similarity_loss_graph
from data_conversion import generate_and_export_data

re_generate_data = False

if re_generate_data:
  chosen_option = absorption_data[0]
  generate_and_export_data(chosen_option.dataset, chosen_option.mol_label, chosen_option.sol_label, chosen_option.pred_label, chosen_option.out_folder, chosen_option.out_file)
  
  chosen_option = absorption_data[1]
  generate_and_export_data(chosen_option.dataset, chosen_option.mol_label, chosen_option.sol_label, chosen_option.pred_label, chosen_option.out_folder, chosen_option.out_file)
  
molecules_list, solvents_list, y_mean, y_std, train_smiles_for_similarity = generate_graphs_labels(absorption_data[0])
test_molecules_list, test_solvents_list, test_y_mean, test_y_std, test_smiles_for_similarity = generate_graphs_labels(absorption_data[1], y_mean=y_mean, y_std=y_std, normalize=False)

#EASY CONTROLS vvv
n_epochs = 100
collect_data = True
early_stopper = EarlyStop(9, 0.005)
#EASY CONTROLS ^^^

ext_mol_loader = DataLoader(test_molecules_list, batch_size=256, shuffle=False)
ext_sol_loader = DataLoader(test_solvents_list, batch_size=256, shuffle=False)

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

# Set up the Model class (GNN/FFNN), AdamW optimizer, and MAE Loss function
node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(node_features, edge_features, 128, [128, 128, 128]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.L1Loss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

total_weights = 0
total_biases = 0

for name, param in model.named_parameters():
    if 'weight' in name:
        total_weights += param.numel()
    elif 'bias' in name:
        total_biases += param.numel()

# Train takes in mol & sol loader, zips them to return a forward pass through the model, loss, backprop, repeat
def train(mol_loader, sol_loader):
  model.train()

  for mol_data, sol_data in zip(mol_loader, sol_loader):
    mol_data.to(device)
    sol_data.to(device)

    vector_out, out = model(
      mol_data.x, mol_data.edge_index, mol_data.edge_attr, mol_data.batch,
      sol_data.x, sol_data.edge_index, sol_data.edge_attr, sol_data.batch
    )
    loss = criterion(out, mol_data.y.view(-1,1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

train_vectors_for_similarity = []
test_vectors_for_similarity = []
test_losses_for_similarity = []

# Train has the evaluation mode (output actual vs predicted for sample) and the non-evaluation mode (just avg MAE output)
def test(mol_loader, sol_loader, mean, std, compute_mae=True, is_test_set=False, collect_plot_data=False):
  model.eval()

  total_mae = 0.0
  total_graphs = 0

  with torch.no_grad():

    for mol_data, sol_data in zip(mol_loader, sol_loader):
      mol_data.to(device)
      sol_data.to(device)

      vector_out, out = model(
        mol_data.x, mol_data.edge_index, mol_data.edge_attr, mol_data.batch,
        sol_data.x, sol_data.edge_index, sol_data.edge_attr, sol_data.batch
      )
      
      # Reverse normalization for prediction
      pred_log = out.squeeze() * std + mean
      pred_actual = torch.exp(pred_log)
      pred_actual = pred_actual.flatten()

      # Reverse normalization for target too
      target_log = mol_data.y.flatten() * std + mean
      target_actual = torch.exp(target_log)

      loss = torch.mean(torch.abs(pred_actual - target_actual))

      if collect_plot_data and not is_test_set:
          train_vectors_for_similarity.extend(vector_out.unbind(0))
      elif collect_plot_data:
          test_vectors_for_similarity.extend(vector_out.unbind(0))
          test_losses_for_similarity.extend([torch.abs(p - t).item() 
                                              for p, t in zip(pred_actual, target_actual)])

      if compute_mae:
        num_graphs = mol_data.num_graphs
        total_mae += loss * num_graphs
        total_graphs += num_graphs
  
  if compute_mae:
    avg_mae = total_mae / total_graphs

    return avg_mae
  else:
    return 0.0

# Train & Test the Model
with open("./data/plot-data/loss.txt", "w") as f_:
  for epoch in range(1,n_epochs+1):
    train(mol_train_loader, sol_train_loader)

    train_avg_mae = test(mol_train_loader, sol_train_loader, y_mean, y_std)
    sample_avg_mae = test(mol_validate_loader, sol_validate_loader, y_mean, y_std)
    scheduler.step(float(sample_avg_mae))

    if early_stopper.stop_early(sample_avg_mae, model):
      print(f'Early stop has been initiated on Epoch #{epoch}')
      early_stopper.restore_best(model)
      break

    print(f"Epoch #{epoch} | Train Average MAE: {train_avg_mae:.4f} | Test Average MAE: {sample_avg_mae:.4f} | Early stopper count: {early_stopper.count}")
    if(collect_data): print(f"{train_avg_mae:.4f}, {sample_avg_mae:.4f}", file=f_) #loading data for plotting (train, test)

print("UNDERGOING TESTING")
print("-" * 45)
test_avg_mae = test(mol_test_loader, sol_test_loader, y_mean, y_std, compute_mae=True)
print(f"TEST AVERAGE MAE (FINAL RESULTS): {test_avg_mae}")
print("-" * 45)

print("TESTING ON EXTERNAL QMWF DATASET")
external_avg_mae = test(ext_mol_loader, ext_sol_loader, test_y_mean, test_y_std, compute_mae=True, is_test_set=True, collect_plot_data=True)
print("-" * 45)
print(f"EXTERNAL AVERAGE MAE (FINAL RESULTS): {external_avg_mae}")
print("-" * 45)

train_avg_mse = test(mol_train_dataset, sol_train_dataset, y_mean, y_std, compute_mae=False, collect_plot_data=True)

if(collect_data):
  want_visuals = input("\n Do you want to create Visuals (Y/N): ").lower()
  if(want_visuals == 'y'):

    print("Creating plotting loss visuals \n ...")
    subprocess.run([sys.executable, "./plots-visuals/plot-loss.py"])
    print("Plotting loss sucessfully created!\n Check plots-visuals/new-plots.")
    
    print("Creating scatterplot of the error vs similarity (vectors) \n ...")
    plot_vector_similarity_loss_graph(train_vectors_for_similarity, test_vectors_for_similarity, test_losses_for_similarity)
    print("Scatterplot successfully created! \n Check plots-visuals/new-plots")

    print("Creating scatterplot of the error vs similarity (smiles) \n ...")
    plot_smiles_similarity_loss_graph(train_smiles_for_similarity, test_smiles_for_similarity, test_losses_for_similarity)
    
    print("DONE")