import torch
import numpy as np
import sys
import subprocess
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold

import sys
import os

from neural_networks import ModelTwo
from early_stop import EarlyStop

from process_data import absorption_data_options, PredOption, generate_graphs_labels

#EASY CONTROLS vvv
n_epochs = 100
collect_data = True
early_stopper = EarlyStop(9, 0.005)
#EASY CONTROLS ^^^

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(root_dir, 'data-wrangling'))
sys.path.append(os.path.join(root_dir, 'plots-visuals'))

from plot_similarity_error import plot_vector_similarity_loss_graph, plot_smiles_similarity_loss_graph

re_generate_data = False

molecules_list, y_mean, y_std, train_smiles_for_similarity, train_solv_features = generate_graphs_labels(absorption_data_options[0], generate_data=re_generate_data)
test_molecules_list, test_y_mean, test_y_std, test_smiles_for_similarity, test_solv_features = generate_graphs_labels(absorption_data_options[1], generate_data=re_generate_data, y_mean=y_mean, y_std=y_std, normalize=False)

ext_loader = DataLoader(test_molecules_list, batch_size=256, shuffle=True)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
folds = list(kf.split(molecules_list))

train_loaders = []
test_loaders = []
validate_loaders =[]

for i in range(9):
  test_fold_idx = i
  val_fold_idx = (i + 1) % 10
  train_folds_indices = [j for j in range(10) if j != test_fold_idx and j != val_fold_idx]

  test_indices = folds[test_fold_idx][1]
  val_indices = folds[val_fold_idx][1]
  train_indices = np.concatenate([folds[j][1] for j in train_folds_indices])

  train_dataset = [molecules_list[idx] for idx in train_indices]
  validate_dataset = [molecules_list[idx] for idx in val_indices]
  test_dataset = [molecules_list[idx] for idx in test_indices]

  train_loaders.append(DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True))
  validate_loaders.append(DataLoader(validate_dataset, batch_size=128, shuffle=True))
  test_loaders.append(DataLoader(test_dataset, batch_size=128, shuffle=True))

# Set up the Model class (GNN/FFNN), AdamW optimizer, and MAE Loss function
node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = []
optimizers = []
schedulers = []
stoppers = []

for i in range(9):
  model = ModelTwo(node_features, edge_features, 128, train_solv_features, [128, 128, 128], [128, 128, 128]).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
  
  #appending to lists
  models.append([model, ""])
  optimizers.append(optimizer)
  schedulers.append(scheduler)
  stoppers.append(EarlyStop(9, 0.005))

criterion = torch.nn.L1Loss()

# Train takes in mol & sol loader, zips them to return a forward pass through the model, loss, backprop, repeat
def train(model, opt, loader):
  model.train()

  for data in loader:
    data.to(device)

    sol_fp = torch.tensor(np.array(data.sol_fp), dtype=torch.float).to(device)
    vector_out, out = model(
      data.x, data.edge_index, data.edge_attr, data.batch, sol_fp
    )
    loss = criterion(out, data.y.view(-1,1))
    opt.zero_grad()
    loss.backward()
    opt.step()

train_vectors_for_similarity = []
test_vectors_for_similarity = []
test_losses_for_similarity = []

# Train has the evaluation mode (output actual vs predicted for sample) and the non-evaluation mode (just avg MAE output)
def test(model, loader, mean, std, compute_mae=True, is_test_set=False, collect_plot_data=False):
  model.eval()

  total_mae = 0.0
  total_graphs = 0

  with torch.no_grad():

    for data in loader:
      data.to(device)

      sol_fp = torch.tensor(np.array(data.sol_fp), dtype=torch.float).to(device)
      vector_out, out = model(
        data.x, data.edge_index, data.edge_attr, data.batch, sol_fp
      )
      
      # Reverse normalization for prediction
      pred_log = out.squeeze() * std + mean
      pred_actual = torch.exp(pred_log)
      pred_actual = pred_actual.flatten()

      # Reverse normalization for target too
      target_log = data.y.flatten() * std + mean
      target_actual = torch.exp(target_log)

      loss = torch.mean(torch.abs(pred_actual - target_actual))

      if collect_plot_data and not is_test_set:
          train_vectors_for_similarity.extend(vector_out.unbind(0))
      elif collect_plot_data:
          test_vectors_for_similarity.extend(vector_out.unbind(0))
          test_losses_for_similarity.extend([torch.abs(p - t).item() 
                                              for p, t in zip(pred_actual, target_actual)])

      if compute_mae:
        num_graphs = data.num_graphs
        total_mae += loss * num_graphs
        total_graphs += num_graphs
  
  if compute_mae:
    avg_mae = total_mae / total_graphs

    return avg_mae
  else:
    return 0.0

def run_model(model, train_loader, val_loader, opt, sched, idx):
  # Train & Test the Model
  with open(f"./data/plot-data/loss-model-{idx}.txt", "w") as f_:
    for epoch in range(1,n_epochs+1):
      train(model, opt, train_loader)

      train_avg_mae = test(model, train_loader, y_mean, y_std)
      sample_avg_mae = test(model, val_loader, y_mean, y_std)
      sched.step(float(sample_avg_mae))

      if stoppers[idx].stop_early(sample_avg_mae, model):
        print(f'Early stop has been initiated on Epoch #{epoch}')
        stoppers[idx].restore_best(model)
        break

      print(f"Epoch #{epoch} | Train Average MAE: {train_avg_mae:.4f} | Test Average MAE: {sample_avg_mae:.4f} | Early stopper count: {stoppers[idx].count}")
      if(collect_data): print(f"{train_avg_mae:.4f}, {sample_avg_mae:.4f}", file=f_) #loading data for plotting (train, test)

def test_model(model, test_loader, idx):
  test_avg_mae = test(model, test_loader, y_mean, y_std, compute_mae=True)
  print(f"TEST AVERAGE MAE FOR MODEL #{idx} (FINAL RESULTS): {test_avg_mae}")

for idx in range(len(models)):
  print(f"\n---------------------------------------- MODEL #{idx} RUNNING NOW ----------------------------------------")
  run_model(models[idx][0], train_loaders[idx], validate_loaders[idx], optimizers[idx], schedulers[idx], idx)

print("UNDERGOING TESTING")
print("-" * 45)
for idx in range(len(models)):
  test_model(models[idx][0], test_loaders[idx], idx)
  torch.save(models[idx][0].state_dict(), f"./models/model_{idx}_weights.pth")
  models[idx].append(f"./models/model_{idx}_weights.pth")

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

print("PROCESS DONE")