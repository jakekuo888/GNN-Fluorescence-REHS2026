import torch
import numpy as np
import pandas as pd
import sys
import subprocess
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

import sys
import os

from models.neural_networks import ModelTwo
from models.early_stop import EarlyStop

from setup.process_data import absorption_data_options, PredOption, generate_graphs_labels

# from train_test import node_features, edge_features, train_solv_features, ext_loader

re_generate_data = False

molecules_list, y_mean, y_std, train_smiles_for_similarity, train_solv_features = generate_graphs_labels(absorption_data_options[0], generate_data=re_generate_data)
test_molecules_list, test_y_mean, test_y_std, test_smiles_for_similarity, test_solv_features = generate_graphs_labels(absorption_data_options[1], generate_data=re_generate_data, y_mean=y_mean, y_std=y_std, normalize=False)

ext_loader = DataLoader(test_molecules_list, batch_size=256, shuffle=True)

node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_models():
  models = []

  for idx in range(9):
    model = ModelTwo(node_features, edge_features, 128, train_solv_features, [128, 128, 128], [128, 128, 128]).to(device)
    model.load_state_dict(torch.load(f"./models/model_{idx}_weights.pth"))
    models.append(model)

  return models

def get_model_predictions(models, loader):
  all_predictions = []
  all_smiles = []
  
  for model in models:
    model.eval()

  with torch.no_grad():
    for data in loader:

      data.to(device)

      for model in models:
        sol_fp = torch.tensor(np.array(data.sol_fp), dtype=torch.float).to(device)

        preds_tensor = torch.stack([model(data.x, data.edge_index, data.edge_attr, data.batch, sol_fp)[1] for model in models])

        preds_tensor = preds_tensor.squeeze(-1)
        preds_numpy = preds_tensor.transpose(0, 1).cpu().numpy()
        all_predictions.append(preds_numpy)
        all_smiles.extend(data.smiles)

  master_preds_array = np.vstack(all_predictions)

  # Create named columns for the 9 models
  columns = [f"model_{i+1}_out" for i in range(len(models))]
  df = pd.DataFrame(master_preds_array, columns=columns)

  # Insert the SMILES strings at the front
  df.insert(0, "SMILES", all_smiles)

  return df

def get_uncertainties(df, num_uncertain):
  row_std = df.iloc[:, 1:].std(axis=1, numeric_only=True)
  df['row_std'] = row_std
  most_uncertain = df.nlargest(num_uncertain, 'row_std')['SMILES']

  return most_uncertain

models = get_models()
df = get_model_predictions(models, ext_loader)
most_uncertain = get_uncertainties(df, 50)

count = 1
print("MOST UNCERTAIN SMILES:\n-------------------")
for uncertain in most_uncertain:
  print(f"{count}. {uncertain}")
  count += 1