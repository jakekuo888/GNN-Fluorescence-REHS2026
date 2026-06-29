import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, BatchNorm

import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Feed forward neural network for post-GNN processing (returns a scalar = fluorescence time)
class FFNN(nn.Module):
  def __init__(self, in_size, out_size, hidden_sizes=None):
    super().__init__()

    # Easter egg / failsafe
    if hidden_sizes == None:
      hidden_sizes = [67, 67]
    
    self.inS = in_size
    self.outS = out_size
    self.hidS = hidden_sizes
    layers = []
    p_size = self.inS

    #set layers
    for h in self.hidS:
      layers.append(nn.Linear(p_size, h))
      layers.append(BatchNorm(h)) # add batch norm to FFNN
      layers.append(nn.ReLU())
      p_size = h
    layers.append(nn.Linear(p_size, self.outS))
    self.net = nn.Sequential(*layers)

  def forward(self, inp):
    return self.net(inp)

# Actual GNN model (takes in one graph)
class GNN(nn.Module):
  def __init__(self, node_features, edge_features, hidden_channels):
    super(GNN, self).__init__()

    # edge and node vectors should be the same size, project edge feature & node feature sizes to hidden channels to facilitate matrix addition
    self.edge_encoder = nn.Linear(edge_features, hidden_channels)
    self.node_encoder = nn.Linear(node_features, hidden_channels)

    # Define MLPs for the GINE layers (one for each of the three layers)
    gine_mlp1 = nn.Sequential(
      nn.Linear(hidden_channels, hidden_channels),
      nn.ReLU(),
      nn.Linear(hidden_channels, hidden_channels)
    )

    gine_mlp2 = nn.Sequential(
      nn.Linear(hidden_channels, hidden_channels),
      nn.ReLU(),
      nn.Linear(hidden_channels, hidden_channels)
    )

    gine_mlp3 = nn.Sequential(
      nn.Linear(hidden_channels, hidden_channels),
      nn.ReLU(),
      nn.Linear(hidden_channels, hidden_channels)
    )

    # Define layers with eps training as well
    self.conv1 = GINEConv(gine_mlp1, eps=0.0, train_eps=True, edge_dim=hidden_channels)
    self.conv2 = GINEConv(gine_mlp2, eps=0.0, train_eps=True, edge_dim=hidden_channels)
    self.conv3 = GINEConv(gine_mlp3, eps=0.0, train_eps=True, edge_dim=hidden_channels)

    # Batch Norm to Regularize Training & Combat Overfitting
    self.bn1 = BatchNorm(hidden_channels)
    self.bn2 = BatchNorm(hidden_channels)
    self.bn3 = BatchNorm(hidden_channels)

  # Forward function with ReLU activation
  def forward(self, x, edge_index, edge_attr, batch):
    if self.training:
        edge_index, edge_mask = dropout_edge(edge_index, p=0.3)
        edge_attr = edge_attr[edge_mask]
    
    x = self.node_encoder(x)
    edge_attr = self.edge_encoder(edge_attr)

    x = self.conv1(x, edge_index, edge_attr)
    x = self.bn1(x)
    x = torch.relu(x)

    x = self.conv2(x, edge_index, edge_attr)
    x = self.bn2(x)
    x = torch.relu(x)

    # x = self.conv3(x, edge_index, edge_attr)
    # x = self.bn3(x)
    # x = torch.relu(x)

    # Readout function is global_add_pool
    graph_readout = global_add_pool(x, batch)
  
    return graph_readout

# Combines two GNNs (molecule & solvent) and passes the concatenated readout through an FFNN
class Model(nn.Module):
  def __init__(self, node_features, edge_features, hidden_channels, hidden_sizes):
    super().__init__()

    self.gnn_mol = GNN(node_features, edge_features, hidden_channels) 
    self.gnn_sol = GNN(node_features, edge_features, hidden_channels)

    self.cross_attn1 = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
    self.cross_attn2 = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)

    self.ffnn = FFNN(2*hidden_channels, 1, hidden_sizes)

  def forward(self, xm, mol_edge_index, mol_edge_attr, mol_batch, xs, sol_edge_index, sol_edge_attr, sol_batch):

    readout_vector_mol = self.gnn_mol(xm, mol_edge_index, mol_edge_attr, mol_batch)
    readout_vector_sol = self.gnn_sol(xs, sol_edge_index, sol_edge_attr, sol_batch)

    mol_q = readout_vector_mol.unsqueeze(1)
    sol_q = readout_vector_sol.unsqueeze(1)

    # mol attends to sol
    mol_attn, _ = self.cross_attn1(mol_q, sol_q, sol_q)
    mol_out = readout_vector_mol + mol_attn.squeeze(1)

    # sol attends to mol
    sol_attn, _ = self.cross_attn2(sol_q, mol_q, mol_q)
    sol_out = readout_vector_sol + sol_attn.squeeze(1)

    final_readout = torch.cat([mol_out, sol_out], dim=-1)
    
    fluorescence_time = self.ffnn(final_readout)

    return readout_vector_mol, fluorescence_time


class Model_two(nn.Module):
  def __init__(self, node_features, edge_features, hidden_channels, solv_features, hidden_sizes=None, solv_hidden_sizes=None):
    super().__init__()

    self.gnn_mol = GNN(node_features, edge_features, hidden_channels) 
    #solv_features is the length of the morgan fingerprint
    self.ffnn_solv = FFNN(solv_features, hidden_channels, solv_hidden_sizes)

    self.cross_attn1 = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
    self.cross_attn2 = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)

    self.ffnn = FFNN(2*hidden_channels, 1, hidden_sizes)

  def forward(self, xm, mol_edge_index, mol_edge_attr, mol_batch, solv_morgan):
    readout_vector_mol = self.gnn_mol(xm, mol_edge_index, mol_edge_attr, mol_batch)
    readout_vector_sol = self.ffnn_solv(solv_morgan)

    mol_q = readout_vector_mol.unsqueeze(1)
    sol_q = readout_vector_sol.unsqueeze(1)

    # mol attends to sol
    mol_attn, _ = self.cross_attn1(mol_q, sol_q, sol_q)
    mol_out = readout_vector_mol + mol_attn.squeeze(1)

    # sol attends to mol
    sol_attn, _ = self.cross_attn2(sol_q, mol_q, mol_q)
    sol_out = readout_vector_sol + sol_attn.squeeze(1)

    final_readout = torch.cat([mol_out, sol_out], dim=-1)
    
    fluorescence_time = self.ffnn(final_readout)

    return readout_vector_mol, fluorescence_time