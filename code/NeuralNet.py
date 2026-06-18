import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool

import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix


class FFNN(nn.Module):
  def __init__(self, in_size, out_size, hidden_sizes=None):
    super().__init__()

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
      layers.append(nn.ReLU())
      p_size = h
    layers.append(nn.Linear(p_size, self.outS))
    self.net = nn.Sequential(*layers)

  def forward(self, inp):
    return self.net(inp)

class GNN(nn.Module):
  def __init__(self, node_features, edge_features, hidden_channels):
    super(GNN, self).__init__()
    torch.manual_seed(12345)

    # vector addition requires same size between the edge and node vectors
    self.edge_encoder = nn.Linear(edge_features, hidden_channels) # map edge vector to size of hidden layers
    self.node_encoder = nn.Linear(node_features, hidden_channels) # map node vector to size of hidden layers

    gine_mlp = nn.Sequential(
      nn.Linear(hidden_channels, hidden_channels),
      nn.ReLU(),
      nn.Linear(hidden_channels, hidden_channels)
    )

    self.conv1 = GINEConv(gine_mlp, eps=0.0, train_eps=True, edge_dim=edge_features)

  def forward(self, x, edge_index, edge_attr, batch):
    x = self.node_encoder(x)
    edge_attr = self.edge_encoder(edge_attr)

    x = self.conv1(x, edge_index, edge_attr)
    x = torch.relu(x)

    graph_readout = global_add_pool(x, batch)
    return graph_readout

class Model(nn.Module):
  def __init__(self, node_features, edge_features, hidden_channels, hidden_sizes):
    self.gnn = GNN(node_features, edge_features, hidden_channels) 
    self.ffnn = FFNN(hidden_channels, out_size=1, hidden_sizes)

  def forward(self, x, edge_index, edge_attr, batch):
    self.gnn.train()
    readout_vector = self.gnn(x, edge_index, edge_attr, batch)

    self.ffnn.train()
    fluorescence_time = self.ffnn(readout_vector)

    return fluorescence_time

molecules_list = torch.load('./data-wrangling/lifetime-data/molecularGraphs.pt', weights_only=False) # list of PyG Data objects
loader = DataLoader(molecules_list, batch_size=32, shuffle=True) # mini-batching
  
node_features = molecules_list[0].num_node_features
edge_features = molecules_list[0].num_edge_features

with open('./data-wrangling/lifetime-data/lifetime.txt', 'r') as f:
  fluorescence_times = f.read().splitlines()
  
model = GNN(node_features, edge_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)