import numpy as np
import torch
import torch.nn as nn
from egnn_pytorch import EGNN as EGNNLayer
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import to_dense_adj

class FragEGNN(nn.Module):
	def __init__(self, node_features, edge_features, num_layers = 3):
		super().__init__()
		
		self.node_features = node_features
		self.edge_features = edge_features
		#Note these are the dimensions and not the features themselves

		self.num_layers = num_layers

		self.layers = nn.ModuleList([
			EGNNLayer (dim = node_features, edge_dim=edge_features) for _ in range(num_layers)
		])

	def forward(self, x, pos, edge_index, edge_attr):
		#needs some preprocessing to seperate pos from x?
        feats, mask = to_dense_batch(x, batch)
        coors, _ = to_dense_batch(pos, batch)

        edges = to_dense_adj(
            edge_index,
            batch=batch,
            edge_attr=edge_attr
        )

        for layer in self.layers:
            feats, coors = layer(feats, coors, edges, mask=mask)

        mask_f = mask.unsqueeze(-1).float()
        frag_vectors = (feats * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return frag_vectors