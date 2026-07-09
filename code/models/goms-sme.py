import numpy as np
import torch
import torch.nn as nn
from egnn_pytorch import EGNN as EGNNLayer
from torch_geometric.utils import to_dense_batch

class FragEGNN(nn.Module):
	def __init__(self, feat_dim, num_layers = 3):
		super().__init__()
		
		self.feat_dim = feat_dim
		self.num_layers = num_layers

		#change this is we want the model to have a different architecture
		#right now it's just num_layers of layers of size feat_dim
		self.layers = nn.ModuleList([
			EGNNLayer (dim = feat_dim) for _ in range(num_layers)
		])

	def forward(self, x, pos, batch):
		#needs some preprocessing to seperate pos from x?

		feats, mask = to_dense_batch(x, batch)
		coors, _ = to_dense_batch(pos, batch)

		for layer in self.layers:
			feats, coors = layer(feats, coors, mask=mask) #msg passing

		mask_f = mask.unsqueeze(-1).float()
		frag_vectors = (feats * mask_f).sum(dim = 1) / mask_f.sum(dim = 1).clamp(min=1)

		return frag_vectors