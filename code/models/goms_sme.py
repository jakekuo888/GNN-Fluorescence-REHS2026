from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, BatchNorm
from egnn_pytorch import EGNN as EGNNLayer
from torch_geometric.data import Data, Batch
from neural_networks import FFNN


class FragEGNN(nn.Module):
    def __init__(self, node_features, edge_features, num_layers=3):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        # Note these are the dimensions and not the features themselves

        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            EGNNLayer(dim=node_features, edge_dim=edge_features) for _ in range(num_layers)
        ])

    def forward(self, x, pos, edge_index, edge_attr, batch):
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
        frag_vectors = global_mean_pool(feats[mask], batch)

        return frag_vectors


class GAT(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, n_layers, heads=4, dropout=0.3):
        super().__init__()

        self.in_dim = node_features
        self.edge_dim = edge_features
        self.hidden_channels = hidden_channels
        self.num_layers = n_layers
        self.heads = heads
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(in_channels=self.in_dim, out_channels=self.hidden_channels,
                           edge_dim=self.edge_dim, heads=self.heads, dropout=self.dropout))
        self.layers.append(BatchNorm(self.hidden_channels))

        for _ in range(n_layers-1):
            self.layers.append(TransformerConv(in_channels=self.hidden_channels, out_channels=self.hidden_channels,
                               edge_dim=self.edge_dim, heads=self.heads, dropout=self.dropout))
            self.layers.append(BatchNorm(self.hidden_channels))

    def forward(self, x, edge_index, edge_attr, batch):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            if isinstance(layer, BatchNorm):
                x = torch.relu(x)

        graph_readout = global_mean_pool(x, batch)

        return graph_readout


class Model(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, solv_features, num_layers=3):
        super().__init__()

        self.egnn = FragEGNN(node_features, edge_features, num_layers)
        self.gat = GAT(node_features, edge_features,
                       hidden_channels, num_layers)
        self.sol_ffnn = FFNN(solv_features, hidden_channels,
                             hidden_sizes=[64, 64, 64])
        self.ffnn = FFNN(hidden_channels, 1, [64, 64, 64])

    def forward(self, x, pos, edge_index, edge_attr, batch, frags_per_mol, mol_dicts, solv_morgan):
        frag_vecs = self.egnn(x, pos, edge_index, edge_attr, batch)

        per_mol_fragment_vectors = []
        offset = 0
        for k in frags_per_mol:
            per_mol_fragment_vectors.append(frag_vecs[offset: offset + k])
            offset += k

        goms = []
        for frag_x, d in zip(per_mol_fragment_vectors, mol_dicts):
            data = Data(
                x=frag_x, edge_index=d["gs_edge_index"], edge_attr=d["gs_edge_attr"])
            goms.append(data)

        gs_batch = Batch.from_data_list(goms)
        mol_readout = self.gat(
            gs_batch.x, gs_batch.edge_index, gs_batch.edge_attr, gs_batch.batch)  # type: ignore
        solv_readout = self.sol_ffnn(solv_morgan)

        final_readout = torch.cat([mol_readout, solv_readout], dim=-1)

        final_out = self.ffnn(final_readout)

        return final_out
