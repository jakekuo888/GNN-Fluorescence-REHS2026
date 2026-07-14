from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
import numpy as np
import torch
import torch.nn as nn
from egnn_pytorch import EGNN as EGNNLayer


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


class Model(nn.Module):
    def __init__(self, node_features, edge_features, num_layers=3):
        super().__init__()

        self.egnn = FragEGNN(node_features, edge_features, num_layers)

    def generate_GOMS(dicts, frags):
        GOMS = []

        for dict_, frag_ in zip(dicts, frags):
            n_frags = len(frag_)
            x_ = torch.tensor(frag_, dtype = torch.float)

            """
            entr_graph = dict_['entire_graph']
            fta = dict_['frag_to_atom']
            cut_bonds = dict_['cut_bonds']
            """
            
            #waiting for dict to be updated before I can proceed any further

            graph = Data(
                x = x_,
                edge_index = e_idx,
                edge_attr = edge_attr,
                pos = pos,
                y = y,
            )

            GOMS.append(graph)

        return GOMS

    def forward(self, x, pos, edge_index, edge_attr, batch, frags_per_mol, mol_dicts):
        frag_vecs = self.egnn(x, pos, edge_index, edge_attr, batch)

        per_mol_fragment_vectors = []
        offset = 0
        for k in frags_per_mol:
            per_mol_fragment_vectors.append(frag_vecs[offset: offset + k])
            offset += k