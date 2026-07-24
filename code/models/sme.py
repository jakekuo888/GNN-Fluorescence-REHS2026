import torch
import random
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import LayerNorm

def sme_pool(x, batch, num_mols, mask = None):
	#Make sure to skip molecules with only one fragment total
    if mask is None:
        #Rm none if mask = None
        mask = torch.ones(x.size(0), 1, device = x.device)
    else:
        mask = mask.view(-1, 1).float()

    summed = torch.zeros(num_mols, x.size(1), device = x.device)
    summed.index_add_(0, batch, x * mask)
    counts = torch.bincount(batch, minlength=num_mols).clamp(min=1).unsqueeze(-1).float()

    return summed / counts


@torch.no_grad()
def sme_attribution(model, data, frags_per_mol, mol_dicts, sol_fp, device, max_combinations=100):
    model.eval()

    # ---- 1. Atom -> fragment embeddings (unmasked, run once) ----
    frag_vecs = model.egnn(data.x, data.pos, data.edge_index, data.edge_attr, data.batch)

    per_mol_fragment_vectors = []
    offset = 0
    for k in frags_per_mol:
        per_mol_fragment_vectors.append(frag_vecs[offset: offset + k])
        offset += k

    goms = [
        Data(x=frag_x, edge_index=d["edge_index"], edge_attr=d["edge_attr"])
        for frag_x, d in zip(per_mol_fragment_vectors, mol_dicts)
    ]
    gs_batch = Batch.from_data_list(goms).to(device)

    # ---- 2. Fragment-level message passing (unmasked, run once) ----
    # NOTE: this mirrors GAT.forward's layer loop by hand so we can
    # intercept before pooling. If GAT.forward changes (new norm type,
    # new activation, new dropout, etc.), this loop must be updated to match.
    x = gs_batch.x
    for layer in model.gat.layers:
        if isinstance(layer, LayerNorm):
            x = layer(x)
            x = torch.relu(x)
            x = model.gat.feature_drop(x)
        else:
            x = layer(x, gs_batch.edge_index, gs_batch.edge_attr)
    # x == h_v for every fragment, frozen for all subsequent masks

    solv_readout = model.sol_ffnn(sol_fp)  # unaffected by fragment masking

    num_mols = len(frags_per_mol)

    # ---- 3. Full (unmasked) prediction, per molecule ----
    mol_readout_full = sme_pool(x, gs_batch.batch, num_mols)
    Y_full = model.ffnn(torch.cat([mol_readout_full, solv_readout], dim=-1))  # [num_mols, 1]

    # ---- 4. Per-fragment (and optionally combination) attributions ----
    results = []
    frag_offset = 0
    for mol_idx, k in enumerate(frags_per_mol):
        global_frag_ids = list(range(frag_offset, frag_offset + k))
        attributions = {}

        for local_i, gid in enumerate(global_frag_ids):
            mask = torch.ones(x.size(0), device=device)
            mask[gid] = 0.0

            mol_readout_masked = sme_pool(x, gs_batch.batch, num_mols, mask)
            Y_masked = model.ffnn(torch.cat([mol_readout_masked, solv_readout], dim=-1))

            attributions[local_i] = (Y_full[mol_idx] - Y_masked[mol_idx]).item()

        results.append(attributions)
        frag_offset += k

    return results