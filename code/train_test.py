from plotting.plot_similarity_error import plot_vector_similarity_loss_graph, plot_smiles_similarity_loss_graph
import torch
from torch.utils.data import DataLoader as TorchLoader
import numpy as np
import sys
import subprocess
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold

import sys
import os

from models.neural_networks import ModelTwo
from models.goms_sme import Model
from models.early_stop import EarlyStop
from models.goms_sme import FragEGNN

from setup.process_data import absorption_data_options, generate_graphs_labels, FragmentDataset, collate_fn
from models.sme import sme_attribution

import json
from rdkit import Chem

# EASY CONTROLS vvv
n_epochs = 10
collect_data = True
early_stopper = EarlyStop(9, 0.005)
re_generate_data = False
# EASY CONTROLS ^^^

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(root_dir, 'data-wrangling'))
sys.path.append(os.path.join(root_dir, 'plots-visuals'))

if __name__ == "__main__":

    # D4C
    molecules_dicts, y_mean, y_std, train_smiles_for_similarity, train_solv_features = generate_graphs_labels(
        absorption_data_options[0], generate_data=re_generate_data)

    # External Set
    ext_dataset, test_y_mean, test_y_std, test_smiles_for_similarity, test_solv_features = generate_graphs_labels(
        absorption_data_options[1], generate_data=re_generate_data, y_mean=y_mean, y_std=y_std, normalize=False)

    if re_generate_data:
        print("DATA FINISHED GENERATING")

    def has_reasonable_geometry(mol_dict, max_coord=200.0):
        for frag in mol_dict["frag_graphs"]:
            if frag.pos.abs().max().item() > max_coord:
                return False
        return True

    before = len(molecules_dicts)
    molecules_dicts = [
        d for d in molecules_dicts if has_reasonable_geometry(d)]
    print(
        f"Filtered {before - len(molecules_dicts)} molecules with corrupted geometry")

    y_vals = torch.tensor([d["y_normalized"].item() for d in molecules_dicts])
    baseline_mae_normalized = y_vals.abs().mean()
    print(
        f"Baseline (predict mean) normalized MAE: {baseline_mae_normalized:.4f}")

    y_real_vals = torch.tensor([d["y_real"].item() for d in molecules_dicts])
    baseline_mae_real = (y_real_vals - y_real_vals.mean()).abs().mean()
    print(f"Baseline (predict mean) real-units MAE: {baseline_mae_real:.4f}")

    # Splitting Datasets Randomly (still a list of dictionaries)
    train_dataset, split_dataset = train_test_split(
        molecules_dicts, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(
        split_dataset, test_size=0.5, random_state=42)

    # Loaders (returns the list of all fragments, maps, and dicts for later use)
    train_loader = TorchLoader(FragmentDataset(
        train_dataset), batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = TorchLoader(FragmentDataset(
        val_dataset), batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = TorchLoader(FragmentDataset(
        test_dataset), batch_size=128, shuffle=True, collate_fn=collate_fn)
    ext_loader = TorchLoader(FragmentDataset(
        ext_dataset), batch_size=256, shuffle=True, collate_fn=collate_fn)

    # Set up the Model class (GNN/FFNN), AdamW optimizer, and MAE Loss function
    node_features = molecules_dicts[0]["frag_graphs"][0].num_node_features
    edge_features = molecules_dicts[0]["frag_graphs"][0].num_edge_features
    gs_edge_features = molecules_dicts[0]["edge_attr"].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(node_features, edge_features, 64,
                  gs_edge_features, train_solv_features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    stopper = EarlyStop(9, 0.005)

    criterion = torch.nn.L1Loss()

    # Train takes in mol & sol loader, zips them to return a forward pass through the model, loss, backprop, repeat

    def train(model, opt, loader):
        model.train()

        skipped, total = 0, 0
        for data, frags_per_mol, mol_dicts in loader:
            data.to(device)

            mol_dicts = [
                {
                    **d,
                    "edge_index": d["edge_index"].to(device),
                    "edge_attr": d["edge_attr"].to(device),
                }
                for d in mol_dicts
            ]

            sol_fps = [d['sol_fp'] for d in mol_dicts]
            sol_fps = np.array(sol_fps)
            sol_fp = torch.tensor(sol_fps, dtype=torch.float, device=device)
            final_readout, out = model(data.x, data.pos, data.edge_index, data.edge_attr,
                                       data.batch, frags_per_mol, mol_dicts, sol_fp)

            y = torch.tensor([d["y_normalized"] for d in mol_dicts],
                             dtype=torch.float).to(device).unsqueeze(-1)

            loss = criterion(out, y)
            total += 1

            if not torch.isfinite(loss) or loss.item() > 1e6:
                offending_smiles = [d["smiles"] for d in mol_dicts]
                print(
                    f"!!! Skipping batch, loss={loss.item()}. Molecules in batch: {offending_smiles}")
                opt.zero_grad()
                skipped += 1
                continue  # skip this batch entirely -- don't let it corrupt the weights

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grads_ok = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters() if p.grad is not None
            )

            if not grads_ok:
                skipped += 1
                print(
                    f"!!! Skipping step, NaN/Inf gradient detected. Molecules: {[d['smiles'] for d in mol_dicts]}")
                opt.zero_grad()
                continue

            opt.step()
        # print(f"  [train] skipped {skipped}/{total} batches")

    train_vectors_for_similarity = []
    test_vectors_for_similarity = []
    test_losses_for_similarity = []
    train_smiles_collected = []
    test_smiles_collected = []

    # Train has the evaluation mode (output actual vs predicted for sample) and the non-evaluation mode (just avg MAE output)

    def test(model, loader, mean, std, compute_mae=True, is_test_set=False, collect_plot_data=False):
        model.eval()

        total_mae = 0.0
        total_graphs = 0

        skipped, total = 0, 0

        with torch.no_grad():

            for data, frags_per_mol, mol_dicts in loader:
                total += 1
                data.to(device)

                mol_dicts = [
                    {
                        **d,
                        "edge_index": d["edge_index"].to(device),
                        "edge_attr": d["edge_attr"].to(device),
                    }
                    for d in mol_dicts
                ]

                sol_fps = [d['sol_fp'] for d in mol_dicts]
                sol_fps = np.array(sol_fps)
                sol_fp = torch.tensor(
                    sol_fps, dtype=torch.float, device=device)
                vector_out, out = model(data.x, data.pos, data.edge_index, data.edge_attr,
                                        data.batch, frags_per_mol, mol_dicts, sol_fp)
                y = torch.tensor([d["y_normalized"] for d in mol_dicts],
                                 dtype=torch.float).to(device).unsqueeze(-1)

                loss = criterion(out, y)

                # Reverse normalization for prediction
                pred_log = out.squeeze(-1) * std + mean
                pred_log = torch.clamp(pred_log, max=12)
                pred_actual = torch.exp(pred_log)
                pred_actual = pred_actual.flatten()

                # Reverse normalization for target too
                target_actual = torch.tensor([d["y_real"] for d in mol_dicts],
                                             dtype=torch.float).to(device).unsqueeze(-1)

                loss = torch.mean(torch.abs(pred_actual - target_actual))

                if not torch.isfinite(loss) or loss.item() > 1e6:
                    skipped += 1
                    offending_smiles = [d["smiles"] for d in mol_dicts]
                    print(
                        f"!!! Skipping eval batch, loss={loss.item()}. Molecules: {offending_smiles}")
                    continue  # don't let this batch's inf poison the epoch average

                if collect_plot_data and not is_test_set:
                    train_vectors_for_similarity.extend(vector_out.unbind(0))
                    train_smiles_collected.extend([d["smiles"] for d in mol_dicts])
                elif collect_plot_data:
                    test_vectors_for_similarity.extend(vector_out.unbind(0))
                    test_losses_for_similarity.extend([torch.abs(p - t).item()
                                                       for p, t in zip(pred_actual, target_actual)])
                    test_smiles_collected.extend([d["smiles"] for d in mol_dicts])

                if compute_mae:
                    num_graphs = data.num_graphs
                    total_mae += loss * num_graphs
                    total_graphs += num_graphs
            # print(f"  [test] skipped {skipped}/{total} batches")
        if compute_mae:
            avg_mae = total_mae / total_graphs

            return avg_mae
        else:
            return 0.0

    def run_model(model, train_loader, val_loader, opt, sched, stopper):
        # Train & Test the Model
        with open(f"./data/plot-data/loss.txt", "w") as f_:
            for epoch in range(1, n_epochs+1):
                train(model, opt, train_loader)

                train_avg_mae = test(model, train_loader, y_mean, y_std)
                val_avg_mae = test(model, val_loader, y_mean, y_std)
                sched.step(float(val_avg_mae))

                if stopper.stop_early(val_avg_mae, model):
                    print(f'Early stop has been initiated on Epoch #{epoch}')
                    stopper.restore_best(model)
                    break

                print(
                    f"Epoch #{epoch} | Train Average MAE: {train_avg_mae:.4f} | Test Average MAE: {val_avg_mae:.4f} | Early stopper count: {stopper.count}")
                if (collect_data):
                    # loading data for plotting (train, test)
                    print(f"{train_avg_mae:.4f}, {val_avg_mae:.4f}", file=f_)

    run_model(model, train_loader, val_loader, optimizer, scheduler, stopper)

    print("UNDERGOING TESTING")
    print("-" * 45)

    # NOTE: collect_plot_data=True populates train_vectors_for_similarity /
    # test_vectors_for_similarity / test_losses_for_similarity, which the
    # similarity scatter plots below need. Previously these were never
    # populated (all test() calls used the collect_plot_data default of
    # False), so the plotting calls crashed with an empty-list np.stack error.
    test(model, train_loader, y_mean, y_std, compute_mae=False,
         is_test_set=False, collect_plot_data=True)

    test_avg_mae = test(model, test_loader, y_mean, y_std, compute_mae=True,
                        is_test_set=True, collect_plot_data=True)
    print(
        f"TEST AVERAGE MAE (FINAL RESULTS): {test_avg_mae}\n-------------------------------")

    test_avg_mae = test(model, ext_loader, y_mean, y_std, compute_mae=True)
    print(
        f"EXTERNAL AVERAGE MAE (FINAL RESULTS): {test_avg_mae}\n-------------------------------")

    print("\n \n COMPUTING SME ATTR. ON TEST SET")
    model.eval()
    all_attr = []
    with torch.no_grad():
        for data, frags_per_mol, mol_dicts in test_loader:
            data.to(device)
            mol_dicts = [
                {
                    **d,
                    "edge_index": d["edge_index"].to(device),
                    "edge_attr": d["edge_attr"].to(device),
                }
                for d in mol_dicts
            ]

            sol_fps = np.array([d['sol_fp'] for d in mol_dicts])
            sol_fp = torch.tensor(sol_fps, dtype=torch.float, device=device)
            batch_attrs = sme_attribution(
                model, data, frags_per_mol, mol_dicts, sol_fp, device, combo_search=True)

            for mol_dict, attrs in zip(mol_dicts, batch_attrs):
                smiles = mol_dict["smiles"]

                heavy_mol = Chem.MolFromSmiles(smiles)
                n_heavy = heavy_mol.GetNumAtoms()

                filtered_atom_groups = [
                    [a for a in group if a < n_heavy]
                    for group in mol_dict["atom_groups"]
                ]

                fragment_removal = {
                    str(k): v for k, v in attrs.items() if k != "combinations"}

                brics_types = {
                    str(frag_id): sorted(labels)
                    for frag_id, labels in mol_dict["frag_brics_types"].items()
                }

                record = {
                    "smiles": smiles,
                    "atom_groups": filtered_atom_groups,
                    "fragment_removal": fragment_removal,
                    "frag_brics_types": brics_types,
                }

                if "combinations" in attrs:
                    record["combinations"] = {
                        str(k): v for k, v in attrs["combinations"].items()}

                all_attr.append(record)

    print("COMPUTING SME DONE; CONTINUING.")


    with open("./data/plot-data/sme.json", "w") as f:
        json.dump(all_attr, f, indent=2)

    # visuals
    if (collect_data):
        want_visuals = input(
            "\n Do you want to create Visuals (Y/N): ").lower()
        # print("Visuals not available currently.")
        if (want_visuals == 'y'):
            print("Creating plotting loss visuals \n ...")
            subprocess.run([sys.executable, "./code/plotting/plot-loss.py"])
            print("Plotting loss sucessfully created!\n Check plots-visuals/new-plots.")

            print("Creating scatterplot of the error vs similarity (vectors) \n ...")
            plot_vector_similarity_loss_graph(
                train_vectors_for_similarity, test_vectors_for_similarity, test_losses_for_similarity)
            print("Scatterplot successfully created! \n Check plots-visuals/new-plots")


            print("Creating scatterplot of the error vs similarity (smiles) \n ...")
            plot_smiles_similarity_loss_graph(
                train_smiles_collected, test_smiles_collected, test_losses_for_similarity)

            print("Creating SME masking graphs \n ...")
            subprocess.run([sys.executable, "./code/plotting/plot_sme_graph.py"])

    print("PROCESS DONE. \n EXITING.")