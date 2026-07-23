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
import random

from models.neural_networks import ModelTwo
from models.goms_sme import Model
from models.early_stop import EarlyStop
from models.goms_sme import FragEGNN
from plotting.plot_loss import plot_loss, plot_error_histogram

from setup.process_data import absorption_data_options, generate_graphs_labels, FragmentDataset, collate_fn, PredOption, has_reasonable_geometry
from models.sme import sme_attribution

import json

from rdkit import Chem

# EASY CONTROLS vvv
n_epochs = 1
collect_data = True
early_stopper = EarlyStop(9, 0.005)
re_generate_data = True
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
FACTOR = 0.1
DROPOUT = 0.2
ERROR_BOUND = 150
# EASY CONTROLS ^^^

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(root_dir, 'data-wrangling'))
sys.path.append(os.path.join(root_dir, 'plots-visuals'))

if __name__ == "__main__":
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    set_seed(42)

    d4c_absorption = PredOption(
        "d4c", "Absorption max (nm)", "absorption-data", "absorption-d4c.txt")
    qmwf_absorption = PredOption(
        "qmwf", "lambda_max (Exp nm)", "absorption-data", "absorption-qmwf.txt")

    nabla_train = PredOption(
        "nabla_train", "peakwavs_max", "absorption-data", "abs-nabla-train.txt")
    nabla_val = PredOption("nabla_val", "peakwavs_max",
                           "absorption-data", "abs-nabla-val.txt")
    nabla_test = PredOption("nabla_test", "peakwavs_max",
                            "absorption-data", "abs-nabla-test.txt")

    # Nabla Colors
    molecules_dicts_train, y_mean, y_std, train_smiles_for_similarity, train_solv_features = generate_graphs_labels(
        nabla_train, generate_data=re_generate_data)

    molecules_dicts_val, val_y_mean, val_y_std, val_smiles_for_similarity, val_solv_features = generate_graphs_labels(
        nabla_val, generate_data=re_generate_data, y_mean=y_mean, y_std=y_std, normalize=False)

    molecules_dicts_test, ts_y_mean, ts_y_std, test_smiles_for_similarity, test_solv_features = generate_graphs_labels(
        nabla_test, generate_data=re_generate_data, y_mean=y_mean, y_std=y_std, normalize=False)

    # External Set
    ext_dataset, ext_y_mean, ext_y_std, test_smiles_for_similarity, test_solv_features = generate_graphs_labels(
        qmwf_absorption, generate_data=re_generate_data, y_mean=y_mean, y_std=y_std, normalize=False)

    ALLOWED_ATOMIC_NUMS = {1, 5, 6, 7, 8, 9,
                           14, 15, 16, 17, 32, 34, 35, 50, 52, 53}

    datasets = [molecules_dicts_train, molecules_dicts_val,
                molecules_dicts_test, ext_dataset]
    labels = ["train", "val", "test", "ext"]

    for dataset, label in zip(datasets, labels):
        before = len(dataset)
        dataset = [d for d in dataset if has_reasonable_geometry(d)]
        print(
            f"Filtered {before - len(dataset)} molecules with corrupted geometry from {label} set")

    y_vals = torch.tensor([d["y_normalized"] for d in molecules_dicts_train])
    baseline_mae_normalized = y_vals.abs().mean()
    print(
        f"Baseline (predict mean) normalized MAE: {baseline_mae_normalized:.4f}")

    y_real_vals = torch.tensor([d["y_real"] for d in molecules_dicts_train])
    baseline_mae_real = (y_real_vals - y_real_vals.mean()).abs().mean()
    print(f"Baseline (predict mean) real-units MAE: {baseline_mae_real:.4f}")

    # Loaders (returns the list of all fragments, maps, and dicts for later use)
    train_loader = TorchLoader(FragmentDataset(
        molecules_dicts_train), batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = TorchLoader(FragmentDataset(
        molecules_dicts_val), batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_loader = TorchLoader(FragmentDataset(
        molecules_dicts_test), batch_size=128, shuffle=False, collate_fn=collate_fn)
    ext_loader = TorchLoader(FragmentDataset(
        ext_dataset), batch_size=256, shuffle=False, collate_fn=collate_fn)

    # Set up the Model class (GNN/FFNN), AdamW optimizer, and MAE Loss function
    node_features = molecules_dicts_train[0]["frag_graphs"][0].num_node_features
    edge_features = molecules_dicts_train[0]["frag_graphs"][0].num_edge_features
    gs_edge_features = molecules_dicts_train[0]["edge_attr"].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(node_features, edge_features, 64,
                  gs_edge_features, train_solv_features, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=FACTOR, patience=PATIENCE)
    stopper = EarlyStop(10, 0.005)

    criterion = torch.nn.L1Loss()

    # Train takes in mol & sol loader, zips them to return a forward pass through the model, loss, backprop, repeat

    def train(model, opt, loader, log_losses=None):
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
                             dtype=torch.float, device=device).view(-1)

            out = out.view(-1)

            loss = criterion(out, y)
            if log_losses is not None:
                log_losses.append(loss.item())
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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

        norm_total_mae, human_total_mae = 0.0, 0.0
        signed_total = 0.0
        total_graphs = 0
        outliers = []
        massive = []

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
                                 dtype=torch.float, device=device).view(-1)
                out = out.view(-1)

                norm_loss = criterion(out, y)

                # Reverse normalization for prediction
                pred_log = out.squeeze(-1) * std + mean
                pred_log = torch.clamp(pred_log, max=12)
                pred_actual = torch.exp(pred_log)
                pred_actual = pred_actual.flatten()

                # Reverse normalization for target too
                target_actual = torch.tensor(
                    [d["y_real"] for d in mol_dicts], dtype=torch.float, device=device).view(-1)

                loss_tensor = torch.abs(pred_actual - target_actual)

                human_loss = torch.mean(torch.abs(pred_actual - target_actual))

                if not torch.isfinite(human_loss) or human_loss.item() > 1e6:
                    skipped += 1
                    offending_smiles = [d["smiles"] for d in mol_dicts]
                    print(
                        f"!!! Skipping eval batch, loss={human_loss.item()}. Molecules: {offending_smiles}")
                    continue  # don't let this batch's inf poison the epoch average

                for d, loss in zip(mol_dicts, loss_tensor.flatten()):
                    index = int(loss.item()//100)
                    if len(outliers) < index+1:
                        while len(outliers) < index+1:
                            outliers.append(0)
                    outliers[index] = outliers[index] + 1

                    if loss.item() >= 500:
                        massive.append(d["smiles"])

                signed_bias = (pred_actual - target_actual).mean()
                signed_total += (pred_actual - target_actual).sum()

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

                    norm_total_mae += norm_loss * num_graphs
                    human_total_mae += human_loss * num_graphs

                    total_graphs += num_graphs
            # print(f"  [test] skipped {skipped}/{total} batches")
        if compute_mae:
            norm_avg_mae = norm_total_mae / total_graphs
            human_avg_mae = human_total_mae / total_graphs
            avg_signed_bias = signed_total / total_graphs

            return norm_avg_mae, human_avg_mae, avg_signed_bias, outliers, massive
        else:
            return 0.0, 0.0, 0.0, [], []

    def run_model(model, train_loader, val_loader, opt, sched, stopper):
        # Train & Test the Model
        with open(f"./data/plot-data/nm-loss.txt", "w") as f_, open(f"./data/plot-data/norm-loss.txt", 'w') as f2_:
            for epoch in range(1, n_epochs+1):
                train(model, opt, train_loader)

                train_avg_mae_norm, train_avg_mae_human, tr_signed_bias, _, _ = test(
                    model, train_loader, y_mean, y_std)
                val_avg_mae_norm, val_avg_mae_human, val_signed_bias, _, _ = test(
                    model, val_loader, y_mean, y_std)
                sched.step(float(val_avg_mae_human))

                if stopper.stop_early(val_avg_mae_norm, model):
                    print(f'Early stop has been initiated on Epoch #{epoch}')
                    stopper.restore_best(model)
                    break

                print(
                    f"Epoch #{epoch} | Train Average MAE (norm): {train_avg_mae_norm:.4f} | Train Average MAE (nm): {train_avg_mae_human:.4f} | Test Average MAE (norm): {val_avg_mae_norm} | Test Average MAE (nm): {val_avg_mae_human:.4f} | Early stopper count: {stopper.count}")
                # print(f"epoch {epoch}: avg signed bias = {tr_signed_bias:.4f}")

                if (collect_data):
                    # loading data for plotting (train, test)
                    print(
                        f"{train_avg_mae_human:.4f}, {val_avg_mae_human:.4f}", file=f_)
                    print(
                        f"{train_avg_mae_norm:.4f}, {val_avg_mae_norm:.4f}", file=f2_)

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
            print(f"NORMALIZATION MEAN: {y_mean} | NORMALIZATION STD: {y_std}")
            plot_loss("nm-loss", "nm")
            plot_loss("norm-loss", "norm")

            _, _, _, tr_outliers, tr_massive = test(
                model, train_loader, y_mean, y_std, compute_mae=True)
            _, _, _, ts_outliers, ts_massive = test(
                model, test_loader, y_mean, y_std, compute_mae=True)
            _, _, _, ex_outliers, ex_massive = test(
                model, ext_loader, y_mean, y_std, compute_mae=True)

            smiles = ""
            for s in (tr_massive + ts_massive + ex_massive):
                smiles = smiles + f"{s}, "
            print(f"SMILES WITH ERROR ABOVE 10k NM: {smiles}\n")

            plot_error_histogram(tr_outliers, "train")
            plot_error_histogram(ts_outliers, "test")
            plot_error_histogram(ex_outliers, "ext")
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