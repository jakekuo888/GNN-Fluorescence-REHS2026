import pandas as pd
import numpy as np
import torch
from wrangling.model_generator import smiles_to_graph, smiles_to_morgan_fp, resolve_smiles
from rdkit.Chem import rdFingerprintGenerator
import os
import json
import pickle

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def is_null_graph(graph):
    return graph.x.shape[0] == 1 and graph.x.sum() == 0


y_values = []

CACHE_FILE = './data/solvent_cache.json'
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        SOLVENT_SMILES = json.load(f)
else:
    SOLVENT_SMILES = {}

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def generate_and_export_data(dataset, mol_label, sol_label, predicted_name, folder, out_file):
    print("Going through data")

    data_path = f"./data/{dataset}.csv"

    chromophore_df = pd.read_csv(data_path)
    col_headers = chromophore_df.columns.tolist()

    chromophore_df = chromophore_df.drop(columns=[
        h for h in col_headers
        if h not in (mol_label, sol_label, predicted_name)
    ])

    valid_rows = []
    for idx, row in chromophore_df.iterrows():
        if pd.isna(row[predicted_name]):
            continue
        if row[sol_label] == "gas":
            continue
        if pd.isna(row[mol_label]) or pd.isna(row[sol_label]):
            continue
        valid_rows.append(row)

    unique_solvents = {row[sol_label] for row in valid_rows}
    for sol_name in unique_solvents:
        resolve_smiles(sol_name, SOLVENT_SMILES, CACHE_FILE)

    unique_mol_smiles = list({row[mol_label] for row in valid_rows})

    SHARD_SIZE = 500
    shard_files = []
    output_dir = "temp_shards"
    os.makedirs(output_dir, exist_ok=True)

    #Process molecules in shards of SHARD_SIZE
    
    print("QUICK INFO:")
    print(f"\n Est. total number of shards: {len(unique_mol_smiles)//SHARD_SIZE + 1}")
    print(f" There are {len(unique_mol_smiles)} to process in total.")
    print(f" Shard size of {SHARD_SIZE}. Temp shards are located within ./{output_dir}/...")
    print("-------------------\n")

    for i in range(0, len(unique_mol_smiles), SHARD_SIZE):
        shard_smiles = unique_mol_smiles[i: i + SHARD_SIZE]
        shard_index = i // SHARD_SIZE
        shard_path = os.path.join(output_dir, f"shard_{shard_index}.pickle")

        shard_dict = {}

        print(
            f"\nProcessing shard {shard_index + 1} (molecules {i} to {i + len(shard_smiles)})...")

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(smiles_to_graph, s): s for s in shard_smiles}

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Shard {shard_index + 1}"):
                s = futures[fut]
                shard_dict[s] = fut.result()

        # Save the current shard to disk immediately
        # Note: If your graph objects aren't JSON-serializable, use `pickle.dump` instead.
        with open(shard_path, 'bw') as f:
            pickle.dump(shard_dict, f)

        shard_files.append(shard_path)

    # 2. Combine all shards at the end
    print("\nCombining all shards into final dictionary...")
    smiles_to_dict = {}

    for shard_path in shard_files:
        with open(shard_path, 'br') as f:
            shard_data = pickle.load(f)
            smiles_to_dict.update(shard_data)

    # Optional: Clean up the shard file after merging to save disk space
    os.remove(shard_path)

    # Optional: Clean up the empty temp directory
    if os.path.exists(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)

    final_m_dicts, s_prints, y_values = [], [], []
    for row in valid_rows:
        mol_dict = smiles_to_dict.get(row[mol_label])
        if mol_dict is None:
            continue
        sol_smiles = resolve_smiles(row[sol_label], SOLVENT_SMILES, CACHE_FILE)
        if sol_smiles is None:
            print(f"Failed to resolve: '{row[sol_label]}'")
            continue
        final_m_dicts.append(mol_dict)
        s_prints.append(smiles_to_morgan_fp(fp_gen, sol_smiles))
        y_values.append(row[predicted_name])
    # export

    print("Uploading data")

    with open(f"./data/{folder}/{out_file}", "w") as f:
        for d in y_values:
            print(d, file=f)

    fp_matrix = np.vstack(s_prints)

    torch.save(final_m_dicts, f"./data/{folder}/molecularGraphs-{dataset}.pt")
    # torch.save(s_graphs, f"./data/{folder}/solventGraphs-{dataset}.pt")
    np.savez_compressed(
        f"./data/{folder}/solventFingerprints-{dataset}.npz", fps=fp_matrix)

    print("Process DONE")
    print(f"Data points collected: {len(y_values)}")
    print(f"{predicted_name} txt @ {folder}/{predicted_name.split()[0]}.txt")
