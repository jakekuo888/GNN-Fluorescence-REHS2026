import pandas as pd
import numpy as np
import torch
from model_generator import smiles_to_graph

def is_null_graph(graph):
    return graph.x.shape[0] == 1 and graph.x.sum() == 0

def generate_and_export_data(dataset, mol_label, sol_label, predicted_name, folder, out_file):
    dest_path = "edited_chromophores.csv"
    data_path = f"./data/{dataset}.csv"

    chromophore_df = pd.read_csv(data_path)
    col_headers = chromophore_df.columns.tolist()

    chromophore_df = chromophore_df.drop(columns=[
        h for h in col_headers
        if h not in (mol_label, sol_label, predicted_name)
    ])

    m_graphs = []
    s_graphs = []
    v_rows = [] #valid_rows

    print("Going through data")

    for idx, row in chromophore_df.iterrows():
        if np.isnan(row[predicted_name]):
            #print(f"Data @{idx} is not provided \n SKIPPING")
            continue

        if row[sol_label] == "gas":
            continue

        if pd.isna(row[mol_label]) or pd.isna(row[sol_label]):
            #print(f"Missing SMILES @{idx} \n SKIPPING")
            continue

        mgraph = smiles_to_graph(row[mol_label])
        sgraph = smiles_to_graph(row[sol_label])
        if mgraph is None or sgraph is None:
            #print(f"Cannot parse either molecular or solvent smiles @{idx} \n SKIPPING")
            continue

        mgraph.smiles = str(row[mol_label])
        sgraph.smiles = str(row[sol_label])

        m_graphs.append(mgraph)
        s_graphs.append(sgraph)
        v_rows.append(idx)

        Data = chromophore_df.loc[v_rows, predicted_name].tolist()

    #export

    print("Uploading data")

    with open(f"./data/{folder}/{out_file}", "w") as f:
        for d in Data:
            print(d, file=f)

    torch.save(m_graphs, f"./data/{folder}/molecularGraphs-{dataset}.pt")
    torch.save(s_graphs, f"./data/{folder}/solventGraphs-{dataset}.pt")

    print("Process DONE")
    print(f"Data points collected: {len(Data)}")
    print(f"{predicted_name} txt @ {folder}/{predicted_name.split()[0]}.txt")