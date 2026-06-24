import pandas as pd
import numpy as np
import torch
from model_generator import smiles_to_graph

import os
print("Current folder:", os.getcwd())

dest_path = "edited_chromophores.csv"
data_path = "./data/chromophores.csv"

chromophore_df = pd.read_csv(data_path)
col_headers = chromophore_df.columns.tolist()

predicted_name = "Absorption max (nm)"
folder = "absorption-data"

chromophore_df = chromophore_df.drop(columns=[
    h for h in col_headers
    if h not in ("Chromophore", "Solvent", predicted_name)
])


m_graphs = []
s_graphs = []
v_rows = [] #valid_rows

print("Going through data")

for idx, row in chromophore_df.iterrows():
    if np.isnan(row[predicted_name]):
        #print(f"Data @{idx} is not provided \n SKIPPING")
        continue

    if row["Solvent"] == "gas":
        #replace with dummy variable?
        print(f"SKIP@{idx}: Solvent is labeled as 'gas'")
        continue

    mgraph = smiles_to_graph(row["Chromophore"])
    sgraph = smiles_to_graph(row["Solvent"])
    if mgraph is None or sgraph is None:
        #print(f"Cannot parse either molecular or solvent smiles @{idx} \n SKIPPING")
        continue
    m_graphs.append(mgraph)
    s_graphs.append(sgraph)
    v_rows.append(idx)

Data = chromophore_df.loc[v_rows, predicted_name].tolist()

#export

print("Uploading data")

with open(f"./data/{folder}/{predicted_name.split()[0]}.txt", "w") as f:
    for d in Data:
        print(d, file=f)

torch.save(m_graphs, f"./data/{folder}/molecularGraphs.pt")
torch.save(s_graphs, f"./data/{folder}/solventGraphs.pt")

print("Process DONE")
print(f"Data points collected: {len(Data)}")
print(f"{predicted_name} txt @ {folder}/{predicted_name.split()[0]}.txt")