import pandas as pd
import numpy as np
import torch
from Model_Generator import smiles_to_graph

dest_path = "edited_chromophores.csv"
data_path = "../../data/chromophores.csv"

chromophore_df = pd.read_csv(data_path)
col_headers = chromophore_df.columns.tolist()
chromophore_df = chromophore_df.drop(columns=[
    h for h in col_headers
    if h not in ("Chromophore", "Solvent", "Lifetime (ns)")
])


m_graphs = []
s_graphs = []
v_rows = [] #rows that are valid

for idx, row in chromophore_df.iterrows():
    if np.isnan(row["Lifetime (ns)"]):
        print(f"Lifetime @{idx} is not provided \n SKIPPING")
        continue
    mgraph = smiles_to_graph(row["Chromophore"])
    sgraph = smiles_to_graph(row["Solvent"])
    if mgraph is None or sgraph is None:
        print(f"Cannot parse either molecular or solvent smiles @{idx} \n SKIPPING")
        continue
    m_graphs.append(mgraph)
    s_graphs.append(sgraph)
    v_rows.append(idx)

lifetimes = chromophore_df.loc[v_rows, "Lifetime (ns)"].tolist()

#export
with open("Lifetime.txt", "w") as f:
    for l in lifetimes:
        print(l, file=f)

torch.save(m_graphs, "molecularGraphs.pt")
torch.save(s_graphs, "solventGraphs.pt")