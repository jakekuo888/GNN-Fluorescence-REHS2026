import pandas as pd
import Model_Generator
from Model_Generator import smiles_to_graph


# DATA: https://www.nature.com/articles/s41597-020-00634-8

dest_path = "edited_chromophores.csv"

chromophore_df = pd.read_csv('../../data/chromophores.csv')
column_headers = chromophore_df.columns.to_list()
chromophore_df = chromophore_df.drop(columns=[header for header in column_headers if header not in ("Chromophore", "Solvent", "Lifetime (ns)")])

chromophore_df.to_csv(dest_path, index = False)

#molecule
for smiles in chromophore_df.iloc[:, 0]:
    nData = smiles_to_graph(smiles)

#solvent
for smiles in chromophore_df.iloc[:, 1]:
    nData = smiles_to_graph(smiles)

#print(smiles_to_graph(chromophore_df.iloc[1, 0]))