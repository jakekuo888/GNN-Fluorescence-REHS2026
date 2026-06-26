from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import os
import matplotlib.pyplot as plt

#title and other information on graph
plt.title(f'Error vs Similarity of molecules')

#data parsing & finding similarity (incomplete)
actual = []
predicted = []
mol_smiles = []
solvents = []

#Reading in data
with open("./data/plot-data/plot-similarity.txt") as f:
	for line in f.read().splitlines():
		items = line.split(',')
		actual.append(float(items[0]))
		predicted.append(float(items[1]))
		mol_smiles.append(items[2])
		solvents.append(items[3])

mol = []
for m in mol_smiles:
	mol.append(Chem.MolFromSmiles(m))