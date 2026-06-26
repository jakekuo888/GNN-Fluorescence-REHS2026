from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import os
import matplotlib.pyplot as plt

def plot_smiles_similarity_loss_graph():
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

def plot_vector_similarity_loss_graph(train_vec, test_vec, loss_list):
	plt.title(f'MAE vs Similarity of GNN Output Vectors (D4C & QMWF Datasets)')

	compare_to = []

	for vec in test_vec:
		for pot_vec in train_vec:
			