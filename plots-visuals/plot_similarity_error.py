from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

import os
import matplotlib.pyplot as plt

import numpy as np

def dice_similarity(u, v):
    u = u.detach().cpu().numpy()
    v = v.detach().cpu().numpy()
    dot_product = np.dot(u, v)
    denominator = np.sum(u**2) + np.sum(v**2)
    return (2.0 * dot_product) / denominator if denominator > 0 else 0.0

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
	
	train_mat = np.stack([v.detach().cpu().numpy() for v in train_vec])  # (n_train, hidden)
	test_mat = np.stack([v.detach().cpu().numpy() for v in test_vec])    # (n_test, hidden)

	# vectorized dice similarity
	dot_products = test_mat @ train_mat.T  # (n_test, n_train)
	test_sq = np.sum(test_mat**2, axis=1, keepdims=True)   # (n_test, 1)
	train_sq = np.sum(train_mat**2, axis=1, keepdims=True)  # (n_train, 1)
	denominators = test_sq + train_sq.T  # (n_test, n_train)
	
	similarities = np.where(denominators > 0, 2.0 * dot_products / denominators, 0.0)
	output_similarities = similarities.max(axis=1)  # (n_test,) max over train set

	print("OUT_SIM LEN: " + str(len(output_similarities)))

	plt.scatter(output_similarities, loss_list, c=loss_list, cmap='Blues', edgecolors='black')
	plt.colorbar(label='')
	plt.xlabel('Dice Similarity')
	plt.ylabel('MAE')
	plt.savefig("./plots-visuals/new-plots/vec-similarity.png")