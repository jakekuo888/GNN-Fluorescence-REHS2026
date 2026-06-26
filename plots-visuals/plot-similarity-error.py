from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

import os
import matplotlib.pyplot as plt

import numpy as np

def dice_similarity(u, v):
    # Standard continuous Dice for non-negative embeddings
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

	output_similarities = []

	for vec in test_vec:
		max_dice_sim = 0.0
		for pot_vec in train_vec:
			temp_sim = dice_similarity(vec, pot_vec)
			if temp_sim > max_dice_sim:
				max_dice_sim = temp_sim
		output_similarities.append(max_dice_sim)

	plt.scatter(output_similarities, loss_list, c=loss_list, cmap='Blues', edgecolors='black')
	plt.colorbar(label='')
	plt.xlabel('Dice Similarity')
	plt.ylabel('MAE')
	plt.savefig("./plots-visuals/new-plots/vec-similarity.png")