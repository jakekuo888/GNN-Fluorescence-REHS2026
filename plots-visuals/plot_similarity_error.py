from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

import os
import matplotlib.pyplot as plt

import numpy as np

def dice_similarity_matrix(train_mat, test_mat):
    dot_products = test_mat @ train_mat.T
    test_sq = np.sum(test_mat**2, axis=1, keepdims=True)
    train_sq = np.sum(train_mat**2, axis=1, keepdims=True)
    denominators = test_sq + train_sq.T
    similarities = np.where(denominators > 0, 2.0 * dot_products / denominators, 0.0)
    return similarities.max(axis=1)

def smiles_to_morgan_fp(fp_gen, smiles):
	mol = Chem.MolFromSmiles(smiles)
	bit_vect = fp_gen.GetFingerprint(mol)
	fp_array = np.zeros((2048,), dtype=np.int8)
	Chem.DataStructs.ConvertToNumpyArray(bit_vect, fp_array)

	return fp_array

def process_fps(train_smiles_list, test_smiles_list):
	fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

	train_fps = []
	for smiles in train_smiles_list:
		fp_arr = smiles_to_morgan_fp(fp_gen, smiles)
		train_fps.append(fp_arr)

	test_fps = []
	for smiles in test_smiles_list:
		fp_arr = smiles_to_morgan_fp(fp_gen, smiles)
		test_fps.append(fp_arr)
	
	return train_fps, test_fps

def plot_smiles_similarity_loss_graph(train_smiles_list, test_smiles_list, loss_list):
	plt.title(f'MAE vs Dice Similarity for GNN Output Vectors (Pre-GNN)')

	train_fps, test_fps = process_fps(train_smiles_list, test_smiles_list)
	train_mat = np.stack(train_fps)
	test_mat = np.stack(test_fps)

	output_similarities = dice_similarity_matrix(train_mat, test_mat)
	plt.scatter(output_similarities, loss_list, c=loss_list, cmap='Reds', edgecolors='black')
	plt.colorbar(label='')
	plt.xlabel('Dice Similarity')
	plt.ylabel('MAE')
	plt.savefig("./plots-visuals/new-plots/fp-similarity.png")

	plt.close()

def plot_vector_similarity_loss_graph(train_vec, test_vec, loss_list):
	plt.title(f'MAE vs Dice Similarity of GNN Output Vectors (Post-GNN)')
	
	train_mat = np.stack([v.detach().cpu().numpy() for v in train_vec])  # (n_train, hidden)
	test_mat = np.stack([v.detach().cpu().numpy() for v in test_vec])    # (n_test, hidden)

	output_similarities = dice_similarity_matrix(train_mat, test_mat)

	plt.scatter(output_similarities, loss_list, c=loss_list, cmap='Blues', edgecolors='black')
	plt.colorbar(label='')
	plt.xlabel('Dice Similarity')
	plt.ylabel('MAE')
	plt.savefig("./plots-visuals/new-plots/vec-similarity.png")

	plt.close()