from rdkit import Chem
import numpy as np
import os 
import sys


def returnFragments(mol):
	#Make sure that you input as struct && !smiles
	fragment_mol = Chem.GetMolFrags(mol, asMols = True)
	return fragment_mol

def removeFragments(smiles, amount = -1):
	mol = Chem.MolFromSmiles(smiles)
	fmol = returnFragments(mol)
	
	if amount == -1:
		n_frag = len(fmol)
	else:
		n_frag = amount

	results = []

	#turn each of the fragments as a vector embedding
	#find a way to connect them back as a graph of vector embeddings