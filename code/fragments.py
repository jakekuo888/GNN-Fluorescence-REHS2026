from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom

import numpy as np
import os 
import sys


def returnFragments(mol):
	fragment_mol = Chem.GetMolFrags(mol, asMols = True)
	AllChem.EmbedMolecule(fragment_mol, AllChem.ETKDG())
	return fragment_mol

def Fragments_Graph(smiles):
	mol = Chem.MolFromSmiles(smiles)
	mol = Chem.AddHs(mol)

	fmol = returnFragments(mol)
	
	results = []