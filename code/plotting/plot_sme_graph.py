from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs
import numpy as np

import sys
import matplotlib.pyplot as plt

import json
import os
import numpy as np
import shutil

n_img_gen = 10

print(f"Plot_sme_graph.py is running \n Generating {n_img_gen} images \n ...")

try:
	with open("./data/plot-data/sme.json", "r") as f:
		data = json.load(f)
except Exception as e:
	print("Error retrieving ./data/plot-data/sme.json \n Try running ./code/train_test.py")
	print(f"Full error code: {e}")
	sys.exit(e)

fpath = "./plots-visuals/SME-Graphs/"

#reset folder
if os.path.exists(fpath):
    shutil.rmtree(fpath)

os.makedirs(fpath)


FragColors = [
    (1.0, 0.0, 0.0),
    (0.0, 0.5, 1.0),
    (0.0, 1.0, 0.2),
    (1.0, 0.5, 0.0),
    (0.5, 0.0, 1.0),
    (1.0, 0.0, 0.7),
    (0.0, 1.0, 0.9),
    (1.0, 0.9, 0.0),
    (0.2, 0.0, 1.0),
    (0.0, 0.8, 0.5),
    (1.0, 0.2, 0.5),
    (0.5, 1.0, 0.0),
    (0.8, 0.0, 1.0),
    (0.0, 0.4, 0.8),
    (0.0, 1.0, 0.6),
    (1.0, 0.7, 0.0),
    (0.7, 1.0, 0.0),
    (0.0, 0.6, 1.0),
    (1.0, 0.0, 0.3),
    (0.3, 1.0, 0.0),
    (0.9, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.8, 0.4),
    (0.4, 0.8, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 0.7, 0.3),
    (0.0, 0.5, 0.8),
    (0.9, 1.0, 0.0),
    (1.0, 0.3, 0.0),
    (0.6, 0.0, 0.5),
    (0.5, 0.8, 1.0),
    (0.0, 0.9, 0.8),
    (0.8, 0.0, 0.5),
    (0.5, 0.0, 0.8),
    (0.6, 1.0, 0.5),
    (1.0, 0.4, 0.5),
    (0.0, 0.8, 1.0),
    (1.0, 0.6, 0.7),
    (0.7, 0.0, 1.0),
    (0.0, 1.0, 0.4),
    (1.0, 0.5, 0.3),
    (0.0, 0.3, 1.0),
    (1.0, 0.1, 0.0),
    (0.3, 0.0, 0.8),
    (0.8, 1.0, 0.0),
    (0.6, 0.0, 1.0),
    (1.0, 0.0, 0.8),
    (0.0, 0.4, 0.6),
    (1.0, 0.5, 0.8),
    (0.2, 1.0, 0.5)
]

mol_num = 0

for mol in data[:n_img_gen]:
	mol_num += 1
	struct = Chem.MolFromSmiles(mol['smiles'])

	drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
	opts = drawer.drawOptions()

	opts.useBWAtomPalette()

	#generate group colors
	fragN = 0
	HAC = {}
	for fragment in mol['atom_groups']:
		for atom in fragment:
			HAC[atom] = FragColors[fragN]
		fragN += 1

	radii = {}

	drawer.DrawMolecule(
		struct,
		highlightAtoms = list(range(struct.GetNumAtoms())),
		highlightAtomColors = HAC
	)
	drawer.FinishDrawing()
	with open(f'{fpath}M{mol_num}-SME.png', 'wb') as f:
		f.write(drawer.GetDrawingText())
