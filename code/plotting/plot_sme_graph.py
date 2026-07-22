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

# reset folder
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

# BRICS families are numbered 1-16, we index FamColor[0] = "no family"
# (fallback gray) and FamColor[1..16] = actual BRICS types, so the same
# family number always maps to the same color across every molecule/image.
FamColor = {0: (0.6, 0.6, 0.6)}  # fallback: fragment with no BRICS boundary
for i in range(1, 17):
    FamColor[i] = FragColors[i - 1]

print(FamColor)


def brics_label_to_family(label):
    # strip trailing letters, e.g. '3a' -> 3, '4b' -> 4
    digits = ''.join(ch for ch in label if ch.isdigit())
    if not digits:
        return 0
    fam = int(digits)
    return fam if fam in FamColor else 0


def fragment_family(labels):
    # a fragment can border multiple BRICS types if it's bonded to more
    # than one neighbor fragment -- use the smallest label as representative
    if not labels:
        return 0
    return brics_label_to_family(sorted(labels)[0])


mol_num = 0

for mol in data[:n_img_gen]:
    mol_num += 1
    struct = Chem.MolFromSmiles(mol['smiles'])

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    opts = drawer.drawOptions()

    opts.useBWAtomPalette()

    # generate colors by BRICS family instead of by fragment index
    HAC = {}
    for frag_id_str, fragment in enumerate(mol['atom_groups']):
        labels = mol['frag_brics_types'].get(str(frag_id_str), [])
        fam = fragment_family(labels)
        color = FamColor[fam]
        for atom in fragment:
            HAC[atom] = color

    abs_vals = [abs(n) for n in mol['fragment_removal'].values()]
    max_val = max(abs_vals) if abs_vals and max(abs_vals) != 0 else 1.0

    radii = {}
    for key, rm_frag in mol['fragment_removal'].items():
        # Scale between 0.2 (minimum radius) and 0.8 (maximum radius)
        scaled_radius = 0.2 + 0.6 * (abs(rm_frag) / max_val)
        for atom in mol['atom_groups'][int(key)]:
            radii[atom] = scaled_radius

    drawer.DrawMolecule(
        struct,
        highlightAtoms=list(range(struct.GetNumAtoms())),
        highlightAtomColors=HAC,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    with open(f'{fpath}M{mol_num}-SME.png', 'wb') as f:
        f.write(drawer.GetDrawingText())