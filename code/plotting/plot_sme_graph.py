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

    # No AddHs here -- atom_groups was filtered to heavy-atom-only indices
    # at write time, and heavy-atom indices are unchanged by AddHs, so
    # plain MolFromSmiles indices line up correctly.
    struct = Chem.MolFromSmiles(mol['smiles'])

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    opts = drawer.drawOptions()

    opts.useBWAtomPalette()
    opts.highlightBondWidthMultiplier = 20

    # atom -> fragment color map
    fragN = 0
    HAC = {}
    atom_to_frag_color = {}
    for fragment in mol['atom_groups']:
        color = FragColors[fragN % len(FragColors)]
        for atom in fragment:
            HAC[atom] = color
            atom_to_frag_color[atom] = color
        fragN += 1

    # radii scaled by attribution magnitude, per-fragment
    vals = np.array(list(mol['fragment_removal'].values()))
    vmin, vmax = vals.min(), vals.max()

    radii = {}
    for key, rm_frag in mol['fragment_removal'].items():
        if vmax == vmin:
            radius = 0.3
        else:
            radius = 0.3 + 0.4 * (rm_frag - vmin) / (vmax - vmin)

        for atom in mol['atom_groups'][int(key)]:
            radii[atom] = radius

    # only highlight bonds INSIDE a fragment (both endpoints same fragment),
    # colored to match -- this is what makes fragments read as solid
    # connected blobs instead of separately-colored dots. Bonds BETWEEN
    # fragments (the ones BRICS actually cut) are left unhighlighted so
    # fragment boundaries are visually clear.
    highlight_bonds = []
    HBC = {}
    for bond in struct.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        c1 = atom_to_frag_color.get(a1)
        c2 = atom_to_frag_color.get(a2)
        if c1 is not None and c1 == c2:
            highlight_bonds.append(bond.GetIdx())
            HBC[bond.GetIdx()] = c1

    drawer.DrawMolecule(
        struct,
        highlightAtoms=list(HAC.keys()),
        highlightBonds=highlight_bonds,
        highlightAtomColors=HAC,
        highlightBondColors=HBC,
        highlightAtomRadii=radii,
    )

    drawer.FinishDrawing()

    with open(f'{fpath}M{mol_num}-SME.png', 'wb') as f:
        f.write(drawer.GetDrawingText())

print(f"Done. {mol_num} images written to {fpath}")