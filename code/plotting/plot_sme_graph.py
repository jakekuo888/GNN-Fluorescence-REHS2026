from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import sys
import json
import os
import shutil 

n_img_gen = 2

print(f"Plot_sme_graph.py is running \n Generating {n_img_gen*2} images \n ...")

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


FamColor = {0: (0.6, 0.6, 0.6)}

for i in range(1, 17):
    FamColor[i] = FragColors[i - 1]

def brics_label_to_family(label):
    # strip trailing letters, e.g. '3a' -> 3, '4b' -> 4
    digits = ''.join(ch for ch in label if ch.isdigit())
    if not digits:
        return 0
    fam = int(digits)
    return fam if fam in FamColor else 0


def fragment_family(labels):
    if not labels:
        return 0
    return brics_label_to_family(sorted(labels)[0])


mol_num = 0

print(f"Making {n_img_gen} BRICS images.")
for mol in data[:n_img_gen]:
    mol_num += 1
    struct = Chem.MolFromSmiles(mol['smiles'])

    drawer = rdMolDraw2D.MolDraw2DCairo(700, 700)
    opts = drawer.drawOptions()

    opts.useBWAtomPalette()

    # generate colors by BRICS family
    HAC = {}
    Fams = []
    for frag_id_str, fragment in enumerate(mol['atom_groups']):
        labels = mol['frag_brics_types'].get(str(frag_id_str), [])
        fam = fragment_family(labels)
        color = FamColor[fam]
        Fams.append(fam)
        for atom in fragment:
            HAC[atom] = color

    abs_vals = [abs(n) for n in mol['fragment_removal'].values()]
    max_val = max(abs_vals) if abs_vals and max(abs_vals) != 0 else 1.0

    radii = {}
    for key, rm_frag in mol['fragment_removal'].items():
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
    PATH = f'{fpath}M{mol_num}-BRICS-SME.png'
    with open(PATH, 'wb') as f:
        f.write(drawer.GetDrawingText())

    img = Image.open(PATH)
    W, H = img.size
    left_margin = 200
    W += left_margin
    canv_mode = img.mode if img.mode in ("RGB", "RGBA") else "RGB"
    bg_color = (255, 255, 255, 255) if canv_mode == "RGBA" else (255, 255, 255)

    nImg = Image.new(canv_mode, (W, H), bg_color)
    nImg.paste(img, (left_margin, 0))
    draw = ImageDraw.Draw(nImg)

    txtX = 70
    txtY = 150

    try:
        font = ImageFont.truetype("./data/inter.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()


    draw.text((txtX, txtY), "Legend:", fill = (0, 0, 0), font = font)

    for F_ in Fams:
        NT = tuple(min(int(float(FN)*255), 255) for FN in FamColor[F_])
        txtY += int(1.6*font.size)
        l, t, r, b = draw.textbbox((txtX, txtY), f"FTYPE-{F_}", font = font)
        pad = 6
        h_box = (l-pad, t - pad, r + pad, b + pad)
        draw.rectangle(h_box, fill=NT)
        draw.text((txtX, txtY), f"FTYPE-{F_}", fill = (0, 0, 0), font = font)

    nImg.save(PATH)


print(f"Process (1) done \nMaking {n_img_gen} FRAG images.")

mol_num = 0
for mol in data[:n_img_gen]:
    mol_num += 1
    struct = Chem.MolFromSmiles(mol['smiles'])

    drawer = rdMolDraw2D.MolDraw2DCairo(700, 700)
    opts = drawer.drawOptions()

    opts.useBWAtomPalette()

    #generate fragment colors
    fragN = 0
    HAC = {}
    for fragment in mol['atom_groups']:
        for atom in fragment:
            HAC[atom] = FragColors[fragN]
        fragN += 1

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
    with open(f'{fpath}M{mol_num}-FRAG-SME.png', 'wb') as f:
        f.write(drawer.GetDrawingText())

print("PROCESS DONE")