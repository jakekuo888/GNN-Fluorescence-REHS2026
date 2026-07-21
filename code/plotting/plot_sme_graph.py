from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

import os
import matplotlib.pyplot as plt

import numpy as np

print("If you see this, plot_sme_graph.py has begun running. \n ...")

with open("./data/plot-data/sme.json", "r") as f:
	