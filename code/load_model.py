import torch
import torch.nn as nn
from torch_geometric.data import Data

import numpy as np
import pandas as pd

from neural_networks import ModelTwo
#from process_data import 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = []
for i in range(9):
	#path to retrieve the model weights
	#MAKE SURE TO CHANGE IF MOVED
	path = f"./models/model_{i}_weights.pth"
	#model = ModelTwo()
	#torch.load(path)