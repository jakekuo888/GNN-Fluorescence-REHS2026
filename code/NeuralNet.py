import torch
import numpy as np
import pandas as pd

from torch import nn


#feed forward aspect of project
class NeuralNet(nn.Module):
  def __init__(self, in_size, out_size, hidden_sizes=None):
    super().__init__()

    if hidden_sizes == None:
      hidden_sizes = [67, 67]
    
    self.inS = in_size
    self.outS = out_size
    self.hidS = hidden_sizes
    layers = []
    p_size = self.inS

    #set layers
    for h in self.hidS:
      layers.append(nn.Linear(p_size, h))
      layers.append(nn.ReLU())
      p_size = h
    layers.append(nn.Linear(p_size, self.outS))
    self.net = nn.Sequential(*layers)

  def forward(self, inp):
    return self.net(inp)

mainNN = NeuralNet(10, 1, [20, 20, 20])
#todo: training & configure data?