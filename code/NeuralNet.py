import torch
import numpy as np
import pandas as pd

from torch import nn

class NeuralNet(nn.Module):
  def __init__(self, in_size, out_size, hidden_sizes=[64, 64]):
    super().__init__()
    
    self.inS = in_size
    self.outS = out_size
    self.hidS = hidden_sizes
    self.layers = []
    p_size = self.inS

    #set layers
    for h in self.hidS:
      self.layers.append(nn.Linear(p_size, h))
      self.layers.append(nn.ReLU())
      p_size = h
    self.layers.append(nn.Linear(p_size, self.outS))
    self.net = nn.Sequential(*self.layers)

  def forward(self, inp):
    return self.net(inp)
