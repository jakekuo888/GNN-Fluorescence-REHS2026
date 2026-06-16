import torch
from torch import nn

class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    #insert any other important things

  def forward(self, inp):
    x = self.flatten(x)

#more stuff
