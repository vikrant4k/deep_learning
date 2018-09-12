"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    """
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(3 * 32 * 32, 100)
    self.linear2 = torch.nn.Linear(100, 10)




  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    print("forward called")
    out=None
    x=self.linear1(x)
    out_x=F.relu(x)
    input_in_2=self.linear2(out)
    print(input_in_2.shape)
    out=F.softmax(input_in_2)

    return out


import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(3*32*32, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.linear1(x)
        out_x = F.relu(x)
        y_pred = self.linear2(out_x)
        out=F.softmax(y_pred)
        print(out.shape)
        return out



