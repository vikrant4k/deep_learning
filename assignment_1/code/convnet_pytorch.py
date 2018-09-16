"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 3,padding=1)
    self.bat1 = nn.BatchNorm2d(64)
    self.max1=nn.MaxPool2d(3,2,padding=1)

    self.conv2 = nn.Conv2d(64, 128, 3,padding=1)
    self.bat2 = nn.BatchNorm2d(128)
    self.max2= nn.MaxPool2d(3, 2,padding=1)

    self.conv3_a = nn.Conv2d(128, 256, 3,padding=1)
    self.bat3_a = nn.BatchNorm2d(256)
    self.conv3_b = nn.Conv2d(256, 256, 3,padding=1)
    self.bat3_b = nn.BatchNorm2d(256)
    self.max3 = nn.MaxPool2d(3, 2,padding=1)

    self.conv4_a = nn.Conv2d(256, 512, 3,padding=1)
    self.bat4_a = nn.BatchNorm2d(512)
    self.conv4_b = nn.Conv2d(512, 512, 3,padding=1)
    self.bat4_b = nn.BatchNorm2d(512)
    self.max4 = nn.MaxPool2d(3, 2,padding=1)

    self.conv5_a = nn.Conv2d(512, 512, 3,padding=1)
    self.bat5_a = nn.BatchNorm2d(512)
    self.conv5_b = nn.Conv2d(512, 512, 3,padding=1)
    self.bat5_b = nn.BatchNorm2d(512)
    self.max5 = nn.MaxPool2d(3, 2,padding=1)
    self.avgpool=nn.AvgPool2d(1,stride=1,padding=0)
    self.linear_layer=nn.Linear(512,10)

    ########################
    # END OF YOUR CODE    #
    #######################

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

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out_1=self.max1(F.relu(self.bat1(self.conv1(x))))
    out_2=self.max2(F.relu(self.bat2(self.conv2(out_1))))
    out_3=self.max3(F.relu(self.bat3_b(self.conv3_b(F.relu(self.bat3_a(self.conv3_a(out_2)))))))
    out_4 = self.max4(F.relu(self.bat4_b(self.conv4_b(F.relu(self.bat4_a(self.conv4_a(out_3)))))))
    out_5=self.max5(F.relu(self.bat5_b(self.conv5_b(F.relu(self.bat5_a(self.conv5_a(out_4)))))))
    out_6=self.avgpool(out_5)
    out_6=out_6.reshape(out_6.shape[0],out_6.shape[1])
    out_7=self.linear_layer(out_6)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out_7
