"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
from GlobalVariables import glb
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.functional as F

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """
  accur = 0.0
  ##print(targets.shape)
  targets_index = np.argmax(targets, axis=1)
  predictions_index = np.argmax(predictions, axis=1)
  value_arr = predictions_index - targets_index
  matches = (value_arr == 0).sum()
  accur = matches / len(targets_index)

  return accur

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  glb.net = ConvNet( [100], 10)
  glb.net.cuda()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(glb.net.parameters(), lr=LEARNING_RATE_DEFAULT)
  ##optimizer = optim.Adagrad(glb.net.parameters(),lr=0.01,weight_decay=0.001)
  entropy_sum_list = []
  entropies = []
  accuracies = []
  cifar10 = cifar10_utils.get_cifar10(
    '/home/vik1/Downloads/subj/deep_learning/uvadlc_practicals_2018/assignment_1/code/cifar10/cifar-10-batches-py')
  running_loss = 0
  for i in range(1, MAX_STEPS_DEFAULT+1):
    x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)
    ##x = x.reshape((BATCH_SIZE_DEFAULT, 32 * 32 * 3))
    x = torch.from_numpy(x)
    x = x.cuda()
    y = torch.from_numpy(y)
    y = y.type(torch.LongTensor)
    y = y.cuda()
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = glb.net(x)
    loss = criterion(outputs, torch.max(y, 1)[1])
    loss.backward()
    optimizer.step()
    # print statistics
    running_loss += loss.item()
    if (i % EVAL_FREQ_DEFAULT == 0):
      acc = test("test", 312)
      print(i, running_loss, acc)
      entropy_sum_list.append(running_loss)
      running_loss = 0
      accuracies.append(acc)
  print(entropy_sum_list)
  print(accuracies)
  plt.plot(entropy_sum_list, 'r-')
  plt.show()
  plt.close()
  plt.plot(accuracies, 'r-')
  plt.show()
  print("done")
  ########################
  # END OF YOUR CODE    #
  #######################

def test(data_type,num_times):
    cifar10 = cifar10_utils.get_cifar10(
        '/home/vik1/Downloads/subj/deep_learning/uvadlc_practicals_2018/assignment_1/code/cifar10/cifar-10-batches-py')
    accu=[]
    for i in range(0, num_times):
        batch_size=BATCH_SIZE_DEFAULT
        if(i==(num_times-1)):
            batch_size=batch_size/2
        x, y = cifar10[data_type].next_batch(BATCH_SIZE_DEFAULT)
        ##x = x.reshape((BATCH_SIZE_DEFAULT, 32 * 32 * 3))
        x = torch.from_numpy(x)
        x=x.cuda()
        output = glb.net(x)
        softmax=torch.nn.Softmax(1)
        x=softmax(output)
        output=output.cpu()
        output=output.detach().numpy()
        acc=accuracy(output,y)
        accu.append(acc)
    full_acc=sum(accu)/len(accu)
    return  full_acc

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()