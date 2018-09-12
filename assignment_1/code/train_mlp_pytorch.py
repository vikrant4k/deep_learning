"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
from mlp_pytorch import TwoLayerNet
import cifar10_utils
from GlobalVariables import glb
import torch.optim as optim
import torch.nn as nn
import torch
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
  accur = matches / BATCH_SIZE_DEFAULT

  return accur

  return accur

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  glb.net=MLP(1,1,1)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(glb.net.parameters(), lr=0.002, momentum=0.0)
  cifar10 = cifar10_utils.get_cifar10(
      '/home/vik1/Downloads/subj/deep_learning/uvadlc_practicals_2018/assignment_1/code/cifar10/cifar-10-batches-py')
  x, y = cifar10['train'].next_batch(50000)
  x_red = x[:, 0, :, :]
  x_green = x[:, 1, :, :]
  x_blue = x[:, 2, :, :]
  glb.mean_red = np.mean(x_red)
  glb.mean_blue = np.mean(x_blue)
  glb.mean_green = np.mean(x_green)
  glb.std_red = np.std(x_red)
  glb.std_green = np.std(x_green)
  glb.std_blue = np.std(x_blue)
  entropy_sum_list = []
  for epoch in range(0, 40):
      entropies = []
      cifar10 = cifar10_utils.get_cifar10(
          '/home/vik1/Downloads/subj/deep_learning/uvadlc_practicals_2018/assignment_1/code/cifar10/cifar-10-batches-py')
      running_loss=0
      for i in range(0, 250):
          x, y = cifar10['train'].next_batch(200)
          ##x[:, 0, :, :] = (x[:, 0, :, :] - glb.mean_red) / (glb.std_red * glb.std_red)
          ##x[:, 1, :, :] = (x[:, 1, :, :] - glb.mean_green) / (glb.std_green * glb.std_green)
          ##x[:, 2, :, :] = (x[:, 2, :, :] - glb.mean_blue) / (glb.std_blue * glb.std_blue)
          x = x.reshape((200, 32 * 32 * 3))
          x = torch.from_numpy(x)
          y=torch.from_numpy(y)
          y=y.type(torch.LongTensor)
          optimizer.zero_grad()
          # forward + backward + optimize
          outputs = glb.net(x)
          loss = criterion(outputs, torch.max(y, 1)[1])
          loss.backward()
          optimizer.step()
          # print statistics
          running_loss += loss.item()
      print(running_loss)

def test(data_type,num_times):
    cifar10 = cifar10_utils.get_cifar10(
        '/home/vik1/Downloads/subj/deep_learning/uvadlc_practicals_2018/assignment_1/code/cifar10/cifar-10-batches-py')
    accu=[]
    for i in range(0, num_times):
        x, y = cifar10[data_type].next_batch(BATCH_SIZE_DEFAULT)
        x[:, 0, :, :] = (x[:, 0, :, :] - glb.mean_red) / (glb.std_red * glb.std_red)
        x[:, 1, :, :] = (x[:, 1, :, :] - glb.mean_green) / (glb.std_green * glb.std_green)
        x[:, 2, :, :] = (x[:, 2, :, :] - glb.mean_blue) / (glb.std_blue * glb.std_blue)
        x = x.reshape((200, 32 * 32 * 3))
        x = torch.from_numpy(x)
        output = glb.net(x)
        output=output.detach().numpy()
        print(output.shape,y.shape)
        acc=accuracy(output,y)
        print(acc)
        accu.append(acc)
    full_acc=sum(accu)/len(accu)
    print("Full Accuracy",full_acc)




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
  test("test",50)

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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