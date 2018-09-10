"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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
    
    TODO:
    Implement initialization of the network.
    """
    self.n_hidden=n_hidden
    self.n_classes=n_classes
    self.hidden_layers=len(n_hidden)
    self.weights=[]
    self.biases=[]
    self.n_inputs=n_inputs
    for i in range(0,len(n_hidden)):
        weight_arr=None
        bias_arr=None
        if i==0:
            weight_arr=np.random.randn(self.n_inputs[1],n_hidden[i])*np.sqrt(2/n_hidden[i])
        else:
            weight_arr=np.random.randn(self.weights[i-1].shape[1],n_hidden[i])*np.sqrt(2/n_hidden[i])
        self.weights.append(weight_arr)
        bias_arr = np.random.randn(1,self.n_hidden[i])*np.sqrt(2/n_hidden[i])
        self.biases.append(bias_arr)
    bias_arr = np.random.randn(1,n_classes[0])*np.sqrt(2/n_classes[0])
    self.biases.append(bias_arr)
    weight_arr=np.random.randn(self.n_hidden[self.hidden_layers-1],n_classes[0])*np.sqrt(2/n_classes[0])
    self.weights.append(weight_arr)
    print("Initlialization done")


  def relu(self,x):
      return np.maximum(0,x)

  def derv_relu(self,x,dout,hidden_layer):
      x[x>0]=1
      std_dev=dout[str(hidden_layer)+"std"]
      for i in range(0,100):
          x[:,i]=x[:,i]/std_dev[i]
      return x

  def softmax(self,output):
      ##output-=np.max(output,axis=1)
      output=np.exp(np.subtract(output,np.max(output,axis=1)[:,None]))
      sum=np.sum(output,axis=1)
      output=output/sum[:,None]
      return output


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
    dict_layer={}
    last_layer_input=x;
    dict_layer["0"]=x
    for i in range(0,len(self.weights)-1):
        h_in=np.dot(last_layer_input,self.weights[i])
        h_in=np.add(h_in,self.biases[i])
        mean_h_in=np.mean(h_in,axis=0)
        std_h_in=np.std(h_in,axis=0)
        std_h_in=np.multiply(std_h_in,std_h_in)
        for t in range(0,100):
            h_in[:,t]-=mean_h_in[t]
            h_in[:,t]=h_in[:,t]/std_h_in[t]
        h_out =self.relu(h_in)
        last_layer_input=h_out
        dict_layer[str(i+1)]=last_layer_input
        dict_layer[str(i+1)+"std"]=std_h_in
    h_out=np.dot(h_out,self.weights[self.hidden_layers])
    h_out=h_out+self.biases[self.hidden_layers]
    dict_layer["out"]=h_out
    prob=self.softmax(h_out)
    dict_layer["prob"]=prob
    return prob,dict_layer

  def backward(self, dout,batch_size):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    delta_w=[]
    delta_b=[]
    delta_last_layer=dout["prob"]-dout["true_prob"]
    delta_last_layer=delta_last_layer/batch_size;
    ##softmax_delta=np.zeros(delta_last_layer.shape)
    ##targets_index = np.argmax(dout["true_prob"], axis=1)
    ##for i in range(0,batch_size):
    ##    softmax_delta[i][targets_index[i]]=(dout["prob"][i][targets_index[i]]) * (1 - dout["prob"][i][targets_index[i]])
    ##delta_last_layer=np.multiply(delta_last_layer,softmax_delta)
    delta_w_last=np.dot(dout[str(self.hidden_layers)].T,delta_last_layer)
    delta_b_last=np.sum(delta_last_layer,axis=0)
    delta_b_last=delta_b_last.reshape((1,delta_b_last.shape[0]))
    delta_w.append(delta_w_last)
    delta_b.append(delta_b_last)
    ### For second layer
    layer_index=self.hidden_layers
    while(layer_index>0):
        delta_last_layer = np.dot(delta_last_layer, self.weights[layer_index].T)
        relu_derivative = self.derv_relu(dout[str(layer_index)], dout,layer_index)
        delta_last_layer = np.multiply(delta_last_layer, relu_derivative)
        delta_w_last = np.dot(dout[str(layer_index-1)].T, delta_last_layer)
        delta_b_last = np.sum(delta_last_layer, axis=0)
        delta_b_last = delta_b_last.reshape((1, delta_b_last.shape[0]))
        delta_w.append(delta_w_last)
        delta_b.append(delta_b_last)
        layer_index=layer_index-1

    return delta_w,delta_b


  def sgd(self,delta_w,delta_b,lr):
      delta_index=0
      layer_index=self.hidden_layers
      while(layer_index>=0):
          self.weights[layer_index] -=  (lr * delta_w[delta_index])
          self.biases[layer_index] -= (lr * delta_b[delta_index])
          layer_index=layer_index-1
          delta_index=delta_index+1


  def calc_entropy(self,dic):
      true_prob=dic["true_prob"]
      true_prob=-true_prob
      prob=dic["prob"]
      prob=np.log2(prob)
      out=np.multiply(true_prob,prob)
      sum=np.sum(out)

      return sum/200