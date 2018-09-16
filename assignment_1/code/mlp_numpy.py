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
    """
    self.n_hidden=n_hidden
    self.n_classes=n_classes
    self.hidden_layers=len(n_hidden)
    self.weights=[]
    self.biases=[]
    self.gamma_arr=[]
    self.beta_arr=[]
    self.n_inputs=n_inputs
    for i in range(0,len(n_hidden)):
        weight_arr=None
        bias_arr=None
        if i==0:
            weight_arr=np.random.randn(self.n_inputs[1],n_hidden[i])*0.01
        else:
            weight_arr=np.random.randn(self.weights[i-1].shape[1],n_hidden[i])*0.01
        self.weights.append(weight_arr)
        bias_arr = np.zeros((1,self.n_hidden[i]))
        self.biases.append(bias_arr)
    self.gamma_arr = np.ones((1, n_hidden[i]))
    self.beta_arr = np.zeros((1, n_hidden[i]))
    bias_arr = np.zeros((1,n_classes[0]))
    self.biases.append(bias_arr)
    weight_arr=np.random.randn(self.n_hidden[self.hidden_layers-1],n_classes[0])*np.sqrt(2/n_classes[0])
    self.weights.append(weight_arr)
    """
    self.modules=[]
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.hidden_layers = len(n_hidden)
    self.n_inputs=n_inputs
    self.modules.append(LinearModule(self.n_inputs[1],n_hidden[0]))
    self.modules.append(ReLUModule())
    self.modules.append(LinearModule(self.n_hidden[0],n_classes[0]))
    self.modules.append(SoftMaxModule())
    self.modules.append(CrossEntropyModule())

    print("Initlialization done")


  def relu(self,x):
      return np.maximum(x,0)

  def derv_relu(self,x,dout,hidden_layer):
      x[x>0]=1.0
      ##std_dev=dout[str(hidden_layer)+"std"]
      ##for i in range(0,100):
      ##    x[:,i]=x[:,i]/std_dev[i]
      return x

  def softmax(self,output):
      print(output.shape)
      for i in range(0,output.shape[0]):
          ma=np.max(output[i,:])
          output[i,:]=np.subtract(output[i,:],ma);
      output=np.exp(output)
      sum=np.sum(output,axis=1)
      for i in range(0,output.shape[0]):
          output[i,:]=output[i,:]/sum[i]
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
    """
    dict_layer={}
    last_layer_input=x;
    dict_layer["0"]=x
    for i in range(0,self.hidden_layers):
        h_in=np.dot(last_layer_input,self.weights[i])
        h_in=np.add(h_in,self.biases[i])
        ##mean_h_in=np.mean(h_in,axis=0)
        ##std_h_in=np.std(h_in,axis=0)
        ##std_h_in=np.multiply(std_h_in,std_h_in)
        ##for t in range(0,100):
        ##    h_in[:,t]-=mean_h_in[t]
        ##    h_in[:,t]=h_in[:,t]/std_h_in[t]
        h_out =self.relu(h_in)
        last_layer_input=h_out
        dict_layer[str(i+1)]=last_layer_input
        ##dict_layer[str(i+1)+"std"]=std_h_in
    mean_h_out = np.mean(h_out, axis=0)
    ##std_h_out = np.std(h_out, axis=0)
    ##std_h_out = np.multiply(std_h_out, std_h_out)
    ##for t in range(0, 100):
    ##    h_out[:, t] -= mean_h_out[t]
    ##    h_out[:, t] = h_out[:, t] / std_h_out[t]
    ##h_out=np.multiply(h_out,self.gamma_arr)
    ##h_out=np.add(h_out,self.beta_arr)
    z_out=np.dot(h_out,self.weights[self.hidden_layers])
    z_out=np.add(z_out,self.biases[self.hidden_layers])
    dict_layer["out"]=z_out
    ##dict_layer["outstd"]=std_h_out
    prob=self.softmax(z_out)
    dict_layer["prob"]=prob
    return prob,dict_layer
    """
    for i in range(0,len(self.modules)-1):
        output=self.modules[i].forward(x)
        x=output
    return x


  def backward(self, dout,y,batch_size):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    """
    delta_w=[]
    delta_b=[]
    delta_last_layer=dout["prob"]-dout["true_prob"]
    ##delta_last_layer=delta_last_layer
    ##delta_last_layer=delta_last_layer.sum(axis=0)
    ##print(delta_last_layer.shape)
    ##print(delta_last_layer.shape,dout[str(self.hidden_layers)].shape,self.weights[self.hidden_layers].shape)
    ##delta_gamma=np.dot(delta_last_layer,self.weights[self.hidden_layers].T)
    ##delta_gamma=np.multiply(delta_gamma,dout[str(self.hidden_layers)])
    ##for i in range(0,100):
    ##    delta_gamma[:,i]=delta_gamma[:,i]/dout["outstd"][i]
    ##delta_gamma=np.sum(delta_gamma,axis=0)
    ##delta_beta=np.dot(delta_last_layer,self.weights[self.hidden_layers].T)
    ##delta_beta=np.sum(delta_beta,axis=0)
    delta_w_last=np.dot(dout[str(self.hidden_layers)].T,delta_last_layer)/batch_size
    delta_b_last=np.sum(delta_last_layer,axis=0)/batch_size
    delta_b_last=delta_b_last.reshape((1,delta_b_last.shape[0]))
    delta_w.append(delta_w_last)
    delta_b.append(delta_b_last)
    ##gamma_by_sigma=self.gamma_arr

    ### For second layer
    layer_index=self.hidden_layers
    while(layer_index>0):
        delta_last_layer = np.dot(delta_last_layer,self.weights[layer_index].T)
        ##delta_last_layer=np.multiply(delta_last_layer,gamma_by_sigma)
        relu_derivative = self.derv_relu(dout[str(layer_index)], dout,layer_index)
        delta_last_layer = np.multiply(delta_last_layer, relu_derivative)
        delta_w_last = np.dot(dout[str(layer_index-1)].T, delta_last_layer)/batch_size
        delta_b_last = np.sum(delta_last_layer, axis=0)/batch_size
        delta_b_last = delta_b_last.reshape((1, delta_b_last.shape[0]))
        delta_w.append(delta_w_last)
        delta_b.append(delta_b_last)
        layer_index=layer_index-1

    return delta_w,delta_b
    ##return delta_w, delta_b, delta_gamma, delta_beta
    """
    le=len(self.modules)-1
    dout = self.modules[le].backward(dout,y)
    le-=1
    while(le>=0):
        new_dout=self.modules[le].backward(dout)
        dout=new_dout
        le=le-1
    return dout


  def sgd(self,lr):
      """"
      delta_index=0
      layer_index=self.hidden_layers

      while(layer_index>=0):
          self.weights[layer_index] -=  (lr * delta_w[delta_index])
          self.biases[layer_index] -= (lr * delta_b[delta_index])
          layer_index=layer_index-1
          delta_index=delta_index+1
      """
      le=len(self.modules)
      for i in range(0,le):
          if(isinstance(self.modules[i],LinearModule)):
              self.modules[i].params['weight']-=lr*self.modules[i].grads['weight']
              self.modules[i].params['bias'] -= lr * self.modules[i].grads['bias']


  def calc_entropy(self,prob,y):
      index = len(self.modules) - 1
      output = (self.modules[index]).forward(prob, y)
      return output
