"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    weight_arr = np.random.randn(in_features, out_features) * 0.0001
    bias_arr = np.zeros((1, out_features))
    self.params['weight']=weight_arr
    self.params['bias']=bias_arr
    self.grads['weight']=np.zeros((in_features, out_features))
    self.grads['bias']=np.zeros((1,out_features))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    self.x=x
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    h_in = np.dot(x, self.params['weight'])
    h_in = np.add(h_in,self.params['bias'])
    ########################
    # END OF YOUR CODE    #
    #######################
    self.out=h_in

    return h_in

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #delta_last_layer = np.multiply(delta_last_layer, relu_derivative)
    delta_w=np.dot(self.x.T,dout)
    delta_b=np.sum(dout, axis=1)
    self.grads['weight']=delta_w
    self.grads['bias']=delta_b
    dout = np.dot(dout, self.params['weight'].T)
    ########################
    # END OF YOUR CODE    #
    #######################
    return dout

class ReLUModule(object):
  """
  ReLU activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    self.x=np.maximum(x,0)
    return self.x
    ########################
    # END OF YOUR CODE    #
    #######################


  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    derv_relu=self.x.copy()
    derv_relu[derv_relu>0]=1.0
    derv_relu[derv_relu<=0]=0
    derv_relu=np.multiply(dout,derv_relu)
    ########################
    # END OF YOUR CODE    #
    #######################    

    return derv_relu

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, output):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for i in range(0, output.shape[1]):
        ma = np.max(output[:, i])
        output[:, i] = np.subtract(output[:,i], ma);
    output = np.exp(output)
    sum = np.sum(output, axis=0)
    for i in range(0, output.shape[1]):
        output[:,i] = output[:,i]/sum[i]

    ########################
    # END OF YOUR CODE    #
    #######################
    self.output=output
    return output

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    delta=np.zeros(dout.shape)
    for i in range(0,dout.shape[1]):
        dx = self.output[:,i].reshape(-1, 1)
        dx=np.diagflat(dx) - np.dot(dx, dx.T)
        dy=np.dot(dx,dout[:,i])
        delta[:,i]=dy

    ########################
    # END OF YOUR CODE    #
    #######################
    ##dx=np.multiply(dx,dout)
    return delta

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    N = y.shape[0]
    true_prob = y
    true_prob = -true_prob
    prob = x
    prob = np.log(prob)
    out = np.multiply(true_prob, prob)
    sum = np.sum(out)/N
    ########################
    # END OF YOUR CODE    #
    #######################

    return sum

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    y=-y
    dx=np.zeros(y.shape)
    for i in range(0,len(y)):
        dx[i,:]=y[i,:]/x[i,:]
    ########################
    # END OF YOUR CODE    #
    #######################
    dx=dx/len(y)
    return dx
