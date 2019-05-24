import numpy as np
import cv2
import pickle
from utils import *

class myCNN(object):
  """
  A simple-implemented convolutional network with the following architecture:

  conv -> relu -> 2x2 max pool -> FC -> relu -> FC -> softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.

  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength

    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn((H / 2)*(W / 2)*num_filters, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(np.float32)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    conv_forward_out_1, cache_forward_1 = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    affine_forward_out_2, cache_forward_2 = affine_forward(conv_forward_out_1, self.params['W2'], self.params['b2'])
    affine_relu_2, cache_relu_2 = relu_forward(affine_forward_out_2)
    scores, cache_forward_3 = affine_forward(affine_relu_2, self.params['W3'], self.params['b3'])

    if y is None:
      return scores

    loss, grads = 0, {}
    loss, dout = softmax_loss(scores, y)

    # Add regularization
    loss += self.reg * 0.5 * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2))

    dX3, grads['W3'], grads['b3'] = affine_backward(dout, cache_forward_3)
    dX2 = relu_backward(dX3, cache_relu_2)
    dX2, grads['W2'], grads['b2'] = affine_backward(dX2, cache_forward_2)
    dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)

    grads['W3'] = grads['W3'] + self.reg * self.params['W3']
    grads['W2'] = grads['W2'] + self.reg * self.params['W2']
    grads['W1'] = grads['W1'] + self.reg * self.params['W1']

    return loss, grads
